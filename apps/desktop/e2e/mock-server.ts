/**
 * Minimal OpenAI-compatible mock inference server for E2E tests.
 *
 * Implements just enough of the /v1/* surface for `hermes serve` to resolve a
 * provider, list models, and stream a canned chat completion back to the
 * desktop app — without any real LLM.
 *
 * Endpoints:
 *   GET  /v1/models             → { data: [{ id, ... }] }
 *   POST /v1/chat/completions   → streaming (SSE) or non-streaming response
 *
 * The canned response is a short, deterministic assistant message. If the
 * request advertises tools and fewer than MAX_TOOL_CALLS `tool` messages
 * have come back yet, the server instead makes another tool call
 * (preferring a `todo` tool, else the first tool offered) with minimal
 * arguments derived from its JSON schema, so an E2E test can observe several
 * round trips within a single agent turn. Once enough tool results have
 * round-tripped back as `role: 'tool'` messages, the server falls through to
 * the canned reply below, ending the turn.
 */

import http from 'node:http'

/** A canned assistant reply used for every chat completion request. */
const CANNED_REPLY = 'Hello from the mock inference server! The full boot chain is working.'

/** More than one: several round trips in a single turn let a test watch mid-turn state change. */
const MAX_TOOL_CALLS = 3

/**
 * A tool name the agent cannot possibly have.
 *
 * Naming one is how this mock produces a round that finishes a real API call —
 * usage and all — without any tool ever running: the agent rejects the name
 * before it reaches the executor, appends a result saying so, and keeps going.
 * That is the one round shape where reporting occupancy per API response
 * differs from reporting it per completed tool.
 *
 * `zz_`-prefixed and deliberately unlike any real tool: the agent repairs a
 * near-miss name by fuzzy match, and a name that got repaired into a REAL tool
 * would execute, complete, and quietly turn this back into an ordinary round.
 */
export const PROBE_TOOL_NAME = 'zz_e2e_probe_missing'
const PROBE_CALL_ID = 'mock-probe-1'

/** Opt-in (MOCK_TOOL_FREE_ROUND): off by default so the other specs are untouched. */
function toolFreeRoundEnabled(): boolean {
  const raw = process.env.MOCK_TOOL_FREE_ROUND

  return raw === '1' || raw === 'true'
}

let probeRoundPromptTokens: number | null = null

/**
 * The prompt count served on the tool-free round, or null if it hasn't run.
 *
 * Read by the spec instead of hardcoding a number: the usage schedule is a
 * function of conversation length, so a literal in the test would drift the
 * moment the system prompt or tool list changes size.
 */
export function probeRoundPromptCount(): number | null {
  return probeRoundPromptTokens
}

export function resetProbeRoundPromptCount(): void {
  probeRoundPromptTokens = null
}

/**
 * Pause before answering a tool-call request, in ms (MOCK_TOOL_CALL_DELAY_MS).
 *
 * Off by default so the ordinary specs stay fast. A test that has to observe
 * state changing *during* a turn needs that turn to outlast its polling
 * interval — otherwise the whole turn lands between two samples and the test
 * would pass whether or not the behaviour it checks is there.
 */
function toolCallDelayMs(): number {
  const raw = Number(process.env.MOCK_TOOL_CALL_DELAY_MS)

  return Number.isFinite(raw) && raw > 0 ? raw : 0
}

/**
 * Usage for one response, sized from the conversation so far.
 *
 * A real provider reports a prompt count that grows every round trip, because
 * each round appends the last tool's result to the prompt. That growth is the
 * only thing a live context meter has to read, so a mock that answers with a
 * constant — or with nothing — cannot tell a working meter from a frozen one.
 */
function usageFor(messages: any[]): { completion_tokens: number; prompt_tokens: number; total_tokens: number } {
  const promptTokens = 12_000 + messages.length * 1_500
  const completionTokens = 120

  return {
    completion_tokens: completionTokens,
    prompt_tokens: promptTokens,
    total_tokens: promptTokens + completionTokens,
  }
}

/**
 * The final SSE frame of a stream, carrying usage.
 *
 * Hermes requests `stream_options: {include_usage: true}` on every streaming
 * call, and an OpenAI-compatible server answers it with one last chunk whose
 * `choices` are empty and whose `usage` covers the whole response. Without this
 * the stream looks usage-less, the usage recorder never runs, and everything
 * downstream of it is untested.
 */
function usageChunk(model: string, messages: any[]): string {
  return `data: ${JSON.stringify({
    id: 'mock-completion',
    object: 'chat.completion.chunk',
    created: 0,
    model,
    choices: [],
    usage: usageFor(messages),
  })}\n\n`
}

/** A trivially valid value for a JSON schema property, honoring `enum` and `type`. */
function trivialValueFor(propSchema: any): unknown {
  if (Array.isArray(propSchema?.enum) && propSchema.enum.length > 0) {
    return propSchema.enum[0]
  }

  switch (propSchema?.type) {
    case 'string':
      return 'e2e'
    case 'number':
    case 'integer':
      return 1
    case 'boolean':
      return true
    case 'array':
      return []
    case 'object':
      return {}
    default:
      return 'e2e'
  }
}

/** Build minimal valid arguments for a tool call from its JSON schema. */
function buildToolArguments(schema: any): Record<string, unknown> {
  const properties = schema?.properties || {}
  const required: string[] = Array.isArray(schema?.required) ? schema.required : []
  const args: Record<string, unknown> = {}

  for (const key of required) {
    args[key] = trivialValueFor(properties[key])
  }

  return args
}

/**
 * Start the mock server on an ephemeral port.
 *
 * @returns a handle with `port`, `url`, and `close()`.
 */
export function startMockServer(): Promise<{ port: number; url: string; close: () => Promise<void> }> {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      // CORS headers — the Electron renderer doesn't need them, but they
      // don't hurt and make the server usable from a browser context too.
      res.setHeader('Access-Control-Allow-Origin', '*')
      res.setHeader('Access-Control-Allow-Headers', '*')
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

      if (req.method === 'OPTIONS') {
        res.writeHead(204)
        res.end()
        return
      }

      // GET /v1/models — return a single fake model.
      if (req.method === 'GET' && req.url === '/v1/models') {
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(
          JSON.stringify({
            object: 'list',
            data: [
              {
                id: 'mock-model',
                object: 'model',
                created: 0,
                owned_by: 'mock',
              },
            ],
          }),
        )
        return
      }

      // POST /v1/chat/completions — return a canned response.
      if (req.method === 'POST' && req.url?.startsWith('/v1/chat/completions')) {
        let body = ''

        req.on('data', (chunk: Buffer) => {
          body += chunk.toString()
        })

        req.on('end', async () => {
          let parsed: any = {}

          try {
            parsed = JSON.parse(body)
          } catch {
            // malformed JSON — treat as non-streaming with defaults
          }

          const stream = parsed.stream === true
          const model = parsed.model || 'mock-model'

          const messages = Array.isArray(parsed.messages) ? parsed.messages : []
          const tools = Array.isArray(parsed.tools) ? parsed.tools : []

          // Count only THIS turn's rounds, not the whole conversation. Counting
          // every `tool` message ever sent meant a second turn in the same chat
          // started already over budget and collapsed to a single round — so a
          // test that meant to watch a multi-round turn silently watched a
          // one-round one.
          const lastUserIndex = messages.map((message: any) => message?.role).lastIndexOf('user')
          const turnToolResults = messages
            .slice(lastUserIndex + 1)
            .filter((message: any) => message?.role === 'tool').length

          if (toolFreeRoundEnabled() && tools.some((t: any) => t?.function?.name === PROBE_TOOL_NAME)) {
            // The probe only works while the agent has no such tool. If one ever
            // appears under this name it would really execute and complete, and
            // the round would quietly stop being tool-free — the test would go
            // on passing while testing nothing. Answer with an error instead of
            // throwing: this runs inside an async request handler, where a throw
            // becomes an unhandled rejection in the test process rather than a
            // legible failure, and leaves the request hanging.
            res.writeHead(500, { 'Content-Type': 'application/json' })
            res.end(
              JSON.stringify({
                error: { message: `${PROBE_TOOL_NAME} is a real tool — pick a name the agent cannot have` },
              }),
            )
            return
          }

          // One round mid-turn names a tool that does not exist. It is a full
          // API round trip that reports usage and completes no tool.
          const shouldProbe = toolFreeRoundEnabled() && turnToolResults === 1 && tools.length > 0
          const shouldCallTool = shouldProbe || (turnToolResults < MAX_TOOL_CALLS && tools.length > 0)

          if (shouldCallTool) {
            const tool = tools.find((t: any) => t?.function?.name === 'todo') || tools[0]
            const toolName = shouldProbe ? PROBE_TOOL_NAME : tool?.function?.name || 'unknown'
            const toolArgs = shouldProbe ? {} : buildToolArguments(tool?.function?.parameters)
            const toolCallId = shouldProbe ? PROBE_CALL_ID : `mock-tool-call-${turnToolResults + 1}`

            if (shouldProbe) {
              probeRoundPromptTokens = usageFor(messages).prompt_tokens
            }
            const argsJson = JSON.stringify(toolArgs)
            const delayMs = toolCallDelayMs()

            if (delayMs > 0) {
              await new Promise((resolveDelay) => setTimeout(resolveDelay, delayMs))
            }

            if (stream) {
              res.writeHead(200, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                Connection: 'keep-alive',
              })

              // One chunk carries the whole tool call, then a final chunk
              // closes the turn with finish_reason: 'tool_calls'.
              res.write(
                `data: ${JSON.stringify({
                  id: 'mock-completion',
                  object: 'chat.completion.chunk',
                  created: 0,
                  model,
                  choices: [
                    {
                      index: 0,
                      delta: {
                        tool_calls: [
                          {
                            index: 0,
                            id: toolCallId,
                            type: 'function',
                            function: { name: toolName, arguments: argsJson },
                          },
                        ],
                      },
                      finish_reason: null,
                    },
                  ],
                })}\n\n`,
              )
              res.write(
                `data: ${JSON.stringify({
                  id: 'mock-completion',
                  object: 'chat.completion.chunk',
                  created: 0,
                  model,
                  choices: [
                    {
                      index: 0,
                      delta: {},
                      finish_reason: 'tool_calls',
                    },
                  ],
                })}\n\n`,
              )
              res.write(usageChunk(model, messages))
              res.write('data: [DONE]\n\n')
              res.end()
            } else {
              res.writeHead(200, { 'Content-Type': 'application/json' })
              res.end(
                JSON.stringify({
                  id: 'mock-completion',
                  object: 'chat.completion',
                  created: 0,
                  model,
                  choices: [
                    {
                      index: 0,
                      message: {
                        role: 'assistant',
                        content: null,
                        tool_calls: [
                          {
                            id: toolCallId,
                            type: 'function',
                            function: { name: toolName, arguments: argsJson },
                          },
                        ],
                      },
                      finish_reason: 'tool_calls',
                    },
                  ],
                  usage: usageFor(messages),
                }),
              )
            }
            return
          }

          if (stream) {
            res.writeHead(200, {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
              Connection: 'keep-alive',
            })

            // Send the content in a few chunks to simulate streaming.
            const words = CANNED_REPLY.split(' ')
            let i = 0

            const sendChunk = () => {
              if (i >= words.length) {
                // Final chunk with finish_reason
                res.write(
                  `data: ${JSON.stringify({
                    id: 'mock-completion',
                    object: 'chat.completion.chunk',
                    created: 0,
                    model,
                    choices: [
                      {
                        index: 0,
                        delta: {},
                        finish_reason: 'stop',
                      },
                    ],
                  })}\n\n`,
                )
                res.write(usageChunk(model, messages))
                res.write('data: [DONE]\n\n')
                res.end()
                return
              }

              const word = i === 0 ? words[i] : ' ' + words[i]
              res.write(
                `data: ${JSON.stringify({
                  id: 'mock-completion',
                  object: 'chat.completion.chunk',
                  created: 0,
                  model,
                  choices: [
                    {
                      index: 0,
                      delta: { content: word },
                      finish_reason: null,
                    },
                  ],
                })}\n\n`,
              )
              i++
              // Small delay between chunks to simulate real streaming.
              setTimeout(sendChunk, 20)
            }

            sendChunk()
          } else {
            // Non-streaming response
            res.writeHead(200, { 'Content-Type': 'application/json' })
            res.end(
              JSON.stringify({
                id: 'mock-completion',
                object: 'chat.completion',
                created: 0,
                model,
                choices: [
                  {
                    index: 0,
                    message: { role: 'assistant', content: CANNED_REPLY },
                    finish_reason: 'stop',
                  },
                ],
                usage: usageFor(messages),
              }),
            )
          }
        })

        req.on('error', () => {
          res.writeHead(400)
          res.end('Bad request')
        })
        return
      }

      // Fallback — 404 for anything else
      res.writeHead(404, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: 'Not found' }))
    })

    server.on('error', reject)

    server.listen(0, '127.0.0.1', () => {
      const addr = server.address()
      if (addr === null || typeof addr === 'string') {
        reject(new Error('Failed to get server address'))
        return
      }

      const port = addr.port
      const url = `http://127.0.0.1:${port}`

      resolve({
        port,
        url,
        close: () =>
          new Promise((resolveClose, rejectClose) => {
            server.close((err) => {
              if (err) {
                rejectClose(err)
              } else {
                resolveClose()
              }
            })
          }),
      })
    })
  })
}

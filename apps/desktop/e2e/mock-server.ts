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

import fs from 'node:fs'
import http from 'node:http'
import type { ServerResponse } from 'node:http'
import os from 'node:os'
import nodePath from 'node:path'

/** A canned assistant reply used for every chat completion request. */
export const MOCK_REPLY = 'Hello from the mock inference server! The full boot chain is working.'

export interface MockServerOptions {
  /** Pause the matching stream after its first token for session-switch E2E coverage. */
  holdFirstStreamForPrompt?: string
/** Pause the first completion whose request JSON contains this text. */
holdFirstCompletionContaining?: string
/** Absolute sandbox path written by the verify-on-stop scripted tool call. */
verificationWritePath?: string
/**
 * Sentinel path that ends the E2E_SIDEBAR_CROSS background process.
 *
 * Without it that process is a bare `sleep 5`, which races the agent turn and
 * the 4s auto-dismiss linger — see `createBackgroundReleaseHandle`. Pass a
 * handle's `path` to let the test decide when the process exits.
 */
backgroundReleasePath?: string
}

export interface MockServer {
  port: number
  url: string
  receivedPrompts: string[]
  waitForHeldStream: () => Promise<void>
  waitForHeldCompletion: () => Promise<void>
  releaseHeldStream: () => void
  heldCompletionCount: () => number
  close: () => Promise<void>
}

// ─── Multi-turn interim script ─────────────────────────────────────────
//
// When the user's message contains the trigger keyword, the mock server
// walks through a scripted sequence of responses that exercise the
// interim-assistant-message fix (#65919) across several patterns:
//
//   1. text + single tool_call  → should produce an interim message
//   2. text + single tool_call  → another interim message
//   3. no text + tool_call       → NO interim (no visible text alongside tools)
//   4. text + single tool_call  → another interim message
//   5. final answer (stop)      → message.complete, different from all interims
//
// Each "turn" is one API call. The agent executes the tool after each
// tool_calls response, then re-calls the API, advancing to the next turn.

export interface ScriptedTurn {
  /** Assistant text content to stream. Empty string = no visible text. */
  text: string
  /** Tool calls to emit. Empty array = final turn (finish_reason: stop). */
  toolCalls?: Array<{
    name: string
    args: Record<string, unknown>
  }>
}

const INTERIM_SCRIPT: ScriptedTurn[] = [
  {
    text: 'Let me start by planning the approach.',
    toolCalls: [{ name: 'todo', args: { todos: [{ id: '1', content: 'Plan', status: 'in_progress' }] } }],
  },
  {
    text: 'Now checking the details before answering.',
    toolCalls: [{ name: 'todo', args: { todos: [{ id: '2', content: 'Check details', status: 'in_progress' }] } }],
  },
  {
    // No visible text alongside this tool call — should NOT produce an
    // interim message. The agent fires _emit_interim_assistant_message
    // but _interim_assistant_visible_text returns "" so it's a no-op.
    text: '',
    toolCalls: [{ name: 'todo', args: { todos: [{ id: '3', content: 'Silent step', status: 'completed' }] } }],
  },
  {
    text: 'Found something interesting worth noting.',
    toolCalls: [{ name: 'todo', args: { todos: [{ id: '4', content: 'Note finding', status: 'completed' }] } }],
  },
  {
    // Final answer — different from all interim texts.
    text: 'All done! Here is the complete summary of what I found.',
  },
]

/** Per-server request counter so we can walk through the script turns. */
let _scriptIndex = 0

/** Per-server counter for the sidebar-states script (independent from _scriptIndex). */
let _sidebarScriptIndex = 0

/** Per-server counter for the cross-session sidebar script. */
let _sidebarCrossIndex = 0

/** Per-server counter for the queue-stop script. */
let _queueStopIndex = 0

/** Per-server counter for the correction/session-switch script. */
let _correctionSwitchIndex = 0

/** Per-server counter for the verify-on-stop script. */
let _verificationStopIndex = 0

/** User messages received by the mock, for E2E assertions on real submits. */
const _receivedUserTexts: string[] = []

/** Reset the script indices (called between tests via restartMockServer). */
function resetScriptIndex(): void {
  _scriptIndex = 0
  _sidebarScriptIndex = 0
  _sidebarCrossIndex = 0
  _queueStopIndex = 0
  _correctionSwitchIndex = 0
  _verificationStopIndex = 0
  _receivedUserTexts.length = 0
}

/** Return the user prompts the real backend submitted to this mock server. */
export function receivedUserTexts(): readonly string[] {
  return _receivedUserTexts
}

// ─── Sidebar-states script ─────────────────────────────────────────────
//
// A separate trigger (E2E_SIDEBAR_TRIGGER) exercises the desktop sidebar's
// background-process and subagent states. The mock returns tool_calls that
// the agent executes for real — `terminal(background=true)` spawns a real
// (but trivial) background process, and `delegate_task` spawns a real
// subagent that calls the mock server and gets the canned reply.
//
// Turn 1: text + terminal(bg=true) + delegate_task → tools execute
// Turn 2: final answer → message.complete, dot transitions

const SIDEBAR_SCRIPT: ScriptedTurn[] = [
  {
    text: 'Let me run a background task and delegate some work.',
    toolCalls: [
      {
        name: 'terminal',
        args: {
          command: 'echo "background process output" && sleep 1 && echo "done"',
          background: true,
          notify_on_complete: true,
        },
      },
      {
        name: 'delegate_task',
        args: {
          goal: 'Summarize the test results',
          context: 'This is a test subagent for the sidebar states E2E test.',
        },
      },
    ],
  },
  {
    text: 'All tasks complete. The background process finished and the subagent returned its summary.',
  },
]

// ─── Sidebar cross-session script ──────────────────────────────────────
//
// E2E_SIDEBAR_CROSS starts a long background process plus a subagent so the
// tests can:
//   1. See the background dot while the subagent runs.
//   2. Open a different session and see session A's dot transition to
//      "finished unread" when the background process completes.
//
// The background process must outlive the agent turn — the whole point is a
// dot that is still "running" after the final answer lands. A fixed `sleep`
// cannot guarantee that: on a loaded CI runner the turn (two model round
// trips + a real subagent delegation) can take longer than the sleep, the
// process exits early, the 4s success linger elapses, and the dot is gone
// before the test looks. That is a wall-clock race between three independent
// timers, and it made this the flakiest spec in the suite.
//
// When `backgroundReleasePath` is set the process instead blocks until the
// test creates that sentinel file, so the test — not the clock — decides when
// the dot clears. `sleep 5` remains the fallback for callers that don't pass
// a handle.
function sidebarCrossBgCommand(releasePath?: string): string {
  if (!releasePath) {
    return 'echo "long bg output" && sleep 5 && echo "finished"'
  }
  // Bounded wait (60s): if a test forgets to release (or crashes mid-way),
  // the process still exits instead of hanging the worker until the suite
  // times out.
  const quoted = JSON.stringify(releasePath)
  return [
    'echo "long bg output"',
    `for _ in $(seq 1 600); do [ -e ${quoted} ] && break; sleep 0.1; done`,
    'echo "finished"',
  ].join(' && ')
}

function sidebarCrossScript(releasePath?: string): ScriptedTurn[] {
  return [
    {
      text: 'Starting a long background task and delegating work.',
      toolCalls: [
        {
          name: 'terminal',
          args: {
            command: sidebarCrossBgCommand(releasePath),
            background: true,
            notify_on_complete: true,
          },
        },
        {
          name: 'delegate_task',
          args: {
            goal: 'Analyze cross-session state',
            context: 'Testing that the background dot updates across sessions.',
          },
        },
      ],
    },
    {
      text: 'Both tasks are running in the background now.',
    },
  ]
}

const SIDEBAR_CROSS_SCRIPT: ScriptedTurn[] = sidebarCrossScript()

const QUEUE_STOP_SCRIPT: ScriptedTurn[] = [
  {
    text: 'Starting a task that will keep this turn active.',
    toolCalls: [{ name: 'clarify', args: { question: 'Keep working?', choices: ['Yes', 'No'] } }],
  },
  { text: 'The paused task completed.' },
]

// The reported correction arrived while a foreground tool was still running.
// Keep that boundary open long enough for the renderer to redirect the turn,
// then let the next model request complete normally.
const CORRECTION_SWITCH_SCRIPT: ScriptedTurn[] = [
  {
    text: 'Checking the long-running task before I continue.',
    toolCalls: [{ name: 'terminal', args: { command: 'sleep 5' } }],
  },
  { text: 'The corrected task finished.' },
]

export const CORRECTION_SWITCH_TRIGGER = 'E2E_CORRECTION_SWITCH_TRIGGER'

/**
 * Drives a real code edit followed by two finish attempts. Hermes should add
 * its synthetic verify-on-stop continuation after each finish attempt until
 * the bounded verifier gives up. The mock's request capture proves the nudge
 * reached the model; desktop must never render it as chat content.
 */
function verificationStopScript(writePath: string): ScriptedTurn[] {
  return [
  {
    text: 'I will make the requested code change.',
    toolCalls: [{
      name: 'write_file',
      args: {
        path: writePath,
        content: 'def changed_by_e2e():\n    return "changed"\n',
      },
    }],
  },
  { text: 'The code edit is complete.' },
  { text: 'I cannot provide fresh verification evidence for that edit.' },
  ]
}

export const VERIFICATION_STOP_TRIGGER = 'E2E_VERIFY_ON_STOP_TRIGGER'
export const VERIFICATION_STOP_TEXT = 'I cannot provide fresh verification evidence for that edit.'

/**
 * A marker that makes the mock emit a real blocking clarify tool call. Tests
 * use it to hold a turn open while exercising busy-composer interactions.
 */
export const BLOCKING_CLARIFY_TRIGGER = 'E2E_BLOCKING_CLARIFY_TRIGGER'
export const BLOCKING_CLARIFY_QUESTION = 'Keep this test turn running?'

const BLOCKING_CLARIFY_TURN: ScriptedTurn = {
  text: '',
  toolCalls: [{ name: 'clarify', args: { question: BLOCKING_CLARIFY_QUESTION, choices: ['Yes', 'No'] } }],
}

function includesBlockingClarifyTrigger(value: unknown): boolean {
  if (typeof value === 'string') {
    return value.includes(BLOCKING_CLARIFY_TRIGGER)
  }

  if (Array.isArray(value)) {
    return value.some(includesBlockingClarifyTrigger)
  }

  if (value && typeof value === 'object') {
    return Object.values(value).some(includesBlockingClarifyTrigger)
  }

  return false
}

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
 * @returns a handle with `port`, `url`, received user prompts, and `close()`.
 */
export function startMockServer(options: MockServerOptions = {}): Promise<MockServer> {
  return new Promise((resolve, reject) => {
    const receivedPrompts: string[] = []
    let resolveHeldStreamStarted: (() => void) | null = null
    let releaseHeldStream: (() => void) | null = null
    let heldCompletionCount = 0
    const heldStreamStarted = new Promise<void>(resolveHeld => {
      resolveHeldStreamStarted = resolveHeld
    })
    const heldStreamReleased = new Promise<void>(resolveRelease => {
      releaseHeldStream = resolveRelease
    })
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

          const lastUserMessage = [...(parsed.messages ?? [])]
            .reverse()
            .find((message: { role?: unknown }) => message?.role === 'user')

          if (typeof lastUserMessage?.content === 'string') {
            receivedPrompts.push(lastUserMessage.content)
          }

          const stream = parsed.stream === true
          const model = parsed.model || 'mock-model'
          const holdThisCompletion = Boolean(
            options.holdFirstCompletionContaining &&
            heldCompletionCount === 0 &&
            JSON.stringify(parsed).includes(options.holdFirstCompletionContaining),
          )

          // Detect the interim-message test trigger: the user's message
          // contains a specific keyword. The mock walks through the
          // INTERIM_SCRIPT turns in sequence.
          //
          // The trigger keyword is chosen so normal chat tests (which send
          // "Hello, can you hear me?" etc.) never hit this path.
          const messages: any[] = Array.isArray(parsed.messages) ? parsed.messages : []
          const lastUserMsg = [...messages].reverse().find(m => m?.role === 'user')
          const userText = typeof lastUserMsg?.content === 'string' ? lastUserMsg.content : ''
          if (userText) {
            _receivedUserTexts.push(userText)
          }
          const isInterimTrigger = userText.includes('E2E_INTERIM_TRIGGER')
          const isSidebarTrigger = userText.includes('E2E_SIDEBAR_TRIGGER')
          const isSidebarCrossTrigger = userText.includes('E2E_SIDEBAR_CROSS')
          const isQueueStopTrigger = userText.includes('E2E_QUEUE_STOP_TRIGGER')
          const isVerificationStopTrigger = messages.some(
            message => typeof message?.content === 'string' && message.content.includes(VERIFICATION_STOP_TRIGGER),
          )
          const isCorrectionSwitchTrigger = messages.some(
            message => typeof message?.content === 'string' && message.content.includes(CORRECTION_SWITCH_TRIGGER),
          )

          if (includesBlockingClarifyTrigger(parsed.messages)) {
            if (stream) {
              streamScriptedTurn(res, model, BLOCKING_CLARIFY_TURN)
            } else {
              nonStreamingScriptedTurn(res, model, BLOCKING_CLARIFY_TURN)
            }
            return
          }

          if (isQueueStopTrigger) {
            const turn = QUEUE_STOP_SCRIPT[_queueStopIndex] ?? QUEUE_STOP_SCRIPT[QUEUE_STOP_SCRIPT.length - 1]
            _queueStopIndex++
            if (stream) {
              streamScriptedTurn(res, model, turn)
            } else {
              nonStreamingScriptedTurn(res, model, turn)
            }
            return
          }

          if (isVerificationStopTrigger) {
            const script = verificationStopScript(options.verificationWritePath ?? 'e2e-verification-target.py')
            const turn = script[_verificationStopIndex] ?? script[script.length - 1]
            _verificationStopIndex++
            if (stream) {
              streamScriptedTurn(res, model, turn)
            } else {
              nonStreamingScriptedTurn(res, model, turn)
            }
            return
          }

          if (isCorrectionSwitchTrigger) {
            const turn = CORRECTION_SWITCH_SCRIPT[_correctionSwitchIndex] ?? CORRECTION_SWITCH_SCRIPT[CORRECTION_SWITCH_SCRIPT.length - 1]
            _correctionSwitchIndex++
            if (stream) {
              streamScriptedTurn(res, model, turn)
            } else {
              nonStreamingScriptedTurn(res, model, turn)
            }
            return
          }

          if (isSidebarCrossTrigger) {
            const script = sidebarCrossScript(options.backgroundReleasePath)
            const turn = script[_sidebarCrossIndex] ?? script[script.length - 1]
            _sidebarCrossIndex++

            if (stream) {
              streamScriptedTurn(res, model, turn)
            } else {
              nonStreamingScriptedTurn(res, model, turn)
            }
            return
          }

          if (isSidebarTrigger) {
            const turn = SIDEBAR_SCRIPT[_sidebarScriptIndex] ?? SIDEBAR_SCRIPT[SIDEBAR_SCRIPT.length - 1]
            _sidebarScriptIndex++

            if (stream) {
              streamScriptedTurn(res, model, turn)
            } else {
              nonStreamingScriptedTurn(res, model, turn)
            }
            return
          }

          if (isInterimTrigger) {
            const turn = INTERIM_SCRIPT[_scriptIndex] ?? INTERIM_SCRIPT[INTERIM_SCRIPT.length - 1]
            _scriptIndex++
            if (stream) {
              streamScriptedTurn(res, model, turn)
            } else {
              nonStreamingScriptedTurn(res, model, turn)
            }
            return
          }

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
            const holdThisStream = Boolean(
              options.holdFirstStreamForPrompt && typeof lastUserMessage?.content === 'string' &&
                lastUserMessage.content.includes(options.holdFirstStreamForPrompt),
            )
            res.writeHead(200, {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
              Connection: 'keep-alive',
            })

            // Send the content in a few chunks to simulate streaming.
            const words = MOCK_REPLY.split(' ')
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
              if ((holdThisStream || holdThisCompletion) && i === 1) {
                // Hold after the first content chunk until the test releases it,
                // so holdFirstStreamForPrompt / holdFirstCompletionContaining work
                // the same way here as on every other scripted/streamed response.
                if (holdThisCompletion) {
                  heldCompletionCount++
                }
                resolveHeldStreamStarted?.()
                heldStreamReleased.then(() => setTimeout(sendChunk, 20))
                return
              }
              // Small delay between chunks to simulate real streaming.
              setTimeout(sendChunk, 20)
            }

            sendChunk()
          } else {
            // Non-streaming response. Usage is sized from the conversation so
            // far (see usageFor) rather than a fixed stub, so the context meter
            // has real growth to read in e2e tests.
            const sendNonStreaming = () => {
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
                      message: { role: 'assistant', content: MOCK_REPLY },
                      finish_reason: 'stop',
                    },
                  ],
                  usage: usageFor(messages),
                }),
              )
            }
            if (holdThisCompletion) {
              heldCompletionCount++
              resolveHeldStreamStarted?.()
              void heldStreamReleased.then(sendNonStreaming)
            } else {
              sendNonStreaming()
            }
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
        receivedPrompts,
        waitForHeldStream: () => heldStreamStarted,
        waitForHeldCompletion: () => heldStreamStarted,
        releaseHeldStream: () => releaseHeldStream?.(),
        heldCompletionCount: () => heldCompletionCount,
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

// ─── Response helpers ──────────────────────────────────────────────────

/** SSE chunk shape for a streaming chat completion. */
function sseChunk(model: string, delta: Record<string, unknown>, finishReason: string | null = null): string {
  return `data: ${JSON.stringify({
    id: 'mock-completion',
    object: 'chat.completion.chunk',
    created: 0,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  })}\n\n`
}

/**
 * Stream a single scripted turn: first the text content (word by word),
 * then a chunk carrying the tool_calls (if any), with the appropriate
 * finish_reason.
 *
 * If the turn has no text and no tool calls, it's an empty final response.
 * If it has text but no tool calls, it's a final answer (finish_reason: stop).
 * If it has tool calls (with or without text), finish_reason is "tool_calls".
 */
function streamScriptedTurn(
  res: ServerResponse,
  model: string,
  turn: ScriptedTurn,
): void {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  })

  const hasToolCalls = turn.toolCalls && turn.toolCalls.length > 0
  const finishReason = hasToolCalls ? 'tool_calls' : 'stop'

  // If there's no text to stream, go straight to the tool_calls / finish.
  if (!turn.text) {
    if (hasToolCalls) {
      res.write(
        sseChunk(model, {
          tool_calls: turn.toolCalls!.map((tc, idx) => ({
            index: idx,
            id: `call_e2e_${_scriptIndex}_${idx}`,
            type: 'function',
            function: { name: tc.name, arguments: JSON.stringify(tc.args) },
          })),
        }, finishReason),
      )
    } else {
      res.write(sseChunk(model, {}, finishReason))
    }
    res.write('data: [DONE]\n\n')
    res.end()
    return
  }

  // Stream the text word by word, then emit tool_calls if present.
  const words = turn.text.split(' ')
  let i = 0

  const sendChunk = (): void => {
    if (i >= words.length) {
      // All text streamed — emit tool_calls if present, then finish.
      if (hasToolCalls) {
        res.write(
          sseChunk(model, {
            tool_calls: turn.toolCalls!.map((tc, idx) => ({
              index: idx,
              id: `call_e2e_${_scriptIndex}_${idx}`,
              type: 'function',
              function: { name: tc.name, arguments: JSON.stringify(tc.args) },
            })),
          }, finishReason),
        )
      } else {
        res.write(sseChunk(model, {}, finishReason))
      }
      res.write('data: [DONE]\n\n')
      res.end()
      return
    }

    const word = i === 0 ? words[i] : ' ' + words[i]
    res.write(sseChunk(model, { content: word }))
    i++
    setTimeout(sendChunk, 20)
  }

  sendChunk()
}

/** Non-streaming version of a scripted turn. */
function nonStreamingScriptedTurn(
  res: ServerResponse,
  model: string,
  turn: ScriptedTurn,
): void {
  const hasToolCalls = turn.toolCalls && turn.toolCalls.length > 0
  const finishReason = hasToolCalls ? 'tool_calls' : 'stop'

  const message: Record<string, unknown> = { role: 'assistant' }
  if (turn.text) {
    message.content = turn.text
  }
  if (hasToolCalls) {
    message.tool_calls = turn.toolCalls!.map((tc, idx) => ({
      id: `call_e2e_${_scriptIndex}_${idx}`,
      type: 'function',
      function: { name: tc.name, arguments: JSON.stringify(tc.args) },
    }))
  }

  res.writeHead(200, { 'Content-Type': 'application/json' })
  res.end(
    JSON.stringify({
      id: 'mock-completion',
      object: 'chat.completion',
      created: 0,
      model,
      choices: [{ index: 0, message, finish_reason: finishReason }],
      usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
    }),
  )
}

/**
 * Restart the mock server's script index so each test starts from turn 0.
 * Call this between tests that use the interim trigger.
 */
export function restartMockServer(): void {
  resetScriptIndex()
}

/** Test-controlled lifetime for the E2E_SIDEBAR_CROSS background process. */
export interface BackgroundReleaseHandle {
  /** Sentinel path — pass as `backgroundReleasePath` to `startMockServer`. */
  path: string
  /** End the background process now (creates the sentinel). */
  release: () => void
  /** Remove the sentinel if it still exists. Safe to call twice. */
  cleanup: () => void
}

/**
 * Create a sentinel that keeps the E2E_SIDEBAR_CROSS background process alive
 * until the test explicitly releases it.
 *
 * The cross-session sidebar tests need a background process that is still
 * RUNNING after the agent turn finishes — that is the state under test (a
 * session whose turn is done but whose background work is not). With a fixed
 * `sleep`, three independent clocks race: the sleep, the agent turn (two model
 * round trips plus a real subagent delegation), and the 4s success linger
 * before a finished task auto-dismisses. When a loaded CI runner makes the
 * turn slower than the sleep, the process is already gone and the assertion
 * samples an empty sidebar. Observed on CI 2026-07-26 across two unrelated
 * PRs: the "should appear" poll needed 7.5s to see the dot, by which point
 * `sleep 5` had exited.
 *
 * With a sentinel there is one clock and the test owns it:
 *
 * ```ts
 * const release = createBackgroundReleaseHandle()
 * const mock = await startMockServer({ backgroundReleasePath: release.path })
 * // ... assert the dot is visible; it cannot vanish on its own ...
 * release.release()   // now, and only now, the process exits
 * ```
 */
export function createBackgroundReleaseHandle(): BackgroundReleaseHandle {
  const path = nodePath.join(
    os.tmpdir(),
    `hermes-e2e-bg-release-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  )
  return {
    path,
    release: () => {
      try {
        fs.writeFileSync(path, 'release')
      } catch {
        // The process also has a bounded fallback wait; a failed write must
        // not crash the test before its real assertions run.
      }
    },
    cleanup: () => {
      try {
        fs.rmSync(path, { force: true })
      } catch {
        // Best-effort — the sentinel lives in the OS temp dir.
      }
    },
  }
}

/**
 * The interim script's text constants, exported for test assertions.
 * Each entry is the visible text of one turn. Turns with empty text
 * produce no interim message and are excluded from this list.
 */
export const INTERIM_TEXTS = {
  /** All interim texts that should appear as sealed messages when the flag is ON. */
  interims: INTERIM_SCRIPT
    .filter((t) => t.text && t.toolCalls)
    .map((t) => t.text),
  /** The final answer text. */
  finalText: INTERIM_SCRIPT[INTERIM_SCRIPT.length - 1].text,
  /** Text that should NOT produce an interim (empty-text tool turn). */
  silentTurnIndex: INTERIM_SCRIPT.findIndex((t) => !t.text && t.toolCalls),
} as const

/** The sidebar-states script's text constants, exported for test assertions. */
export const SIDEBAR_TEXTS = {
  /** The interim text from turn 1 (alongside tool calls). */
  interimText: SIDEBAR_SCRIPT[0].text,
  /** The final answer text. */
  finalText: SIDEBAR_SCRIPT[SIDEBAR_SCRIPT.length - 1].text,
  /** The background process command (for asserting process.list entries). */
  bgCommand: 'echo "background process output" && sleep 1 && echo "done"',
  /** The subagent's goal (for asserting subagent panel state). */
  subagentGoal: 'Summarize the test results',
} as const

/** The cross-session sidebar script's text constants. */
export const SIDEBAR_CROSS_TEXTS = {
  /** The interim text from turn 1. */
  interimText: SIDEBAR_CROSS_SCRIPT[0].text,
  /** The final answer text. */
  finalText: SIDEBAR_CROSS_SCRIPT[SIDEBAR_CROSS_SCRIPT.length - 1].text,
  /**
   * The default (unheld) background process command. Tests that pass a
   * `backgroundReleasePath` get a sentinel-waiting command instead — see
   * `createBackgroundReleaseHandle`.
   */
  bgCommand: sidebarCrossBgCommand(),
  /** The subagent's goal. */
  subagentGoal: 'Analyze cross-session state',
} as const

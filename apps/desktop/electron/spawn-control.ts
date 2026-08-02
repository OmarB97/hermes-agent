import crypto from 'node:crypto'
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'

// Local control channel: let a CLT on this machine ask the RUNNING app to start
// a chat, so the app owns and streams it exactly like a typed one.
//
// Why this lives in Electron main and NOT as a route on the Python backend:
// the gateway routes streaming events by the transport stored on the session
// (tui_gateway/server.py `write_json`), and `prompt.submit` re-pins that
// transport to whichever client made the call. An HTTP caller has no websocket,
// so a backend-side spawn route would create a session whose deltas go to
// stdio — the renderer would see the row appear (via the session-store watcher)
// but never receive a token. Handing the request to the renderer instead means
// the session is created and submitted on the renderer's own gateway socket, so
// every downstream behaviour — streaming, transcript, sidebar, titles — is the
// existing typed-chat path with no new streaming code.
//
// Security posture:
//   - The listener binds 127.0.0.1 only, on an ephemeral port. It is never
//     reachable off-box.
//   - Requests must carry a token minted fresh each app launch and never
//     persisted across runs.
//   - The token is published in a 0600 file under the user's own HERMES_HOME,
//     so "can read the token" is already "is this user". The file is removed on
//     quit, and a stale one only ever yields a connection-refused error.
//   - Non-loopback peers are rejected even if the bind were ever widened.

export const SPAWN_CONTROL_FILE_MODE = 0o600
export const SPAWN_CONTROL_DIR_MODE = 0o700
export const SPAWN_CONTROL_SCHEMA_VERSION = 1
export const SPAWN_TOKEN_HEADER = 'x-hermes-desktop-token'

/** Refuse absurd bodies outright — this endpoint takes a prompt, not an upload. */
export const SPAWN_MAX_BODY_BYTES = 128 * 1024

export interface SpawnRequest {
  prompt: string
  model?: string
  provider?: string
  profile?: string
  /** Toolsets to pin for this one session. Names are checked by the gateway
   *  (`session.create`), which owns the vocabulary; this side only shapes it. */
  toolsets?: string[]
  /** Unattended run: nobody is watching, so the session must not stop to ask. */
  delegated?: boolean
  /** How long a delegated session waits for a person before answering its own
   *  question. Omitted means the renderer's default. */
  delegatedTimeoutMs?: number
}

export type SpawnDelivery = 'delivered' | 'no-window'

export interface SpawnControlOptions {
  hermesHome: string
  /** Hand a validated request to the renderer. */
  deliver: (request: SpawnRequest) => SpawnDelivery
  onLog?: (message: string) => void
}

export interface SpawnControlServer {
  port: () => null | number
  controlFilePath: () => string
  close: () => void
}

export function spawnControlDir(hermesHome: string): string {
  return path.join(hermesHome, 'desktop')
}

export function spawnControlFilePath(hermesHome: string): string {
  return path.join(spawnControlDir(hermesHome), 'control.json')
}

/**
 * Validate an untrusted request body.
 *
 * Returns the normalized request, or an error string. `prompt` is the only
 * required field: every other key is an override that, when omitted, leaves the
 * app on its own current selection — so a spawn with no `model` behaves exactly
 * like the user typing into whatever chat settings they already have.
 */
export function parseSpawnRequest(raw: unknown): { error: string } | { request: SpawnRequest } {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return { error: 'body must be a JSON object' }
  }

  const body = raw as Record<string, unknown>
  const prompt = typeof body.prompt === 'string' ? body.prompt.trim() : ''

  if (!prompt) {
    return { error: 'prompt is required' }
  }

  const request: SpawnRequest = { prompt }

  for (const key of ['model', 'provider', 'profile'] as const) {
    const value = body[key]

    if (value === undefined || value === null) {
      continue
    }

    if (typeof value !== 'string') {
      return { error: `${key} must be a string` }
    }

    const trimmed = value.trim()

    if (trimmed) {
      request[key] = trimmed
    }
  }

  // Shape only — whether these NAME anything real is the gateway's call, since
  // the vocabulary (built-ins, plugin toolsets, enabled MCP servers) lives on
  // the Python side and varies by profile. An all-blank list is a typo rather
  // than a request to inherit, so it is refused here instead of quietly
  // becoming "keep the app's toolsets" — that silent drop is what made this
  // flag a no-op for its entire first life (#298, removed in #315).
  if (body.toolsets !== undefined && body.toolsets !== null) {
    if (!Array.isArray(body.toolsets) || body.toolsets.some(entry => typeof entry !== 'string')) {
      return { error: 'toolsets must be an array of strings' }
    }

    const toolsets = (body.toolsets as string[]).map(entry => entry.trim()).filter(Boolean)

    if (!toolsets.length) {
      return { error: 'toolsets named nothing; omit it to inherit the current toolsets' }
    }

    request.toolsets = toolsets
  }

  if (body.delegated !== undefined && body.delegated !== null) {
    if (typeof body.delegated !== 'boolean') {
      return { error: 'delegated must be a boolean' }
    }

    if (body.delegated) {
      request.delegated = true
    }
  }

  if (body.delegatedTimeoutMs !== undefined && body.delegatedTimeoutMs !== null) {
    if (typeof body.delegatedTimeoutMs !== 'number' || !Number.isFinite(body.delegatedTimeoutMs)) {
      return { error: 'delegatedTimeoutMs must be a number' }
    }

    if (body.delegatedTimeoutMs <= 0) {
      return { error: 'delegatedTimeoutMs must be greater than 0' }
    }

    // A timeout on an attended spawn would be read by nobody. Reject it here
    // rather than accept a request whose caller clearly meant something else.
    if (!request.delegated) {
      return { error: 'delegatedTimeoutMs requires delegated: true' }
    }

    request.delegatedTimeoutMs = body.delegatedTimeoutMs
  }

  return { request }
}

function isLoopback(address: string | undefined): boolean {
  if (!address) {
    return false
  }

  // Node reports IPv4-mapped IPv6 for a v4 client on a dual-stack socket.
  const normalized = address.replace(/^::ffff:/, '')

  return normalized === '127.0.0.1' || normalized === '::1' || normalized.startsWith('127.')
}

function sendJson(response: http.ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload)

  response.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(body),
    // This endpoint is for local tooling, never a browser page.
    'access-control-allow-origin': 'null'
  })
  response.end(body)
}

/**
 * Start the loopback control listener and publish its port + token.
 *
 * Failing to start is non-fatal for the app: the caller logs and carries on
 * without the spawn channel rather than blocking boot.
 */
export function startSpawnControlServer({
  hermesHome,
  deliver,
  onLog
}: SpawnControlOptions): Promise<SpawnControlServer> {
  const token = crypto.randomBytes(32).toString('base64url')
  const filePath = spawnControlFilePath(hermesHome)

  const server = http.createServer((request, response) => {
    if (!isLoopback(request.socket?.remoteAddress)) {
      sendJson(response, 403, { ok: false, error: 'forbidden' })

      return
    }

    if (request.method !== 'POST' || (request.url || '').split('?')[0] !== '/spawn') {
      sendJson(response, 404, { ok: false, error: 'not found' })

      return
    }

    const presented = request.headers[SPAWN_TOKEN_HEADER]
    const supplied = Array.isArray(presented) ? presented[0] : presented

    if (!supplied || !timingSafeEqualStrings(String(supplied), token)) {
      sendJson(response, 401, { ok: false, error: 'unauthorized' })

      return
    }

    const chunks: Buffer[] = []
    let size = 0
    let aborted = false

    request.on('data', chunk => {
      if (aborted) {
        return
      }

      size += chunk.length

      if (size > SPAWN_MAX_BODY_BYTES) {
        aborted = true
        sendJson(response, 413, { ok: false, error: 'body too large' })
        request.destroy()

        return
      }

      chunks.push(chunk)
    })

    request.on('end', () => {
      if (aborted) {
        return
      }

      let parsed: unknown

      try {
        parsed = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')
      } catch {
        sendJson(response, 400, { ok: false, error: 'body must be valid JSON' })

        return
      }

      const result = parseSpawnRequest(parsed)

      if ('error' in result) {
        sendJson(response, 400, { ok: false, error: result.error })

        return
      }

      let delivery: SpawnDelivery

      try {
        delivery = deliver(result.request)
      } catch (err) {
        onLog?.(`[spawn-control] delivery failed: ${(err as Error)?.message ?? err}`)
        sendJson(response, 500, { ok: false, error: 'delivery failed' })

        return
      }

      if (delivery === 'no-window') {
        sendJson(response, 503, { ok: false, error: 'no window' })

        return
      }

      sendJson(response, 202, { ok: true })
    })
  })

  return new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      server.removeListener('error', reject)

      const address = server.address()
      const port = address && typeof address === 'object' ? address.port : null

      if (!port) {
        server.close()
        reject(new Error('control server bound without a port'))

        return
      }

      try {
        writeControlFile(filePath, { port, token })
      } catch (err) {
        server.close()
        reject(err as Error)

        return
      }

      onLog?.(`[spawn-control] listening on 127.0.0.1:${port} (token in ${filePath})`)

      resolve({
        port: () => {
          const current = server.address()

          return current && typeof current === 'object' ? current.port : null
        },
        controlFilePath: () => filePath,
        close: () => {
          removeControlFile(filePath)

          try {
            server.close()
          } catch {
            // already closing
          }
        }
      })
    })
  })
}

export function writeControlFile(filePath: string, { port, token }: { port: number; token: string }): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true, mode: SPAWN_CONTROL_DIR_MODE })

  const payload = JSON.stringify(
    {
      schemaVersion: SPAWN_CONTROL_SCHEMA_VERSION,
      port,
      token,
      pid: process.pid,
      startedAt: new Date().toISOString()
    },
    null,
    2
  )

  // Write through a temp file so a reader never sees a half-written token, and
  // create it already-private rather than widening then narrowing.
  const tmp = `${filePath}.${process.pid}.tmp`

  fs.writeFileSync(tmp, payload, { mode: SPAWN_CONTROL_FILE_MODE })

  try {
    fs.renameSync(tmp, filePath)
  } catch (err) {
    try {
      fs.unlinkSync(tmp)
    } catch {
      // best effort
    }

    throw err
  }

  // rename preserves the temp file's mode, but an existing file's mode wins on
  // some platforms — assert it rather than assume.
  try {
    fs.chmodSync(filePath, SPAWN_CONTROL_FILE_MODE)
  } catch {
    // best effort
  }
}

export function removeControlFile(filePath: string): void {
  try {
    fs.unlinkSync(filePath)
  } catch {
    // already gone
  }
}

function timingSafeEqualStrings(a: string, b: string): boolean {
  const left = Buffer.from(a)
  const right = Buffer.from(b)

  // timingSafeEqual throws on length mismatch; compare lengths first but still
  // run the constant-time compare on equal-length input.
  if (left.length !== right.length) {
    return false
  }

  return crypto.timingSafeEqual(left, right)
}

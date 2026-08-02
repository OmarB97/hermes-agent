/**
 * Tests for the local spawn control channel.
 *
 * This endpoint lets any process that can read a file under the user's home ask
 * the app to run an agent turn, so the auth and binding rules are the security
 * boundary — they get the most attention here, alongside the request validation
 * that decides what reaches the renderer.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import {
  parseSpawnRequest,
  SPAWN_CONTROL_FILE_MODE,
  SPAWN_TOKEN_HEADER,
  spawnControlFilePath,
  type SpawnRequest,
  startSpawnControlServer
} from './spawn-control'

const started: Array<{ close: () => void }> = []

afterEach(() => {
  while (started.length) {
    started.pop()?.close()
  }
})

function tempHome(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-spawn-'))
}

async function boot(deliver: (r: SpawnRequest) => 'delivered' | 'no-window' = () => 'delivered') {
  const home = tempHome()
  const server = await startSpawnControlServer({ hermesHome: home, deliver })
  started.push(server)

  const control = JSON.parse(fs.readFileSync(spawnControlFilePath(home), 'utf8'))

  return { home, server, control }
}

interface Reply {
  status: number
  body: Record<string, unknown>
}

function post(port: number, body: unknown, headers: Record<string, string> = {}): Promise<Reply> {
  return new Promise((resolve, reject) => {
    const payload = typeof body === 'string' ? body : JSON.stringify(body)

    const req = http.request(
      {
        host: '127.0.0.1',
        port,
        path: '/spawn',
        method: 'POST',
        headers: { 'content-type': 'application/json', ...headers }
      },
      res => {
        const chunks: Buffer[] = []
        res.on('data', c => chunks.push(c))
        res.on('end', () => {
          const text = Buffer.concat(chunks).toString('utf8')

          resolve({ status: res.statusCode ?? 0, body: text ? JSON.parse(text) : {} })
        })
      }
    )

    req.on('error', reject)
    req.end(payload)
  })
}

// ---------------------------------------------------------------- validation

test('a prompt is the only required field', () => {
  const result = parseSpawnRequest({ prompt: '  do the thing  ' })

  assert.ok('request' in result)
  assert.deepEqual(result.request, { prompt: 'do the thing' })
})

test('rejects a body with no usable prompt', () => {
  for (const body of [null, 'nope', [], {}, { prompt: '' }, { prompt: '   ' }, { prompt: 42 }]) {
    assert.ok('error' in parseSpawnRequest(body), JSON.stringify(body))
  }
})

// Omitting a field must mean "use the app's current selection", never "clear
// it" — that is what keeps a spawn with no --model identical to a typed chat.
test('omitted overrides stay omitted rather than becoming empty strings', () => {
  const result = parseSpawnRequest({ prompt: 'hi', model: null, provider: undefined, profile: '   ' })

  assert.ok('request' in result)
  assert.deepEqual(Object.keys(result.request), ['prompt'])
})

test('carries model, provider, profile and toolsets when given', () => {
  const result = parseSpawnRequest({
    prompt: 'hi',
    model: ' deepseek-v4-flash-0731-ds4 ',
    provider: 'ai-router',
    profile: 'work',
    toolsets: [' file ', '', 'terminal']
  })

  assert.ok('request' in result)
  assert.deepEqual(result.request, {
    prompt: 'hi',
    model: 'deepseek-v4-flash-0731-ds4',
    provider: 'ai-router',
    profile: 'work',
    toolsets: ['file', 'terminal']
  })
})

test('rejects wrong-typed overrides', () => {
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', model: 7 }))
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', toolsets: 'file' }))
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', toolsets: ['ok', 3] }))
})

// Whether a name means anything is the gateway's call (the vocabulary is
// Python-side and varies by profile), but a list that names NOTHING is a typo
// this layer can catch. Letting it through as "no toolsets key" would silently
// spawn with the app's tools — the accept-and-drop that got this flag removed
// in #315, just relocated.
test('rejects a toolsets list that names nothing', () => {
  for (const toolsets of [[], ['', '   ']]) {
    const result = parseSpawnRequest({ prompt: 'hi', toolsets })

    assert.ok('error' in result, `expected toolsets: ${JSON.stringify(toolsets)} to be rejected`)
    assert.match(result.error, /named nothing/)
  }
})

// An absent/null key is a caller with no opinion, not an empty pin.
test('an omitted or null toolsets key leaves an ordinary request', () => {
  for (const body of [{ prompt: 'hi' }, { prompt: 'hi', toolsets: null }, { prompt: 'hi', toolsets: undefined }]) {
    const result = parseSpawnRequest(body)

    assert.ok('request' in result)
    assert.deepEqual(Object.keys(result.request), ['prompt'])
  }
})

// ------------------------------------------------------------------ delegated

test('carries the delegated flag and its wait', () => {
  const result = parseSpawnRequest({ prompt: 'hi', delegated: true, delegatedTimeoutMs: 30_000 })

  assert.ok('request' in result)
  assert.equal(result.request.delegated, true)
  assert.equal(result.request.delegatedTimeoutMs, 30_000)
})

// `delegated: false` is what an ordinary spawn sends if a caller sets it
// explicitly; it must land as a plain attended request, not a key that later
// code has to keep remembering to check for falseness.
test('an explicit delegated: false leaves an ordinary request', () => {
  const result = parseSpawnRequest({ prompt: 'hi', delegated: false })

  assert.ok('request' in result)
  assert.deepEqual(Object.keys(result.request), ['prompt'])
})

test('rejects a non-boolean delegated', () => {
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', delegated: 'yes' }))
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', delegated: 1 }))
})

test('rejects a wait that is not a usable number', () => {
  for (const value of ['120', Number.NaN, Number.POSITIVE_INFINITY, 0, -5]) {
    assert.ok('error' in parseSpawnRequest({ prompt: 'hi', delegated: true, delegatedTimeoutMs: value }), String(value))
  }
})

// A wait with nothing to wait for means the caller meant something we are not
// doing. Saying so beats running an attended session they think is delegated.
test('rejects a wait on a spawn that is not delegated', () => {
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', delegatedTimeoutMs: 30_000 }))
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', delegated: false, delegatedTimeoutMs: 30_000 }))
})

// ---------------------------------------------------------------------- goal

test('carries a goal and its turn budget when given', () => {
  const result = parseSpawnRequest({ prompt: 'hi', goal: '  ship the parser  ', goalMaxTurns: 40 })

  assert.ok('request' in result)
  assert.deepEqual(result.request, { prompt: 'hi', goal: 'ship the parser', goalMaxTurns: 40 })
})

test('a spawn with no goal keeps no goal keys at all', () => {
  const result = parseSpawnRequest({ prompt: 'hi', goal: null, goalMaxTurns: null })

  assert.ok('request' in result)
  assert.deepEqual(Object.keys(result.request), ['prompt'])
})

// A blank goal is a caller that meant to set an objective and sent nothing.
// Accepting it would start a chat that looks like a goal session right up until
// it stops after one turn — the same accept-and-drop that killed --toolsets.
test('rejects a goal that names nothing', () => {
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', goal: '   ' }))
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', goal: 7 }))
})

test('rejects a turn budget that is not a usable count', () => {
  for (const value of ['40', 1.5, Number.NaN, 0, -1]) {
    assert.ok(
      'error' in parseSpawnRequest({ prompt: 'hi', goal: 'ship it', goalMaxTurns: value }),
      String(value)
    )
  }
})

// A budget with no goal to spend it on means the caller meant something we are
// not doing.
test('rejects a turn budget on a spawn with no goal', () => {
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', goalMaxTurns: 40 }))
})

test('a goal composes with delegated and the other overrides', () => {
  const result = parseSpawnRequest({
    prompt: 'hi',
    goal: 'ship the parser',
    goalMaxTurns: 40,
    delegated: true,
    model: 'deepseek-v4-flash-0731-ds4',
    profile: 'work'
  })

  assert.ok('request' in result)
  assert.deepEqual(result.request, {
    prompt: 'hi',
    goal: 'ship the parser',
    goalMaxTurns: 40,
    delegated: true,
    model: 'deepseek-v4-flash-0731-ds4',
    profile: 'work'
  })
})

// ------------------------------------------- declared command allowlist

// This is the one spawn field that WIDENS what a session may do without a
// person: it says which commands may run when the approval classifier cannot
// be reached and there is nobody to ask. Both halves are required together —
// commands with no root could act anywhere on the machine, and a root with no
// commands allowlists nothing — so accept-and-drop is not an option here.
test('carries a declared command allowlist when both halves are given', () => {
  const result = parseSpawnRequest({
    prompt: 'hi',
    allowedCommands: [' godot ', '', 'git'],
    allowedCommandRoot: '  /Users/me/game  ',
    delegated: true
  })

  assert.ok('request' in result)
  assert.deepEqual(result.request, {
    prompt: 'hi',
    allowedCommands: ['godot', 'git'],
    allowedCommandRoot: '/Users/me/game',
    delegated: true
  })
})

test('rejects a declared allowlist that is not an array of strings', () => {
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', allowedCommands: 'git', allowedCommandRoot: '/w' }))
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', allowedCommands: ['git', 3], allowedCommandRoot: '/w' }))
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', allowedCommands: ['git'], allowedCommandRoot: 7 }))
})

test('rejects an allowlist that names nothing', () => {
  for (const allowedCommands of [[], ['', '   ']]) {
    assert.ok('error' in parseSpawnRequest({ prompt: 'hi', allowedCommands, allowedCommandRoot: '/w' }))
  }
})

test('rejects either half of the allowlist on its own', () => {
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', allowedCommands: ['git'] }))
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', allowedCommandRoot: '/w' }))
  assert.ok('error' in parseSpawnRequest({ prompt: 'hi', allowedCommands: ['git'], allowedCommandRoot: '  ' }))
})

test('an omitted allowlist leaves an ordinary, fail-closed request', () => {
  for (const body of [
    { prompt: 'hi' },
    { prompt: 'hi', allowedCommands: null, allowedCommandRoot: null },
    { prompt: 'hi', allowedCommands: undefined, allowedCommandRoot: undefined }
  ]) {
    const result = parseSpawnRequest(body)

    assert.ok('request' in result)
    assert.deepEqual(result.request, { prompt: 'hi' })
  }
})

// -------------------------------------------------------------------- server

test('binds loopback only, on an ephemeral port', async () => {
  const { server } = await boot()
  const port = server.port()

  assert.ok(typeof port === 'number' && port > 0)

  // A bind to 127.0.0.1 is not reachable on an external interface. Assert the
  // address family/host directly rather than trying to dial a public IP.
  const external = Object.values(os.networkInterfaces())
    .flat()
    .find(i => i && !i.internal && i.family === 'IPv4')

  if (external) {
    await assert.rejects(
      () =>
        new Promise((resolve, reject) => {
          const req = http.request(
            { host: external.address, port, path: '/spawn', method: 'POST', timeout: 2000 },
            resolve
          )

          req.on('error', reject)
          req.on('timeout', () => reject(new Error('timeout')))
          req.end('{}')
        })
    )
  }
})

test('publishes port + token in a private control file', async () => {
  const { home, server, control } = await boot()

  assert.equal(control.schemaVersion, 1)
  assert.equal(control.port, server.port())
  assert.ok(typeof control.token === 'string' && control.token.length >= 32)
  assert.equal(control.pid, process.pid)

  const mode = fs.statSync(spawnControlFilePath(home)).mode & 0o777

  assert.equal(mode, SPAWN_CONTROL_FILE_MODE, 'control file must not be group/world readable')
})

test('removes the control file on close so a stale token cannot linger', async () => {
  const { home, server } = await boot()
  const file = spawnControlFilePath(home)

  assert.ok(fs.existsSync(file))
  server.close()
  assert.equal(fs.existsSync(file), false)
})

test('accepts a correctly authenticated spawn and hands it to the renderer', async () => {
  const seen: SpawnRequest[] = []

  const { server, control } = await boot(r => {
    seen.push(r)

    return 'delivered'
  })

  const reply = await post(
    server.port()!,
    { prompt: 'ship it', model: 'deepseek-v4-flash-0731-ds4', provider: 'ai-router' },
    { [SPAWN_TOKEN_HEADER]: control.token }
  )

  assert.equal(reply.status, 202)
  assert.deepEqual(reply.body, { ok: true })
  assert.deepEqual(seen, [{ prompt: 'ship it', model: 'deepseek-v4-flash-0731-ds4', provider: 'ai-router' }])
})

test('rejects a missing or wrong token without touching the renderer', async () => {
  const seen: SpawnRequest[] = []

  const { server, control } = await boot(r => {
    seen.push(r)

    return 'delivered'
  })

  const noToken = await post(server.port()!, { prompt: 'x' })
  const wrongToken = await post(server.port()!, { prompt: 'x' }, { [SPAWN_TOKEN_HEADER]: 'not-the-token' })

  // Same length as the real token, to exercise the constant-time compare path.
  const sameLength = await post(
    server.port()!,
    { prompt: 'x' },
    { [SPAWN_TOKEN_HEADER]: 'z'.repeat(control.token.length) }
  )

  assert.equal(noToken.status, 401)
  assert.equal(wrongToken.status, 401)
  assert.equal(sameLength.status, 401)
  assert.deepEqual(seen, [], 'an unauthorized request must never reach the renderer')
})

test('reports 503 when there is no window to run the chat', async () => {
  const { server, control } = await boot(() => 'no-window')

  const reply = await post(server.port()!, { prompt: 'x' }, { [SPAWN_TOKEN_HEADER]: control.token })

  assert.equal(reply.status, 503)
  assert.equal(reply.body.ok, false)
})

test('rejects a bad body with 400, after auth', async () => {
  const { server, control } = await boot()

  const noPrompt = await post(server.port()!, { model: 'x' }, { [SPAWN_TOKEN_HEADER]: control.token })
  const notJson = await post(server.port()!, 'not json at all', { [SPAWN_TOKEN_HEADER]: control.token })

  assert.equal(noPrompt.status, 400)
  assert.equal(notJson.status, 400)
})

test('only POST /spawn is routable', async () => {
  const { server, control } = await boot()
  const port = server.port()!

  const wrongPath = await new Promise<Reply>((resolve, reject) => {
    const req = http.request(
      { host: '127.0.0.1', port, path: '/anything', method: 'POST', headers: { [SPAWN_TOKEN_HEADER]: control.token } },
      res => {
        const chunks: Buffer[] = []
        res.on('data', c => chunks.push(c))
        res.on('end', () =>
          resolve({ status: res.statusCode ?? 0, body: JSON.parse(Buffer.concat(chunks).toString() || '{}') })
        )
      }
    )

    req.on('error', reject)
    req.end('{}')
  })

  assert.equal(wrongPath.status, 404)
})

// A renderer that throws must surface as a server error, not wedge the socket.
test('a delivery that throws answers 500 instead of hanging', async () => {
  const { server, control } = await boot(() => {
    throw new Error('renderer exploded')
  })

  const reply = await post(server.port()!, { prompt: 'x' }, { [SPAWN_TOKEN_HEADER]: control.token })

  assert.equal(reply.status, 500)
})

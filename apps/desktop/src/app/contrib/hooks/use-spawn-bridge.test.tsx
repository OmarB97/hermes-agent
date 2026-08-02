import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { DELEGATED_CLARIFY_TIMEOUT_MS, DELEGATION_CONTRACT } from '@/lib/delegated-spawn'
import { setFreshDraftReady } from '@/store/session'

import type { SubmitTextOptions } from '../../session/hooks/use-prompt-actions/utils'
import type { ClientSessionState } from '../../types'

import { SPAWN_DRAFT_SETTLE_TIMEOUT_MS, useSpawnBridge } from './use-spawn-bridge'

// `hermes desktop spawn` hands a prompt to this window so the chat is created
// on OUR gateway socket and therefore streams like a typed one. These tests pin
// the two properties that makes true: it walks the real new-chat + submit path,
// and it never writes the spawned model into the persisted composer selection.

const secondary = vi.hoisted(() => ({ value: false }))

vi.mock('@/store/windows', () => ({
  isSecondaryWindow: () => secondary.value
}))

let spawnHandlers: Array<(payload: unknown) => void> = []
let unsubscribes = 0
let readySignals = 0

beforeEach(() => {
  // Most tests exercise an already-settled draft; the settle-wait has its own.
  setFreshDraftReady(true)
  secondary.value = false
  spawnHandlers = []
  unsubscribes = 0
  readySignals = 0
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    onSpawnSession: (callback: (payload: unknown) => void) => {
      spawnHandlers.push(callback)

      return () => {
        unsubscribes += 1
      }
    },
    signalSpawnReady: async () => {
      readySignals += 1

      return { ok: true }
    }
  }
})

afterEach(() => {
  setFreshDraftReady(false)
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  vi.restoreAllMocks()
})

// The hook only ever patches an existing state, so a shallow double is enough.
function stateDouble() {
  return vi.fn((_sessionId: string, updater: (state: ClientSessionState) => ClientSessionState) =>
    updater({ delegatedTimeoutMs: null } as ClientSessionState)
  )
}

function mount() {
  const startFreshSession = vi.fn()
  const submitText = vi.fn(async (_text: string, _options?: SubmitTextOptions) => true)
  const updateSessionState = stateDouble()

  renderHook(() => useSpawnBridge({ startFreshSession, submitText, updateSessionState }))

  return { startFreshSession, submitText, updateSessionState }
}

it('starts a fresh session and submits the prompt through the normal path', async () => {
  const { startFreshSession, submitText } = mount()

  await act(async () => {
    spawnHandlers[0]({ prompt: 'summarise the repo' })
  })

  expect(startFreshSession).toHaveBeenCalledTimes(1)
  expect(submitText).toHaveBeenCalledTimes(1)
  expect(submitText).toHaveBeenCalledWith('summarise the repo', { attachments: [] })
})

// The ordering is load-bearing: submitting before the draft is reset would
// append the spawned prompt to whatever chat is currently open.
it('resets the draft BEFORE submitting', async () => {
  const order: string[] = []
  const startFreshSession = vi.fn(() => void order.push('fresh'))

  const submitText = vi.fn(async (_text: string, _options?: SubmitTextOptions) => {
    order.push('submit')

    return true
  })

  renderHook(() => useSpawnBridge({ startFreshSession, submitText, updateSessionState: stateDouble() }))

  await act(async () => {
    spawnHandlers[0]({ prompt: 'go' })
  })

  expect(order).toEqual(['fresh', 'submit'])
})

it('passes model/provider/profile as per-session overrides', async () => {
  const { submitText } = mount()

  await act(async () => {
    spawnHandlers[0]({
      prompt: 'go',
      model: 'deepseek-v4-flash-0731-ds4',
      provider: 'ai-router',
      profile: 'work'
    })
  })

  expect(submitText).toHaveBeenCalledWith('go', {
    attachments: [],
    sessionOverrides: {
      model: 'deepseek-v4-flash-0731-ds4',
      provider: 'ai-router',
      profile: 'work'
    }
  })
})

// A spawn with no model must be indistinguishable from a typed chat, so it must
// not send an override object at all.
it('sends no overrides when none were requested', async () => {
  const { submitText } = mount()

  await act(async () => {
    spawnHandlers[0]({ prompt: 'go' })
  })

  expect(submitText.mock.calls[0][1]).toEqual({ attachments: [] })
})

// ------------------------------------------------------------- delegated mode

// The contract has to reach the model, and it has to reach it BEFORE the brief
// — an instruction not to ask, read after the thing that prompts the question,
// is worth nothing.
it('prepends the unattended contract to a delegated prompt', async () => {
  const { submitText } = mount()

  await act(async () => {
    spawnHandlers[0]({ prompt: 'write up the auth flow', delegated: true })
  })

  const sent = submitText.mock.calls[0][0]

  expect(sent.startsWith(DELEGATION_CONTRACT)).toBe(true)
  expect(sent.endsWith('write up the auth flow')).toBe(true)
})

// An ordinary spawn has to stay byte-identical to a typed chat.
it('leaves an undelegated prompt exactly as sent', async () => {
  const { submitText, updateSessionState } = mount()

  await act(async () => {
    spawnHandlers[0]({ prompt: 'write up the auth flow' })
  })

  expect(submitText.mock.calls[0][0]).toBe('write up the auth flow')
  expect(submitText.mock.calls[0][1]).not.toHaveProperty('onSessionCreated')
  expect(updateSessionState).not.toHaveBeenCalled()
})

// Submit resolves to a bare boolean, so the created session id arrives through
// this callback or not at all — and without it there is nothing to hang the
// clarify deadline on.
it('marks the session it created as delegated, with the requested wait', async () => {
  const { submitText, updateSessionState } = mount()

  await act(async () => {
    spawnHandlers[0]({ prompt: 'go', delegated: true, delegatedTimeoutMs: 30_000 })
  })

  const options = submitText.mock.calls[0][1] as SubmitTextOptions

  options.onSessionCreated?.('runtime-1')

  expect(updateSessionState).toHaveBeenCalledTimes(1)
  expect(updateSessionState.mock.calls[0][0]).toBe('runtime-1')
  expect(updateSessionState.mock.results[0].value).toMatchObject({ delegatedTimeoutMs: 30_000 })
})

// An absent or nonsense wait must still leave the session delegated — falling
// back to "not delegated" would silently drop the guarantee.
it('falls back to the default wait when none was given', async () => {
  const { submitText, updateSessionState } = mount()

  await act(async () => {
    spawnHandlers[0]({ prompt: 'go', delegated: true })
  })

  const options = submitText.mock.calls[0][1] as SubmitTextOptions

  options.onSessionCreated?.('runtime-1')

  expect(updateSessionState.mock.results[0].value).toMatchObject({
    delegatedTimeoutMs: DELEGATED_CLARIFY_TIMEOUT_MS
  })
})

it('ignores a spawn with no usable prompt', async () => {
  const { startFreshSession, submitText } = mount()

  await act(async () => {
    spawnHandlers[0]({ prompt: '   ' })
    spawnHandlers[0]({})
  })

  expect(startFreshSession).not.toHaveBeenCalled()
  expect(submitText).not.toHaveBeenCalled()
})

// Each spawn creates its own session, so two of them never share a session to
// race over.
it('gives each of two back-to-back spawns its own fresh session', async () => {
  const { startFreshSession, submitText } = mount()

  await act(async () => {
    spawnHandlers[0]({ prompt: 'first', model: 'a' })
    spawnHandlers[0]({ prompt: 'second', model: 'b' })
  })

  expect(startFreshSession).toHaveBeenCalledTimes(2)
  expect(submitText.mock.calls.map(c => c[0])).toEqual(['first', 'second'])
  expect(submitText.mock.calls[0][1]).toMatchObject({ sessionOverrides: { model: 'a' } })
  expect(submitText.mock.calls[1][1]).toMatchObject({ sessionOverrides: { model: 'b' } })
})

// FAIL-BEFORE. Measured against the sandbox app on 2026-08-02: two spawns 2s
// apart produced only ONE session. `createBackendSessionForSend` closes the
// session it just created if the route token or active-session refs moved while
// `session.create` was in flight (its guard against the user navigating away
// mid-create) — and the second spawn's `startFreshSession` moves exactly those.
// The first spawn silently vanished after its caller had already been told 202.
// Spawns must therefore be serialized: the next one may not touch the draft
// until the previous submit has resolved.
it('does not start the next spawn until the previous submit resolves', async () => {
  const order: string[] = []
  const release: Array<() => void> = []
  const startFreshSession = vi.fn(() => void order.push('fresh'))

  const submitText = vi.fn(
    (text: string, _options?: SubmitTextOptions) =>
      new Promise<boolean>(resolve => {
        order.push(`submit:${text}`)
        release.push(() => resolve(true))
      })
  )

  renderHook(() => useSpawnBridge({ startFreshSession, submitText, updateSessionState: stateDouble() }))

  await act(async () => {
    spawnHandlers[0]({ prompt: 'first' })
    spawnHandlers[0]({ prompt: 'second' })
  })

  // The second spawn must NOT have reset the draft yet — doing so is what
  // cancelled the first spawn's in-flight session.create.
  expect(order).toEqual(['fresh', 'submit:first'])
  expect(startFreshSession).toHaveBeenCalledTimes(1)

  await act(async () => {
    release[0]()
    await Promise.resolve()
    await Promise.resolve()
  })

  expect(order).toEqual(['fresh', 'submit:first', 'fresh', 'submit:second'])
})

// A failing spawn must not strand every later one behind a rejected promise.
it('keeps running later spawns after one fails', async () => {
  const startFreshSession = vi.fn()

  const submitText = vi.fn(async (text: string, _options?: SubmitTextOptions) => {
    if (text === 'boom') {
      throw new Error('submit failed')
    }

    return true
  })

  renderHook(() => useSpawnBridge({ startFreshSession, submitText, updateSessionState: stateDouble() }))

  await act(async () => {
    spawnHandlers[0]({ prompt: 'boom' })
    spawnHandlers[0]({ prompt: 'after' })
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()
  })

  expect(submitText.mock.calls.map(c => c[0])).toEqual(['boom', 'after'])
})

it('tells main it is listening so a spawn racing boot gets flushed', () => {
  mount()

  expect(readySignals).toBe(1)
})

it('a secondary session window neither subscribes nor signals ready', () => {
  secondary.value = true
  mount()

  expect(spawnHandlers).toHaveLength(0)
  expect(readySignals).toBe(0)
})

it('unsubscribes on unmount', () => {
  const startFreshSession = vi.fn()
  const submitText = vi.fn(async (_text: string, _options?: SubmitTextOptions) => true)

  const { unmount } = renderHook(() =>
    useSpawnBridge({ startFreshSession, submitText, updateSessionState: stateDouble() })
  )

  unmount()

  expect(unsubscribes).toBe(1)
})

// An older preload has no bridge; mounting must not throw.
it('degrades quietly when the preload has no spawn bridge', () => {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {}

  expect(() => mount()).not.toThrow()
})

// FAIL-BEFORE. `startFreshSession` navigates to the new-chat route and that
// navigation lands asynchronously. Submitting in the same tick captured the
// PREVIOUS route token, so the submit saw the route move under it mid-flight,
// treated it as the user switching away, and aborted — after the CLI had
// already been told 202. Measured against the sandbox app on 2026-08-02: the
// first spawn after boot ran with freshDraftReady=false and returned ok=false;
// the second, by which point the draft had settled, returned ok=true.
it('waits for the new-chat draft to settle before submitting', async () => {
  setFreshDraftReady(false)

  const { startFreshSession, submitText } = mount()

  await act(async () => {
    spawnHandlers[0]({ prompt: 'go' })
    await Promise.resolve()
    await Promise.resolve()
  })

  // The draft has not settled, so the prompt must not have been submitted yet.
  expect(startFreshSession).toHaveBeenCalledTimes(1)
  expect(submitText).not.toHaveBeenCalled()

  await act(async () => {
    setFreshDraftReady(true)
    await Promise.resolve()
    await Promise.resolve()
  })

  expect(submitText).toHaveBeenCalledWith('go', { attachments: [] })
})

// A draft flag that never arrives must not swallow the spawn entirely.
it('submits anyway if the draft never reports ready', async () => {
  vi.useFakeTimers()
  setFreshDraftReady(false)

  try {
    const { submitText } = mount()

    await act(async () => {
      spawnHandlers[0]({ prompt: 'go' })
      await Promise.resolve()
    })

    expect(submitText).not.toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(SPAWN_DRAFT_SETTLE_TIMEOUT_MS)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(submitText).toHaveBeenCalledTimes(1)
  } finally {
    vi.useRealTimers()
  }
})

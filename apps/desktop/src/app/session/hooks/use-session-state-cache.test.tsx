import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { createFallbackNotice, deferFallbackNotice, restoreFallbackNotices } from '@/lib/fallback-notices'
import {
  $activeSessionStoredIdRotation,
  $currentFallbackPolicy,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $currentServiceTier,
  $currentUsage,
  $messages,
  $primaryFastMode,
  $primaryModel,
  $primaryProvider,
  $primaryReasoningEffort,
  $turnStartedAt,
  clearActiveSessionRuntime,
  setActiveSessionId,
  setActiveSessionStoredIdRotation,
  setCurrentFallbackPolicy,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setCurrentUsage,
  setTurnStartedAt
} from '@/store/session'

import { useSessionStateCache } from './use-session-state-cache'

type Cache = ReturnType<typeof useSessionStateCache>

interface HarnessProps {
  activeSessionId: string | null
  onReady: (cache: Cache) => void
  selectedStoredSessionId: string | null
}

describe('useSessionStateCache — stored-id rotation provenance', () => {
  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    setActiveSessionStoredIdRotation(null)
  })

  it('emits the previous, next, and runtime ids and removes the stale reverse mapping', () => {
    let cache!: Cache

    setActiveSessionId('runtime-A')
    render(
      <Harness activeSessionId="runtime-A" onReady={value => (cache = value)} selectedStoredSessionId="stored-A" />
    )

    act(() => {
      cache.updateSessionState('runtime-A', state => state, 'stored-A')
      cache.updateSessionState('runtime-A', state => state, 'stored-A-next')
    })

    expect($activeSessionStoredIdRotation.get()).toEqual({
      nextStoredSessionId: 'stored-A-next',
      previousStoredSessionId: 'stored-A',
      runtimeSessionId: 'runtime-A'
    })
    expect(cache.runtimeIdByStoredSessionIdRef.current.has('stored-A')).toBe(false)
    expect(cache.runtimeIdByStoredSessionIdRef.current.get('stored-A-next')).toBe('runtime-A')
  })

  it('does not publish a foreground-navigation event for a background runtime rotation', () => {
    let cache!: Cache

    setActiveSessionId('runtime-B')
    render(
      <Harness activeSessionId="runtime-B" onReady={value => (cache = value)} selectedStoredSessionId="stored-B" />
    )

    act(() => {
      cache.updateSessionState('runtime-A', state => state, 'stored-A')
      cache.updateSessionState('runtime-A', state => state, 'stored-A-next')
    })

    expect($activeSessionStoredIdRotation.get()).toBeNull()
    expect(cache.runtimeIdByStoredSessionIdRef.current.has('stored-A')).toBe(false)
    expect(cache.runtimeIdByStoredSessionIdRef.current.get('stored-A-next')).toBe('runtime-A')
  })
})

function Harness({ activeSessionId, onReady, selectedStoredSessionId }: HarnessProps) {
  const busyRef: MutableRefObject<boolean> = { current: false }

  const cache = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId,
    setAwaitingResponse: () => undefined,
    setBusy: () => undefined,
    setMessages: () => undefined
  })

  onReady(cache)

  return null
}

describe('useSessionStateCache — per-session turn timer', () => {
  beforeEach(() => {
    // The view-sync flush runs on a real rAF in the browser path; in jsdom we
    // want it synchronous so the global mirror is observable immediately. The
    // hook closes over `window.requestAnimationFrame`, so stub that exact ref.
    // Return null (not a handle) so the hook's `viewSyncRafRef.current = rAF(...)`
    // assignment doesn't overwrite the null the synchronous callback just set —
    // otherwise the ref reads truthy and the NEXT sync is suppressed (a real
    // browser returns a handle but runs the callback async, so this race is a
    // test-only artifact of firing synchronously).
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb: FrameRequestCallback) => {
      cb(0)

      return null as unknown as number
    })
    window.localStorage.clear()
    setTurnStartedAt(null)
    clearActiveSessionRuntime()
    setCurrentModel('')
    setCurrentProvider('')
    setCurrentReasoningEffort('')
    setCurrentServiceTier('')
    setCurrentFastMode(false)
    setCurrentFallbackPolicy('')
    setCurrentUsage({ calls: 0, input: 0, output: 0, total: 0 })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    setTurnStartedAt(null)
    clearActiveSessionRuntime()
    setCurrentModel('')
    setCurrentProvider('')
    setCurrentReasoningEffort('')
    setCurrentServiceTier('')
    setCurrentFastMode(false)
    setCurrentFallbackPolicy('')
    setCurrentUsage({ calls: 0, input: 0, output: 0, total: 0 })
  })

  it("keeps a background session's running turn clock and never mirrors it to the view", () => {
    let cache!: Cache
    // Active session is "fg-runtime"; the turn starts on the BACKGROUND session.
    render(<Harness activeSessionId="fg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="fg-stored" />)

    const startedAt = 1_700_000_000_000

    act(() => {
      cache.updateSessionState('bg-runtime', state => ({ ...state, busy: true, turnStartedAt: startedAt }), 'bg-stored')
    })

    // The background session's own cache entry holds the clock...
    expect(cache.sessionStateByRuntimeIdRef.current.get('bg-runtime')?.turnStartedAt).toBe(startedAt)
    // ...but the global atom (statusbar timer) is untouched — a background turn
    // must not drive the foreground timer.
    expect($turnStartedAt.get()).toBeNull()
  })

  it('backfills a pre-bind fallback notice when the stored session id becomes known', () => {
    let cache!: Cache
    const notice = createFallbackNotice('init fallback', 1_700_000_000_000)
    deferFallbackNotice('fg-runtime', notice, 1)

    render(<Harness activeSessionId="fg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="fg-stored" />)

    act(() => {
      cache.ensureSessionState('fg-runtime', 'fg-stored')
    })

    const restored = restoreFallbackNotices('fg-stored', [userMessage('u', 'hello'), assistantText('a', 'answer')])
    expect(restored.map(message => message.id)).toEqual(['u', notice.id, 'a'])
  })

  it("mirrors the focused session's turn clock into the global atom on view-sync", () => {
    let cache!: Cache
    render(<Harness activeSessionId="fg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="fg-stored" />)

    const startedAt = 1_700_000_111_000

    // A turn on the ACTIVE session stages into the view; the flush mirrors its
    // turnStartedAt into the global atom the statusbar reads.
    act(() => {
      cache.updateSessionState('fg-runtime', state => ({ ...state, busy: true, turnStartedAt: startedAt }), 'fg-stored')
    })

    expect($turnStartedAt.get()).toBe(startedAt)
  })

  it('clears the global clock when the focused turn ends', () => {
    let cache!: Cache
    render(<Harness activeSessionId="fg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="fg-stored" />)

    act(() => {
      cache.updateSessionState(
        'fg-runtime',
        state => ({ ...state, busy: true, turnStartedAt: 1_700_000_222_000 }),
        'fg-stored'
      )
    })
    expect($turnStartedAt.get()).toBe(1_700_000_222_000)

    act(() => {
      cache.updateSessionState('fg-runtime', state => ({ ...state, busy: false, turnStartedAt: null }))
    })
    expect($turnStartedAt.get()).toBeNull()
  })

  it('mirrors the focused session model metadata when switching from a cached session', () => {
    let cache!: Cache

    // The composer's sticky pick — what a NEW chat would be created with. The
    // focused session's model/effort/fast must paint over it on screen without
    // overwriting it in storage.
    setCurrentModel('anthropic/claude-sonnet-4.6')
    setCurrentProvider('anthropic')
    setCurrentReasoningEffort('low')
    setCurrentFastMode(false)

    const { rerender } = render(
      <Harness activeSessionId="fg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="fg-stored" />
    )

    act(() => {
      cache.updateSessionState(
        'bg-runtime',
        state => ({
          ...state,
          fast: true,
          fallbackPolicy: 'local-only',
          model: 'anthropic/claude-opus-4.8',
          provider: 'anthropic',
          reasoningEffort: 'high',
          serviceTier: 'priority'
        }),
        'bg-stored'
      )
    })

    // Background metadata is cached but must not bleed into the visible statusbar.
    expect($primaryModel.get()).toBe('anthropic/claude-sonnet-4.6')
    expect($primaryReasoningEffort.get()).toBe('low')
    expect($primaryFastMode.get()).toBe(false)
    expect($currentFallbackPolicy.get()).toBe('')

    rerender(<Harness activeSessionId="bg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="bg-stored" />)

    const bgState = cache.sessionStateByRuntimeIdRef.current.get('bg-runtime')
    expect(bgState).toBeTruthy()

    act(() => {
      cache.syncSessionStateToView('bg-runtime', bgState!)
    })

    expect($primaryModel.get()).toBe('anthropic/claude-opus-4.8')
    expect($primaryProvider.get()).toBe('anthropic')
    expect($primaryReasoningEffort.get()).toBe('high')
    expect($currentServiceTier.get()).toBe('priority')
    expect($primaryFastMode.get()).toBe(true)
    expect($currentFallbackPolicy.get()).toBe('local-only')

    // ...and the composer's persisted pick is untouched, so the next new chat
    // still starts on the model/effort/fast the user chose.
    expect($currentModel.get()).toBe('anthropic/claude-sonnet-4.6')
    expect($currentProvider.get()).toBe('anthropic')
    expect($currentReasoningEffort.get()).toBe('low')
    expect($currentFastMode.get()).toBe(false)
  })

  it('clears stale model metadata when the newly focused session has no cached value', () => {
    setCurrentModel('previous-model')
    setCurrentProvider('previous-provider')
    setCurrentReasoningEffort('high')
    setCurrentServiceTier('priority')
    setCurrentFastMode(true)
    setCurrentFallbackPolicy('any')

    let cache!: Cache

    const { rerender } = render(
      <Harness activeSessionId="fg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="fg-stored" />
    )

    act(() => {
      cache.updateSessionState('bg-runtime', state => ({ ...state }), 'bg-stored')
    })

    rerender(<Harness activeSessionId="bg-runtime" onReady={c => (cache = c)} selectedStoredSessionId="bg-stored" />)

    const bgState = cache.sessionStateByRuntimeIdRef.current.get('bg-runtime')
    expect(bgState).toBeTruthy()

    act(() => {
      cache.syncSessionStateToView('bg-runtime', bgState!)
    })

    // The focused session reported no model/effort/fast, so the display blanks
    // (the pill's loader) rather than keeping the previous session's — but the
    // composer's own pick survives in storage for the next new chat.
    expect($primaryModel.get()).toBe('')
    expect($primaryProvider.get()).toBe('')
    expect($primaryReasoningEffort.get()).toBe('')
    expect($primaryFastMode.get()).toBe(false)
    expect($currentModel.get()).toBe('previous-model')
    expect($currentProvider.get()).toBe('previous-provider')
    expect($currentReasoningEffort.get()).toBe('high')
    expect($currentServiceTier.get()).toBe('')
    expect($currentFastMode.get()).toBe(true)
    expect($currentFallbackPolicy.get()).toBe('')
  })

  // The DURABLE half of the #298 leak: `state.model` is stamped from every
  // session.info, and the backend puts `model` on every one of them. Routing
  // that through setCurrentModel rewrote the user's persisted pick on each
  // heartbeat, so a `hermes desktop spawn -m <model>` re-aimed every chat
  // started afterwards — and kept re-aiming it for the session's whole life.
  it("does not write a spawned session's model into the persisted composer pick", () => {
    setCurrentModel('anthropic/claude-sonnet-4.6')
    setCurrentProvider('anthropic')

    let cache!: Cache
    render(<Harness activeSessionId="spawn-runtime" onReady={c => (cache = c)} selectedStoredSessionId="spawn-stored" />)

    // Three heartbeats, as a live session emits over its lifetime.
    for (let beat = 0; beat < 3; beat += 1) {
      act(() => {
        cache.updateSessionState(
          'spawn-runtime',
          state => ({ ...state, model: 'deepseek-v4-flash-0731-ds4', provider: 'ai-router' }),
          'spawn-stored'
        )
      })
    }

    // The spawned session shows its own model...
    expect($primaryModel.get()).toBe('deepseek-v4-flash-0731-ds4')
    expect($primaryProvider.get()).toBe('ai-router')

    // ...and the next new chat still gets the user's pick, from memory and
    // from the localStorage the composer re-seeds itself from on restart.
    expect($currentModel.get()).toBe('anthropic/claude-sonnet-4.6')
    expect($currentProvider.get()).toBe('anthropic')
    expect(window.localStorage.getItem('hermes.desktop.composer.model')).toBe('anthropic/claude-sonnet-4.6')
  })

  // The same #298-shaped leak, for the other two sticky composer fields (#318
  // only fixed model/provider). `state.reasoningEffort`/`state.fast` are
  // stamped from every session.info too, so routing them through
  // setCurrentReasoningEffort/setCurrentFastMode here rewrote the user's
  // persisted pick on every heartbeat of a session running a different value.
  it("does not write a focused session's effort/fast into the persisted composer pick", () => {
    setCurrentReasoningEffort('high')
    setCurrentFastMode(true)

    let cache!: Cache
    render(<Harness activeSessionId="spawn-runtime" onReady={c => (cache = c)} selectedStoredSessionId="spawn-stored" />)

    // Three heartbeats, as a live session emits over its lifetime.
    for (let beat = 0; beat < 3; beat += 1) {
      act(() => {
        cache.updateSessionState(
          'spawn-runtime',
          state => ({ ...state, fast: false, reasoningEffort: 'low' }),
          'spawn-stored'
        )
      })
    }

    // The focused session shows its own effort/fast...
    expect($primaryReasoningEffort.get()).toBe('low')
    expect($primaryFastMode.get()).toBe(false)

    // ...and the next new chat still gets the user's pick, from memory and
    // from the localStorage the composer re-seeds itself from on restart.
    expect($currentReasoningEffort.get()).toBe('high')
    expect($currentFastMode.get()).toBe(true)
    expect(window.localStorage.getItem('hermes.desktop.composer.reasoning-effort')).toBe('high')
    expect(window.localStorage.getItem('hermes.desktop.composer.fast')).toBe('true')
  })

  it('switches the context meter immediately to the newly focused cached session', () => {
    let cache!: Cache

    const { rerender } = render(
      <Harness activeSessionId="session-a" onReady={c => (cache = c)} selectedStoredSessionId="stored-a" />
    )

    act(() => {
      cache.updateSessionState(
        'session-a',
        state => ({
          ...state,
          usage: { ...state.usage, context_max: 100_000, context_percent: 72, context_used: 72_000 }
        }),
        'stored-a'
      )
      cache.updateSessionState(
        'session-b',
        state => ({
          ...state,
          usage: { ...state.usage, context_max: 100_000, context_percent: 18, context_used: 18_000 }
        }),
        'stored-b'
      )
    })

    expect($currentUsage.get().context_percent).toBe(72)

    rerender(<Harness activeSessionId="session-b" onReady={c => (cache = c)} selectedStoredSessionId="stored-b" />)
    const sessionB = cache.sessionStateByRuntimeIdRef.current.get('session-b')

    act(() => cache.syncSessionStateToView('session-b', sessionB!))

    expect($currentUsage.get()).toMatchObject({ context_percent: 18, context_used: 18_000 })
  })

  it('keeps a background session usage update out of the focused context meter', () => {
    let cache!: Cache
    render(<Harness activeSessionId="session-a" onReady={c => (cache = c)} selectedStoredSessionId="stored-a" />)

    act(() => {
      cache.updateSessionState(
        'session-a',
        state => ({ ...state, usage: { ...state.usage, context_percent: 42, context_used: 42_000 } }),
        'stored-a'
      )
      cache.updateSessionState(
        'session-b',
        state => ({ ...state, usage: { ...state.usage, context_percent: 91, context_used: 91_000 } }),
        'stored-b'
      )
    })

    expect(cache.sessionStateByRuntimeIdRef.current.get('session-b')?.usage.context_percent).toBe(91)
    expect($currentUsage.get().context_percent).toBe(42)
  })
})

function userMessage(id: string, text: string): ChatMessage {
  return { id, role: 'user', parts: [{ type: 'text', text }] }
}

function assistantText(id: string, text: string): ChatMessage {
  return { id, role: 'assistant', parts: [{ type: 'text', text }] }
}

function assistantError(id: string, error: string): ChatMessage {
  return { id, role: 'assistant', parts: [], error, pending: false }
}

interface ViewHarnessProps {
  activeSessionId: string | null
  onReady: (cache: Cache) => void
}

function ViewHarness({ activeSessionId, onReady }: ViewHarnessProps) {
  const busyRef: MutableRefObject<boolean> = { current: false }

  const cache = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId: null,
    setAwaitingResponse: () => undefined,
    setBusy: () => undefined,
    // Wire the published view back into the real $messages atom the flush
    // reads from, so the round-trip matches production.
    setMessages: messages => $messages.set(messages)
  })

  onReady(cache)

  return null
}

describe('useSessionStateCache — cross-thread error isolation', () => {
  afterEach(() => {
    cleanup()
    $messages.set([])
  })

  it('does not leak a failed turn into another thread on switch', () => {
    $messages.set([])
    let cache!: Cache
    const { rerender } = render(<ViewHarness activeSessionId="thread-A" onReady={c => (cache = c)} />)

    // Thread A ends its turn with an out-of-funds error and is on screen.
    act(() => {
      cache.updateSessionState(
        'thread-A',
        state => ({
          ...state,
          busy: false,
          messages: [userMessage('user-a', 'do the thing'), assistantError('assistant-a-error', 'Out of funds')]
        }),
        'stored-A'
      )
    })

    expect($messages.get().some(message => message.error === 'Out of funds')).toBe(true)

    // Switch to thread B (which completed cleanly). Its cached state syncs to
    // the view while $messages still holds thread A's transcript.
    rerender(<ViewHarness activeSessionId="thread-B" onReady={c => (cache = c)} />)
    act(() => {
      cache.updateSessionState(
        'thread-B',
        state => ({
          ...state,
          busy: false,
          messages: [userMessage('user-b', 'hello'), assistantText('assistant-b', 'hi there')]
        }),
        'stored-B'
      )
    })

    expect($messages.get().map(message => message.id)).toEqual(['user-b', 'assistant-b'])
    expect($messages.get().some(message => message.error === 'Out of funds')).toBe(false)
  })

  it('still preserves a same-session local error a heartbeat dropped', () => {
    $messages.set([])
    let cache!: Cache
    render(<ViewHarness activeSessionId="thread-A" onReady={c => (cache = c)} />)

    // First paint establishes thread A as the on-screen session.
    act(() => {
      cache.updateSessionState(
        'thread-A',
        state => ({ ...state, busy: false, messages: [userMessage('user-a', 'do the thing')] }),
        'stored-A'
      )
    })

    // A local error lands in the view (e.g. failAssistantMessage wrote it).
    $messages.set([userMessage('user-a', 'do the thing'), assistantError('assistant-a-error', 'OpenRouter 403')])

    // A later same-session heartbeat carries cached state that lost the error.
    act(() => {
      cache.updateSessionState('thread-A', state => ({
        ...state,
        busy: false,
        messages: [userMessage('user-a', 'do the thing')]
      }))
    })

    expect($messages.get().some(message => message.error === 'OpenRouter 403')).toBe(true)
  })

  it('only returns a runtime whose cached state owns the requested stored session', () => {
    let cache!: Cache
    render(<Harness activeSessionId={null} onReady={value => (cache = value)} selectedStoredSessionId={null} />)

    act(() => {
      cache.ensureSessionState('runtime-A', 'stored-A')
      cache.ensureSessionState('runtime-B', 'stored-B')
    })

    expect(cache.getRuntimeIdForStoredSession('stored-A')).toBe('runtime-A')
    expect(cache.getRuntimeIdForStoredSession('missing')).toBeNull()

    // Simulate a recycled/cross-wired map entry. The reverse state ownership
    // check must reject it instead of allowing a submit into stored-B.
    cache.runtimeIdByStoredSessionIdRef.current.set('stored-A', 'runtime-B')
    expect(cache.getRuntimeIdForStoredSession('stored-A')).toBeNull()
  })
})

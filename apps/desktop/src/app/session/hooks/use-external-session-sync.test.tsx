import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { EXTERNAL_SESSION_POLL_MS, useExternalSessionSync } from './use-external-session-sync'

// Sessions created outside this app (headless `hermes -z …`, cron, a CLI
// session in another terminal) used to need a manual View > Reload to appear.
// This hook is the renderer half of the fix: an Electron-main watch signal plus
// a focused-only safety-net poll, funnelled through one non-overlapping refresh.

const secondary = vi.hoisted(() => ({ value: false }))

vi.mock('@/store/windows', () => ({
  isSecondaryWindow: () => secondary.value
}))

let storeChangedHandlers: Array<() => void> = []
let unsubscribes = 0
let focused = true

function stubDesktopBridge() {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    onSessionsStoreChanged: (callback: () => void) => {
      storeChangedHandlers.push(callback)

      return () => {
        unsubscribes += 1
        storeChangedHandlers = storeChangedHandlers.filter(h => h !== callback)
      }
    }
  }
}

beforeEach(() => {
  vi.useFakeTimers()
  secondary.value = false
  storeChangedHandlers = []
  unsubscribes = 0
  focused = true
  vi.spyOn(document, 'hasFocus').mockImplementation(() => focused)
  stubDesktopBridge()
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

const setFocus = (next: boolean) => {
  focused = next
  window.dispatchEvent(new Event(next ? 'focus' : 'blur'))
}

it('refreshes the session list when Electron main reports a store change', async () => {
  const refreshSessions = vi.fn(async () => undefined)

  renderHook(() => useExternalSessionSync({ refreshSessions }))

  expect(storeChangedHandlers).toHaveLength(1)

  await act(async () => {
    storeChangedHandlers[0]()
  })

  expect(refreshSessions).toHaveBeenCalledTimes(1)
})

// REGRESSION GUARD. An earlier interval poll here (fork commit 209a9c38d) fired
// every 30s against an API call that could block for 45s while the backend was
// wedged. Overlapping refreshes meant `refreshSessions` only ever cleared its
// loading flag for the LATEST request id, so the sidebar skeletons never
// cleared — a recoverable stall became a permanent spinner. Signals arriving
// during an in-flight refresh must coalesce into ONE trailing pass, never stack.
it('never overlaps refreshes; concurrent signals coalesce into one trailing pass', async () => {
  let release: (() => void) | null = null

  const refreshSessions = vi.fn(
    () =>
      new Promise<void>(resolve => {
        release = () => resolve()
      })
  )

  renderHook(() => useExternalSessionSync({ refreshSessions }))

  await act(async () => {
    storeChangedHandlers[0]()
  })

  expect(refreshSessions).toHaveBeenCalledTimes(1)

  // Five more signals while the first refresh is still pending.
  await act(async () => {
    for (let i = 0; i < 5; i++) {
      storeChangedHandlers[0]()
    }
  })

  expect(refreshSessions).toHaveBeenCalledTimes(1)

  const first = release!

  await act(async () => {
    first()
    await Promise.resolve()
  })

  // Exactly one trailing pass covers all five, not five more calls.
  expect(refreshSessions).toHaveBeenCalledTimes(2)

  await act(async () => {
    release!()
    await Promise.resolve()
  })

  expect(refreshSessions).toHaveBeenCalledTimes(2)
})

it('a rejected refresh does not wedge the hook for later signals', async () => {
  const refreshSessions = vi.fn(async () => {
    throw new Error('backend unreachable')
  })

  renderHook(() => useExternalSessionSync({ refreshSessions }))

  await act(async () => {
    storeChangedHandlers[0]()
  })

  expect(refreshSessions).toHaveBeenCalledTimes(1)

  await act(async () => {
    storeChangedHandlers[0]()
  })

  expect(refreshSessions).toHaveBeenCalledTimes(2)
})

it('polls on an interval while focused', async () => {
  const refreshSessions = vi.fn(async () => undefined)

  renderHook(() => useExternalSessionSync({ refreshSessions }))

  expect(refreshSessions).not.toHaveBeenCalled()

  await act(async () => {
    vi.advanceTimersByTime(EXTERNAL_SESSION_POLL_MS)
  })

  expect(refreshSessions).toHaveBeenCalledTimes(1)

  await act(async () => {
    vi.advanceTimersByTime(EXTERNAL_SESSION_POLL_MS)
  })

  expect(refreshSessions).toHaveBeenCalledTimes(2)
})

it('stops polling entirely while the window is unfocused', async () => {
  const refreshSessions = vi.fn(async () => undefined)

  renderHook(() => useExternalSessionSync({ refreshSessions }))

  await act(async () => {
    setFocus(false)
  })

  await act(async () => {
    vi.advanceTimersByTime(EXTERNAL_SESSION_POLL_MS * 10)
  })

  expect(refreshSessions).not.toHaveBeenCalled()
})

// The watch signal cannot be trusted across a long background stretch (events
// can be dropped, and a store on a network mount may not emit at all), so
// coming back to the app catches up immediately rather than waiting a tick.
it('refreshes once on regaining focus, and not on losing it', async () => {
  const refreshSessions = vi.fn(async () => undefined)

  renderHook(() => useExternalSessionSync({ refreshSessions }))

  await act(async () => {
    setFocus(false)
  })

  expect(refreshSessions).not.toHaveBeenCalled()

  await act(async () => {
    setFocus(true)
  })

  expect(refreshSessions).toHaveBeenCalledTimes(1)
})

it('a secondary session window neither subscribes nor polls', async () => {
  secondary.value = true

  const refreshSessions = vi.fn(async () => undefined)

  renderHook(() => useExternalSessionSync({ refreshSessions }))

  expect(storeChangedHandlers).toHaveLength(0)

  await act(async () => {
    vi.advanceTimersByTime(EXTERNAL_SESSION_POLL_MS * 5)
  })

  expect(refreshSessions).not.toHaveBeenCalled()
})

it('unsubscribes and clears its timer on unmount', async () => {
  const refreshSessions = vi.fn(async () => undefined)

  const { unmount } = renderHook(() => useExternalSessionSync({ refreshSessions }))

  unmount()

  expect(unsubscribes).toBe(1)

  await act(async () => {
    vi.advanceTimersByTime(EXTERNAL_SESSION_POLL_MS * 5)
  })

  expect(refreshSessions).not.toHaveBeenCalled()
})

// An older preload (app updated ahead of a cached bundle) exposes no
// onSessionsStoreChanged. The poll must still work rather than throwing.
it('degrades to the poll when the preload has no store-change bridge', async () => {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {}

  const refreshSessions = vi.fn(async () => undefined)

  renderHook(() => useExternalSessionSync({ refreshSessions }))

  await act(async () => {
    vi.advanceTimersByTime(EXTERNAL_SESSION_POLL_MS)
  })

  expect(refreshSessions).toHaveBeenCalledTimes(1)
})

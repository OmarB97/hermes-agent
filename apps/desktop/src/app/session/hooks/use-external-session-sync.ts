import { useCallback, useEffect, useRef } from 'react'

import { isSecondaryWindow } from '@/store/windows'

/**
 * Safety-net poll interval, used ONLY while the window is focused and visible.
 *
 * fs.watch is the primary signal (Electron main pings us on transcript writes);
 * this covers the cases it structurally cannot see — a store on a network mount,
 * a platform where the watch failed to bind, or events dropped while the app was
 * in the background.
 */
export const EXTERNAL_SESSION_POLL_MS = 30_000

interface UseExternalSessionSyncArgs {
  refreshSessions: () => Promise<unknown> | unknown
}

/**
 * Keep the sidebar current with sessions this app did not create.
 *
 * A headless `hermes -z …` run, a cron job, or a `hermes` CLI session in another
 * terminal writes into the same profile stores the sidebar lists. The renderer
 * used to learn about session changes only from its OWN `message.complete`
 * events and the boot fetch, so those rows needed a manual View > Reload.
 *
 * Two inputs, one coalesced refresh:
 *   1. `onSessionsStoreChanged` — Electron main watches each profile's
 *      transcript directory and pings us (throttled there).
 *   2. A focused-only poll + a refresh when the window regains focus.
 *
 * NON-OVERLAP IS LOAD-BEARING, not tidiness. An earlier interval poll here
 * (fork commit 209a9c38d) fired every 30s against an API call that could block
 * for 45s while the backend was wedged. The refreshes overlapped continuously,
 * and because `refreshSessions` only clears its loading flag for the LATEST
 * request id, the sidebar skeletons never cleared — a recoverable backend stall
 * became a permanent spinner. So: at most one refresh in flight, with a single
 * trailing re-run if signals arrived while it was running.
 */
export function useExternalSessionSync({ refreshSessions }: UseExternalSessionSyncArgs): void {
  // Held in a ref so the subscription and the interval bind ONCE. `refreshSessions`
  // is rebuilt whenever the sidebar's profile scope changes; depending on it
  // directly would tear down and re-add the IPC listener and restart the poll
  // clock on every such change. Assigned in an effect, not during render —
  // a concurrent render that React discards must not publish its closure.
  const refreshRef = useRef(refreshSessions)

  useEffect(() => {
    refreshRef.current = refreshSessions
  }, [refreshSessions])

  const inFlightRef = useRef(false)
  const queuedRef = useRef(false)

  const run = useCallback(async () => {
    if (inFlightRef.current) {
      // Coalesce: whatever arrived is covered by one more pass after this one.
      queuedRef.current = true

      return
    }

    inFlightRef.current = true

    try {
      do {
        queuedRef.current = false
        await refreshRef.current()
      } while (queuedRef.current)
    } catch {
      // Non-fatal: the sidebar keeps its last-known rows and the next signal
      // (or poll tick) retries.
    } finally {
      queuedRef.current = false
      inFlightRef.current = false
    }
  }, [])

  // Signal 1 — Electron main saw a transcript land in a watched profile store.
  useEffect(() => {
    if (isSecondaryWindow()) {
      return
    }

    const unsubscribe = window.hermesDesktop?.onSessionsStoreChanged?.(() => void run())

    return () => unsubscribe?.()
  }, [run])

  // Signal 2 — focused-only safety net. No timer runs while the window is
  // blurred or hidden, so a backgrounded app costs nothing.
  useEffect(() => {
    if (isSecondaryWindow()) {
      return
    }

    let timer: null | ReturnType<typeof setInterval> = null

    const stop = () => {
      if (timer) {
        clearInterval(timer)
        timer = null
      }
    }

    const start = () => {
      if (!timer) {
        timer = setInterval(() => void run(), EXTERNAL_SESSION_POLL_MS)
      }
    }

    const active = () => document.hasFocus() && document.visibilityState !== 'hidden'

    const sync = () => (active() ? start() : stop())

    // Coming back to the app is the moment stale rows are most visible, and it
    // is also when watch events dropped in the background need catching up.
    // Going away only stops the timer — never refresh on the way out.
    const onActivate = () => {
      sync()

      if (active()) {
        void run()
      }
    }

    sync()

    window.addEventListener('focus', onActivate)
    window.addEventListener('blur', sync)
    document.addEventListener('visibilitychange', onActivate)

    return () => {
      stop()
      window.removeEventListener('focus', onActivate)
      window.removeEventListener('blur', sync)
      document.removeEventListener('visibilitychange', onActivate)
    }
  }, [run])
}

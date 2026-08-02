/**
 * Unit tests for the session-store watcher that keeps the sidebar current with
 * sessions this app did not create (headless `hermes -z …`, cron, a CLI session
 * in another terminal).
 *
 * Two behaviours are load-bearing and get the most attention:
 *   - We watch each profile's `sessions/` transcript directory, NOT the profile
 *     home. The home root is unusable: measured on macOS it churns every ~5s
 *     from kanban.db WAL renames alone, so a home watch would re-query every
 *     profile DB forever.
 *   - The throttle must not STARVE. A plain trailing debounce restarts its timer
 *     on every event, and an agent mid-run rewrites its transcript each turn —
 *     so the notify could be deferred for the entire run, which is exactly the
 *     case this feature exists for.
 */

import assert from 'node:assert/strict'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

import {
  createSessionStoreWatcher,
  SESSION_STORE_DEBOUNCE_MS,
  SESSION_STORE_RESCAN_MS,
  sessionStoreWatchDirs
} from './session-store-watch'

const HOME = path.join('/tmp', 'hermes-home')

/** In-memory fs stub: `dirs` is the set of paths that exist as directories. */
function fakeFs(dirs: string[], watchLog?: Array<{ dir: string; handler: () => void }>) {
  const set = new Set(dirs)

  return {
    readdirSync: ((target: string) => {
      const prefix = `${target}${path.sep}`
      const children = new Set<string>()

      for (const dir of set) {
        if (dir.startsWith(prefix)) {
          children.add(dir.slice(prefix.length).split(path.sep)[0])
        }
      }

      if (!set.has(target) && children.size === 0) {
        throw new Error(`ENOENT: ${target}`)
      }

      return [...children]
    }) as never,
    statSync: ((target: string) => {
      if (!set.has(target)) {
        throw new Error(`ENOENT: ${target}`)
      }

      return { isDirectory: () => true }
    }) as never,
    watch: ((dir: string, handler: () => void) => {
      watchLog?.push({ dir, handler })

      return { close: () => {}, on: () => {} }
    }) as never
  }
}

afterEach(() => {
  vi.useRealTimers()
})

test('enumerates the default home plus every named profile transcript dir', () => {
  const io = fakeFs([
    path.join(HOME, 'sessions'),
    path.join(HOME, 'profiles', 'work'),
    path.join(HOME, 'profiles', 'work', 'sessions'),
    path.join(HOME, 'profiles', 'meshboard-worker'),
    path.join(HOME, 'profiles', 'meshboard-worker', 'sessions')
  ])

  assert.deepEqual(sessionStoreWatchDirs(HOME, io), [
    path.join(HOME, 'sessions'),
    path.join(HOME, 'profiles', 'meshboard-worker', 'sessions'),
    path.join(HOME, 'profiles', 'work', 'sessions')
  ])
})

// A profile that has never run a session has no transcripts to watch. Skipping
// (rather than creating) keeps the watcher read-only against the user's store;
// the periodic rescan binds it the moment the directory appears.
test('skips profiles whose transcript dir does not exist yet', () => {
  const io = fakeFs([path.join(HOME, 'sessions'), path.join(HOME, 'profiles', 'fresh')])

  assert.deepEqual(sessionStoreWatchDirs(HOME, io), [path.join(HOME, 'sessions')])
})

// Mirrors hermes_cli/profiles.py list_profiles(): `default` IS the root home
// (already added), and non-conforming names are not profiles.
test('ignores a nested "default" dir and names the CLI would not accept', () => {
  const io = fakeFs([
    path.join(HOME, 'sessions'),
    path.join(HOME, 'profiles', 'default'),
    path.join(HOME, 'profiles', 'default', 'sessions'),
    path.join(HOME, 'profiles', 'Not Valid'),
    path.join(HOME, 'profiles', 'Not Valid', 'sessions'),
    path.join(HOME, 'profiles', '-leading-dash'),
    path.join(HOME, 'profiles', '-leading-dash', 'sessions')
  ])

  assert.deepEqual(sessionStoreWatchDirs(HOME, io), [path.join(HOME, 'sessions')])
})

test('a home with no profiles dir still watches the default transcripts', () => {
  const io = fakeFs([path.join(HOME, 'sessions')])

  assert.deepEqual(sessionStoreWatchDirs(HOME, io), [path.join(HOME, 'sessions')])
})

test('an empty home yields nothing to watch and does not throw', () => {
  assert.deepEqual(sessionStoreWatchDirs(HOME, fakeFs([])), [])
  assert.deepEqual(sessionStoreWatchDirs('', fakeFs([])), [])
})

test('binds a watch to every discovered transcript dir', () => {
  const log: Array<{ dir: string; handler: () => void }> = []

  const io = fakeFs(
    [path.join(HOME, 'sessions'), path.join(HOME, 'profiles', 'work'), path.join(HOME, 'profiles', 'work', 'sessions')],
    log
  )

  vi.useFakeTimers()

  const watcher = createSessionStoreWatcher({ hermesHome: HOME, notify: () => {}, fsImpl: io })

  assert.deepEqual(
    log.map(entry => entry.dir),
    [path.join(HOME, 'sessions'), path.join(HOME, 'profiles', 'work', 'sessions')]
  )
  assert.equal(watcher.watchedDirs().length, 2)
  watcher.close()
})

test('collapses a burst of transcript writes into a single notify', () => {
  vi.useFakeTimers()

  const log: Array<{ dir: string; handler: () => void }> = []
  const io = fakeFs([path.join(HOME, 'sessions')], log)
  let notifies = 0
  const watcher = createSessionStoreWatcher({ hermesHome: HOME, notify: () => (notifies += 1), fsImpl: io })

  log[0].handler()
  log[0].handler()
  log[0].handler()

  assert.equal(notifies, 0)
  vi.advanceTimersByTime(SESSION_STORE_DEBOUNCE_MS)
  assert.equal(notifies, 1)

  watcher.close()
})

// FAIL-BEFORE (design guard): with a trailing debounce that restarts on every
// event, a long agent run rewriting its transcript faster than the window would
// defer the notify indefinitely — the sidebar would stay stale for the entire
// run. The leading-scheduled throttle guarantees delivery at a bounded rate.
test('does not starve while writes keep arriving faster than the window', () => {
  vi.useFakeTimers()

  const log: Array<{ dir: string; handler: () => void }> = []
  const io = fakeFs([path.join(HOME, 'sessions')], log)
  let notifies = 0
  const watcher = createSessionStoreWatcher({ hermesHome: HOME, notify: () => (notifies += 1), fsImpl: io })

  // A write every third of a window, for ten windows' worth of time.
  for (let i = 0; i < 30; i++) {
    log[0].handler()
    vi.advanceTimersByTime(SESSION_STORE_DEBOUNCE_MS / 3)
  }

  assert.equal(notifies, 10)

  watcher.close()
})

test('a notify that throws does not kill the watcher', () => {
  vi.useFakeTimers()

  const log: Array<{ dir: string; handler: () => void }> = []
  const io = fakeFs([path.join(HOME, 'sessions')], log)
  let notifies = 0

  const watcher = createSessionStoreWatcher({
    hermesHome: HOME,
    notify: () => {
      notifies += 1
      throw new Error('window destroyed')
    },
    fsImpl: io
  })

  log[0].handler()
  vi.advanceTimersByTime(SESSION_STORE_DEBOUNCE_MS)
  assert.equal(notifies, 1)

  log[0].handler()
  vi.advanceTimersByTime(SESSION_STORE_DEBOUNCE_MS)
  assert.equal(notifies, 2)

  watcher.close()
})

test('close stops the pending notify and every watch', () => {
  vi.useFakeTimers()

  const closed: string[] = []
  const log: Array<{ dir: string; handler: () => void }> = []
  const io = fakeFs([path.join(HOME, 'sessions')], log)
  const baseWatch = io.watch as unknown as (dir: string, handler: () => void) => unknown

  io.watch = ((dir: string, handler: () => void) => {
    baseWatch(dir, handler)

    return { close: () => closed.push(dir), on: () => {} }
  }) as never

  let notifies = 0
  const watcher = createSessionStoreWatcher({ hermesHome: HOME, notify: () => (notifies += 1), fsImpl: io })

  log[0].handler()
  watcher.close()
  vi.advanceTimersByTime(SESSION_STORE_DEBOUNCE_MS * 5)

  assert.equal(notifies, 0)
  assert.deepEqual(closed, [path.join(HOME, 'sessions')])
  assert.deepEqual(watcher.watchedDirs(), [])
})

// A profile created (or first used) after boot must start refreshing the
// sidebar without an app restart.
test('the rescan binds a transcript dir that appears after boot', () => {
  vi.useFakeTimers()

  const dirs = [path.join(HOME, 'sessions')]
  const log: Array<{ dir: string; handler: () => void }> = []
  const io = fakeFs(dirs, log)
  // Re-point the stub at a growing set so the rescan sees the new profile.
  const live = new Set(dirs)

  io.statSync = ((target: string) => {
    if (!live.has(target)) {
      throw new Error(`ENOENT: ${target}`)
    }

    return { isDirectory: () => true }
  }) as never
  io.readdirSync = ((target: string) => {
    const prefix = `${target}${path.sep}`
    const children = new Set<string>()

    for (const dir of live) {
      if (dir.startsWith(prefix)) {
        children.add(dir.slice(prefix.length).split(path.sep)[0])
      }
    }

    if (!live.has(target) && children.size === 0) {
      throw new Error(`ENOENT: ${target}`)
    }

    return [...children]
  }) as never

  const watcher = createSessionStoreWatcher({ hermesHome: HOME, notify: () => {}, fsImpl: io })

  assert.equal(watcher.watchedDirs().length, 1)

  live.add(path.join(HOME, 'profiles', 'later'))
  live.add(path.join(HOME, 'profiles', 'later', 'sessions'))

  vi.advanceTimersByTime(SESSION_STORE_RESCAN_MS)

  assert.deepEqual(watcher.watchedDirs(), [
    path.join(HOME, 'sessions'),
    path.join(HOME, 'profiles', 'later', 'sessions')
  ])

  watcher.close()
})

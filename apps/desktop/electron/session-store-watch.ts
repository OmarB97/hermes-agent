import fs from 'node:fs'
import path from 'node:path'

// Live session-list refresh for sessions this app did not create.
//
// The sidebar's rows come from each profile's state.db, but the renderer only
// re-pulled them on boot and on its OWN sessions' `message.complete` events. A
// headless `hermes -z …` run, a cron job, or a plain `hermes` CLI session in
// another terminal writes into the same store and stayed invisible until the
// user hit View > Reload. This watches the store and pings the renderer so its
// existing refresh path runs on its own.
//
// WHICH PATH TO WATCH (measured on macOS 2026-08-01, a ~/.hermes holding 5.5k
// transcripts and a 2.5 GB state.db):
//
//   - The profile HOME directory is unusable as a trigger. Over a quiet 120s
//     window it fired continuously: kanban.db-wal/-shm renames every 5s, each
//     profile's state.db-wal every 10s, plus cron/ and the skills snapshot. A
//     home-root watch would re-query every profile DB forever, whether or not
//     any session existed.
//   - `<home>/sessions/` is quiet by comparison — over that same window it
//     fired only for the transcript writes of the run under test. The agent
//     writes the transcript through a temp file + atomic rename ~3s after a run
//     starts and again after each turn, and that write lands together with the
//     state.db row reaching message_count >= 1, which is exactly when the row
//     becomes sidebar-eligible. So the transcript directory is both the
//     quietest and the most accurate signal.
//
// SQLite WAL is why the obvious alternative fails: state.db itself is barely
// touched between checkpoints, so watching the db file misses live writes, and
// watching the -wal file breaks whenever a checkpoint recreates it.

/** Collapse a burst of transcript writes into one renderer ping. */
export const SESSION_STORE_DEBOUNCE_MS = 1500

/** Re-enumerate profile session dirs, picking up profiles created since boot. */
export const SESSION_STORE_RESCAN_MS = 60_000

// Mirrors hermes_cli/profiles.py `_PROFILE_ID_RE` so we enumerate exactly the
// profile homes `list_profiles()` (and therefore the sessions endpoint) scans.
const PROFILE_ID_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

export interface SessionStoreWatchFs {
  readdirSync: typeof fs.readdirSync
  statSync: typeof fs.statSync
  watch: typeof fs.watch
}

const defaultFs: SessionStoreWatchFs = {
  readdirSync: fs.readdirSync,
  statSync: fs.statSync,
  watch: fs.watch
}

function isDirectory(target: string, io: SessionStoreWatchFs): boolean {
  try {
    return io.statSync(target).isDirectory()
  } catch {
    return false
  }
}

/**
 * Every `<profile home>/sessions` directory that currently exists: the default
 * home plus each named profile under `<home>/profiles`. Missing directories are
 * skipped rather than created — a profile that has never run a session has no
 * transcripts to watch, and the periodic rescan picks it up once it does.
 */
export function sessionStoreWatchDirs(hermesHome: string, io: SessionStoreWatchFs = defaultFs): string[] {
  if (!hermesHome) {
    return []
  }

  const dirs: string[] = []

  const addIfDir = (dir: string) => {
    if (isDirectory(dir, io)) {
      dirs.push(dir)
    }
  }

  addIfDir(path.join(hermesHome, 'sessions'))

  const profilesRoot = path.join(hermesHome, 'profiles')

  let entries: string[] = []

  try {
    entries = io.readdirSync(profilesRoot) as unknown as string[]
  } catch {
    return dirs
  }

  for (const entry of [...entries].map(String).sort()) {
    // `default` is the root home, already added above.
    if (entry === 'default' || !PROFILE_ID_RE.test(entry)) {
      continue
    }

    if (isDirectory(path.join(profilesRoot, entry), io)) {
      addIfDir(path.join(profilesRoot, entry, 'sessions'))
    }
  }

  return dirs
}

export interface SessionStoreWatcher {
  /** Directories under watch right now — exposed for tests/diagnostics. */
  watchedDirs: () => string[]
  close: () => void
}

export interface SessionStoreWatcherOptions {
  hermesHome: string
  /** Called (throttled) when a transcript write suggests the store changed. */
  notify: () => void
  debounceMs?: number
  rescanMs?: number
  fsImpl?: SessionStoreWatchFs
  onLog?: (message: string) => void
}

/**
 * Watch every profile's transcript directory and call `notify` when one moves.
 *
 * Throttle shape matters here. A plain trailing debounce would STARVE: an agent
 * mid-run rewrites its transcript every turn, so a timer that restarts on each
 * event may never fire while a long run is in progress — the exact case this
 * feature exists for. Instead the first event of a burst schedules a single
 * notify `debounceMs` later and every event until then is absorbed. That bounds
 * the rate at one notify per window while guaranteeing the first change in any
 * burst is delivered promptly.
 */
export function createSessionStoreWatcher({
  hermesHome,
  notify,
  debounceMs = SESSION_STORE_DEBOUNCE_MS,
  rescanMs = SESSION_STORE_RESCAN_MS,
  fsImpl = defaultFs,
  onLog
}: SessionStoreWatcherOptions): SessionStoreWatcher {
  const watchers = new Map<string, fs.FSWatcher>()
  let pending: ReturnType<typeof setTimeout> | null = null
  let closed = false

  const schedule = () => {
    // A notify is already queued for this burst — absorb the event.
    if (closed || pending) {
      return
    }

    pending = setTimeout(() => {
      pending = null

      try {
        notify()
      } catch {
        // A dead window / destroyed webContents must not kill the watcher.
      }
    }, debounceMs)
  }

  const bind = (dir: string) => {
    if (watchers.has(dir)) {
      return
    }

    try {
      const watcher = fsImpl.watch(dir, () => schedule())

      // A watched directory can vanish (profile deleted). Drop it and let the
      // rescan re-bind if it comes back, rather than throwing into main.
      watcher.on('error', () => {
        watchers.delete(dir)

        try {
          watcher.close()
        } catch {
          // already gone
        }
      })

      watchers.set(dir, watcher)
      onLog?.(`[session-store-watch] watching ${dir}`)
    } catch (err) {
      onLog?.(`[session-store-watch] failed to watch ${dir}: ${(err as Error)?.message ?? err}`)
    }
  }

  const rescan = () => {
    if (closed) {
      return
    }

    const wanted = new Set(sessionStoreWatchDirs(hermesHome, fsImpl))

    for (const [dir, watcher] of watchers) {
      if (!wanted.has(dir)) {
        watchers.delete(dir)

        try {
          watcher.close()
        } catch {
          // already gone
        }
      }
    }

    for (const dir of wanted) {
      bind(dir)
    }
  }

  rescan()

  const rescanTimer = setInterval(rescan, rescanMs)

  // Never hold the process open just to poll for new profiles.
  rescanTimer.unref?.()

  return {
    watchedDirs: () => [...watchers.keys()],
    close: () => {
      closed = true
      clearInterval(rescanTimer)

      if (pending) {
        clearTimeout(pending)
        pending = null
      }

      for (const watcher of watchers.values()) {
        try {
          watcher.close()
        } catch {
          // already gone
        }
      }

      watchers.clear()
    }
  }
}

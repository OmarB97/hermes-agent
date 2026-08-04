import { atom } from 'nanostores'

import type { ComposerAttachment } from './composer'

/**
 * `pending` entries are auto-drained; `stuck` entries are dead-lettered — the
 * drainers skip them on this run AND every future one, because the state rides
 * along in localStorage. Only an explicit user Retry (or Delete) clears it.
 */
export type QueuedPromptState = 'pending' | 'stuck'

/** Why an entry was dead-lettered. A code, not prose, so the panel can localize
 *  the headline while `lastError` carries the verbatim (untruncated) detail. */
export type QueuedPromptStuckReason = 'attachment-missing' | 'drain-failed' | 'expired' | 'origin-unresolved'

export interface QueuedPromptEntry {
  id: string
  text: string
  /** What the queue panel and the sent bubble show, when it differs from the
   *  text the agent receives. A queued `/skill` invocation carries the whole
   *  expanded skill body as `text` — the UI shows the invocation instead. */
  displayText?: string
  attachments: ComposerAttachment[]
  queuedAt: number
  /**
   * Auto-drain attempts already spent on this entry. PERSISTED on purpose: an
   * in-memory ledger is reset by every app launch, so a permanently-failing
   * entry replays its whole retry budget forever — four duplicate sends, four
   * failures and a toast on every single restart. Surviving the restart is the
   * entire point; do not move this back into a ref.
   */
  attempts: number
  state: QueuedPromptState
  stuckReason?: QueuedPromptStuckReason
  /** Verbatim failure detail (e.g. the full attachment path, or the gateway's
   *  message). Shown untruncated in the queue panel. */
  lastError?: string
}

type QueueState = Record<string, QueuedPromptEntry[]>

const STORAGE_KEY = 'hermes.desktop.composerQueue.v1'

/** Auto-drain attempts for one entry before it is dead-lettered. Declared here
 *  (not at the bottom of the file) because module-init `load()` reads it. */
export const MAX_AUTO_DRAIN_ATTEMPTS = 4

/** A queued turn nobody has sent in a month is a fossil, not a pending intent.
 *  Load-time migration dead-letters those so they never auto-send against a
 *  conversation the user stopped thinking about in a previous season. */
export const QUEUE_ENTRY_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)

const normalizeEntry = (raw: unknown, now: number): null | QueuedPromptEntry => {
  if (!isRecord(raw) || typeof raw.id !== 'string' || !raw.id) {
    return null
  }

  const attachments = Array.isArray(raw.attachments)
    ? (raw.attachments.filter(isRecord) as unknown as ComposerAttachment[])
    : []

  const queuedAt = typeof raw.queuedAt === 'number' && Number.isFinite(raw.queuedAt) ? raw.queuedAt : now
  const attempts = typeof raw.attempts === 'number' && raw.attempts > 0 ? Math.floor(raw.attempts) : 0
  const expired = now - queuedAt > QUEUE_ENTRY_MAX_AGE_MS
  const stuck = raw.state === 'stuck' || expired

  return {
    id: raw.id,
    text: typeof raw.text === 'string' ? raw.text : '',
    // Carried, not recomputed: the display projection (a `/skill` invocation
    // standing in for the expanded body) is only known at enqueue time, so
    // dropping it here would silently rewrite restored entries to show the
    // whole skill body — and `load()` persists this result, making it stick.
    ...(typeof raw.displayText === 'string' && raw.displayText ? { displayText: raw.displayText } : {}),
    attachments,
    queuedAt,
    attempts: stuck ? Math.max(attempts, MAX_AUTO_DRAIN_ATTEMPTS) : attempts,
    state: stuck ? 'stuck' : 'pending',
    ...(stuck
      ? {
          stuckReason:
            raw.state === 'stuck' && typeof raw.stuckReason === 'string'
              ? (raw.stuckReason as QueuedPromptStuckReason)
              : ('expired' as const)
        }
      : {}),
    ...(typeof raw.lastError === 'string' && raw.lastError ? { lastError: raw.lastError } : {})
  }
}

/**
 * Parse + migrate persisted queue state. Pure so the load-time contract is
 * testable without a fake `window`: unknown/legacy rows get the dead-letter
 * fields they were written without, and anything older than
 * {@link QUEUE_ENTRY_MAX_AGE_MS} is dead-lettered on the spot rather than
 * auto-sent. Migration never DELETES an entry — the user deletes from the
 * panel, so a fossil is always still there to read (or retry) if they meant it.
 */
export const normalizeLoadedQueueState = (raw: unknown, now: number = Date.now()): QueueState => {
  if (!isRecord(raw)) {
    return {}
  }

  const next: QueueState = {}

  for (const [sid, entries] of Object.entries(raw)) {
    if (!sid.trim() || !Array.isArray(entries)) {
      continue
    }

    const normalized = entries
      .map(entry => normalizeEntry(entry, now))
      .filter((e): e is QueuedPromptEntry => Boolean(e))

    if (normalized.length > 0) {
      next[sid] = normalized
    }
  }

  return next
}

const load = (): QueueState => {
  if (typeof window === 'undefined') {
    return {}
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)

    if (!raw) {
      return {}
    }

    const normalized = normalizeLoadedQueueState(JSON.parse(raw))

    // Write the migration back. Without this the dead-letter decision is only
    // ever in memory, silently recomputed on each boot — so it never shows up
    // in the persisted record, and a user who hits Retry on a fossil would see
    // it quietly re-stick on the next launch. Persisting makes the decision
    // durable, inspectable, and reversible by the same Retry that cleared it.
    save(normalized)

    return normalized
  } catch {
    return {}
  }
}

const save = (state: QueueState) => {
  if (typeof window === 'undefined') {
    return
  }

  try {
    if (Object.keys(state).length === 0) {
      window.localStorage.removeItem(STORAGE_KEY)
    } else {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
    }
  } catch {
    // best-effort: storage may be unavailable, queue still works in-memory
  }
}

export const $queuedPromptsBySession = atom<QueueState>(load())

/**
 * Sessions whose queue the user explicitly halted (Stop button / Esc). A parked
 * queue is skipped by both auto-drain paths until the user acts on it again —
 * resume, send-now, a manual drain, queueing a fresh prompt, or emptying the
 * queue all unpark. Deliberately in-memory only: a fresh app process starts
 * unparked, so restored-entry semantics stay a separate concern.
 */
export const $parkedQueueSessions = atom<Record<string, true>>({})

const setParked = (sid: string, parked: boolean) => {
  const current = $parkedQueueSessions.get()

  if (Boolean(current[sid]) === parked) {
    return
  }

  const next = { ...current }

  if (parked) {
    next[sid] = true
  } else {
    delete next[sid]
  }

  $parkedQueueSessions.set(next)
}

const writeSession = (sid: string, queue: QueuedPromptEntry[]) => {
  const current = $queuedPromptsBySession.get()
  const next = { ...current }

  if (queue.length === 0) {
    delete next[sid]
    // An empty queue has nothing to hold back — drop the park so it can't
    // linger as stale state and silently gate entries queued much later.
    setParked(sid, false)
  } else {
    next[sid] = queue
  }

  $queuedPromptsBySession.set(next)
  save(next)
}

const sidOf = (key: string | null | undefined): null | string => {
  const trimmed = key?.trim()

  return trimmed ? trimmed : null
}

const queueFor = (sid: string) => $queuedPromptsBySession.get()[sid] ?? []

const nextId = () => `queued-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

const cloneAttachments = (attachments: ComposerAttachment[]) => attachments.map(a => ({ ...a }))

export const getQueuedPrompts = (key: string | null | undefined): QueuedPromptEntry[] => {
  const sid = sidOf(key)

  return sid ? queueFor(sid) : []
}

export const enqueueQueuedPrompt = (
  key: string | null | undefined,
  payload: { text: string; attachments: ComposerAttachment[]; displayText?: string }
): null | QueuedPromptEntry => {
  const sid = sidOf(key)

  if (!sid) {
    return null
  }

  const entry: QueuedPromptEntry = {
    id: nextId(),
    text: payload.text,
    ...(payload.displayText ? { displayText: payload.displayText } : {}),
    attachments: cloneAttachments(payload.attachments),
    queuedAt: Date.now(),
    attempts: 0,
    state: 'pending'
  }

  writeSession(sid, [...queueFor(sid), entry])
  // Queueing a new prompt is fresh intent to keep the conversation moving —
  // a park from an earlier Stop must not hold this (or the entries ahead of
  // it) back.
  setParked(sid, false)

  return entry
}

/** Rewrite one entry in place. Returns the updated entry, or null when the
 *  session/entry is gone (a concurrent delete or a session switch). */
const mutateEntry = (
  key: string | null | undefined,
  id: string,
  mutate: (entry: QueuedPromptEntry) => QueuedPromptEntry
): null | QueuedPromptEntry => {
  const sid = sidOf(key)

  if (!sid) {
    return null
  }

  const queue = queueFor(sid)
  const index = queue.findIndex(e => e.id === id)

  if (index < 0) {
    return null
  }

  const next = mutate(queue[index]!)
  writeSession(sid, [...queue.slice(0, index), next, ...queue.slice(index + 1)])

  return next
}

export const isQueuedPromptStuck = (entry: QueuedPromptEntry): boolean => entry.state === 'stuck'

/** The next entry a drainer may send: the first one that is neither
 *  dead-lettered nor being edited. A stuck entry does NOT block the ones
 *  behind it — the queue keeps flowing while the user decides what to do
 *  with the poison one. */
export const nextDrainableQueuedPrompt = (
  entries: readonly QueuedPromptEntry[],
  skipId?: null | string
): undefined | QueuedPromptEntry => entries.find(entry => !isQueuedPromptStuck(entry) && entry.id !== skipId)

/** How many entries an auto-drain could still send. A queue holding nothing but
 *  dead-lettered entries is IDLE, not pending — counting them would re-arm the
 *  drain effect forever against work that will never move. */
export const drainableQueuedPromptCount = (entries: readonly QueuedPromptEntry[]): number =>
  entries.reduce((count, entry) => (isQueuedPromptStuck(entry) ? count : count + 1), 0)

export interface DrainFailureOutcome {
  attempts: number
  /** True only on the pending → stuck edge, so the caller can toast ONCE per
   *  entry instead of once per launch. */
  becameStuck: boolean
}

/**
 * Book a failed auto-drain attempt against the entry itself. At
 * {@link MAX_AUTO_DRAIN_ATTEMPTS} the entry is dead-lettered permanently.
 */
export const recordDrainFailure = (
  key: string | null | undefined,
  id: string,
  detail?: string
): DrainFailureOutcome | null => {
  const before = mutateEntry(key, id, entry => entry)

  if (!before || isQueuedPromptStuck(before)) {
    return null
  }

  const attempts = before.attempts + 1
  const stuck = attempts >= MAX_AUTO_DRAIN_ATTEMPTS

  mutateEntry(key, id, entry => ({
    ...entry,
    attempts,
    state: stuck ? 'stuck' : 'pending',
    ...(stuck ? { stuckReason: 'drain-failed' as const } : {}),
    ...(detail ? { lastError: detail } : {})
  }))

  return { attempts, becameStuck: stuck }
}

/**
 * Dead-letter an entry immediately, without spending the retry budget — for
 * failures that are known to be permanent (a referenced attachment is gone,
 * the origin conversation can't be resolved). Returns true on the pending →
 * stuck edge only.
 */
export const markQueuedPromptStuck = (
  key: string | null | undefined,
  id: string,
  reason: QueuedPromptStuckReason,
  detail?: string
): boolean => {
  const before = mutateEntry(key, id, entry => entry)

  if (!before || isQueuedPromptStuck(before)) {
    return false
  }

  return Boolean(
    mutateEntry(key, id, entry => ({
      ...entry,
      attempts: Math.max(entry.attempts, MAX_AUTO_DRAIN_ATTEMPTS),
      state: 'stuck',
      stuckReason: reason,
      ...(detail ? { lastError: detail } : {})
    }))
  )
}

/**
 * User-driven Retry: clear the dead-letter so auto-drain picks the entry up
 * again with a full, fresh attempt budget.
 *
 * `queuedAt` restarts too. Retry means "I still want this sent, now" — leaving
 * the original timestamp would let the load-time age rule silently re-stick a
 * fossil the user just explicitly revived, on the very next launch.
 */
export const retryQueuedPrompt = (key: string | null | undefined, id: string): boolean =>
  Boolean(
    mutateEntry(key, id, entry => {
      const { lastError: _lastError, stuckReason: _stuckReason, ...rest } = entry

      return { ...rest, attempts: 0, queuedAt: Date.now(), state: 'pending' }
    })
  )

export const dequeueQueuedPrompt = (key: string | null | undefined): null | QueuedPromptEntry => {
  const sid = sidOf(key)

  if (!sid) {
    return null
  }

  const [head, ...rest] = queueFor(sid)

  if (!head) {
    return null
  }

  writeSession(sid, rest)

  return head
}

export const removeQueuedPrompt = (key: string | null | undefined, id: string): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  const queue = queueFor(sid)
  const next = queue.filter(e => e.id !== id)

  if (next.length === queue.length) {
    return false
  }

  writeSession(sid, next)

  return true
}

export const promoteQueuedPrompt = (key: string | null | undefined, id: string): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  const queue = queueFor(sid)
  const index = queue.findIndex(e => e.id === id)

  if (index <= 0) {
    return false
  }

  const entry = queue[index]!
  writeSession(sid, [entry, ...queue.slice(0, index), ...queue.slice(index + 1)])

  return true
}

export const updateQueuedPrompt = (
  key: string | null | undefined,
  id: string,
  update: { text: string; attachments?: ComposerAttachment[] }
): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  const queue = queueFor(sid)
  let changed = false

  const next: QueuedPromptEntry[] = queue.map(entry => {
    if (entry.id !== id) {
      return entry
    }

    const attachments = update.attachments ? cloneAttachments(update.attachments) : entry.attachments

    if (entry.text === update.text && !update.attachments) {
      return entry
    }

    changed = true

    // Editing an entry IS the user fixing it (new text, re-picked attachments),
    // so it leaves the dead-letter box with a fresh budget — same reasoning as
    // the manual send-now path clearing the auto-drain backoff.
    //
    // The display projection goes too: a `/skill` invocation standing in for
    // the expanded body no longer describes text the user just rewrote — what
    // they typed is now what sends.
    const { displayText: _displayText, lastError: _lastError, stuckReason: _stuckReason, ...rest } = entry

    return { ...rest, text: update.text, attachments, attempts: 0, state: 'pending' }
  })

  if (!changed) {
    return false
  }

  writeSession(sid, next)

  return true
}

export const updateQueuedPromptText = (key: string | null | undefined, id: string, text: string): boolean =>
  updateQueuedPrompt(key, id, { text })

export const clearQueuedPrompts = (key: string | null | undefined) => {
  const sid = sidOf(key)

  if (!sid || !(sid in $queuedPromptsBySession.get())) {
    return
  }

  writeSession(sid, [])
}

/**
 * Move pending entries from a dead session key onto a live one, preserving FIFO
 * (existing target entries first, migrated entries appended). A backend bounce /
 * resume can mint a fresh runtime session id for the *same* conversation; the
 * entries enqueued under the old id would otherwise be stranded under a key
 * nothing reads anymore. No-op unless both keys resolve and differ.
 */
export const migrateQueuedPrompts = (fromKey: string | null | undefined, toKey: string | null | undefined): boolean => {
  const from = sidOf(fromKey)
  const to = sidOf(toKey)

  if (!from || !to || from === to) {
    return false
  }

  const pending = queueFor(from)

  if (pending.length === 0) {
    return false
  }

  const next = { ...$queuedPromptsBySession.get() }
  delete next[from]
  next[to] = [...queueFor(to), ...pending]

  $queuedPromptsBySession.set(next)
  save(next)

  // The park is a property of the entries the user halted — it re-homes with
  // them. Without this, a backend bounce right after Stop would shed the park
  // and auto-send the exact prompts the user just held back.
  if ($parkedQueueSessions.get()[from]) {
    setParked(from, false)
    setParked(to, true)
  }

  return true
}

/**
 * Park a session's queue after an explicit user halt (Stop / Esc): entries stay
 * visible in the panel but neither auto-drain path sends them. No-op for a
 * session with nothing queued — parking exists to hold back queued turns, and
 * a park with no queue would only linger as a stale gate.
 */
export const parkQueuedPrompts = (key: string | null | undefined): boolean => {
  const sid = sidOf(key)

  if (!sid || queueFor(sid).length === 0) {
    return false
  }

  setParked(sid, true)

  return true
}

/** Lift a park (user resumed the queue). Safe to call for any session. */
export const unparkQueuedPrompts = (key: string | null | undefined): void => {
  const sid = sidOf(key)

  if (sid) {
    setParked(sid, false)
  }
}

export const isQueueParked = (key: string | null | undefined): boolean => {
  const sid = sidOf(key)

  return sid ? Boolean($parkedQueueSessions.get()[sid]) : false
}

/** Inputs to {@link shouldAutoDrain}. */
export interface AutoDrainInput {
  isBusy: boolean
  /** The user explicitly halted this session's queue (Stop / Esc). */
  parked?: boolean
  queueLength: number
}

/**
 * Decide whether the composer should auto-drain the next queued prompt.
 *
 * Edge-independent on purpose: the queue must advance whenever the session is
 * idle and has pending entries, NOT only on an observed busy true → false edge.
 * A backend bounce / websocket reconnect remounts the composer and resets the
 * busy ref to the current value, swallowing the settle edge — an edge-gated
 * drain would then strand the entry forever. The caller's drain lock
 * (`drainingQueueRef`) serializes sends so being edge-free can't double-submit.
 *
 * `parked` is the one deliberate exception: an explicit Stop/Esc is the user
 * saying HALT, and immediately firing the next queued prompt contradicts the
 * instruction they just gave. Parked entries stay in the panel until the user
 * resumes, sends, edits, or deletes them. Interrupts that exist to reach the
 * queue faster (send-now-while-busy) never park, so they keep draining through
 * this same gate.
 */
export const shouldAutoDrain = ({ isBusy, parked, queueLength }: AutoDrainInput): boolean =>
  !isBusy && !parked && queueLength > 0

// MAX_AUTO_DRAIN_ATTEMPTS is declared near the top of this file, not here:
// module-init `load()` reads it while migrating persisted entries, so a
// bottom-of-file const would be in its temporal dead zone.

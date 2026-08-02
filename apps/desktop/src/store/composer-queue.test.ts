import { beforeEach, describe, expect, it } from 'vitest'

import type { ComposerAttachment } from './composer'
import {
  $queuedPromptsBySession,
  clearQueuedPrompts,
  dequeueQueuedPrompt,
  drainableQueuedPromptCount,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  isQueuedPromptStuck,
  markQueuedPromptStuck,
  MAX_AUTO_DRAIN_ATTEMPTS,
  migrateQueuedPrompts,
  nextDrainableQueuedPrompt,
  normalizeLoadedQueueState,
  promoteQueuedPrompt,
  QUEUE_ENTRY_MAX_AGE_MS,
  recordDrainFailure,
  removeQueuedPrompt,
  retryQueuedPrompt,
  shouldAutoDrain,
  updateQueuedPrompt,
  updateQueuedPromptText
} from './composer-queue'

const SESSION_KEY = 'session-abc'
const QUEUE_STORAGE_KEY = 'hermes.desktop.composerQueue.v1'

function attachment(id: string, kind: ComposerAttachment['kind'] = 'file'): ComposerAttachment {
  return {
    id,
    kind,
    label: id,
    refText: `@file:${id}`
  }
}

describe('composer queue store', () => {
  beforeEach(() => {
    window.localStorage.removeItem(QUEUE_STORAGE_KEY)
    $queuedPromptsBySession.set({})
  })

  it('queues prompts in FIFO order', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first' })
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'second' })

    expect(dequeueQueuedPrompt(SESSION_KEY)?.text).toBe('first')
    expect(dequeueQueuedPrompt(SESSION_KEY)?.text).toBe('second')
    expect(dequeueQueuedPrompt(SESSION_KEY)).toBeNull()
  })

  it('clones attachments when queueing', () => {
    const source = [attachment('a-1')]
    const queued = enqueueQueuedPrompt(SESSION_KEY, { attachments: source, text: 'check clones' })

    expect(queued).not.toBeNull()
    expect(getQueuedPrompts(SESSION_KEY)[0]?.attachments[0]).toEqual(source[0])
    expect(getQueuedPrompts(SESSION_KEY)[0]?.attachments[0]).not.toBe(source[0])
  })

  it('updates and removes queued entries by id', () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'draft one' })
    const second = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'draft two' })

    expect(first).not.toBeNull()
    expect(second).not.toBeNull()

    expect(updateQueuedPromptText(SESSION_KEY, first!.id, 'draft one edited')).toBe(true)
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.text)).toEqual(['draft one edited', 'draft two'])

    expect(removeQueuedPrompt(SESSION_KEY, first!.id)).toBe(true)
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.text)).toEqual(['draft two'])
  })

  it('promotes a queued entry to the front', () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first' })
    const second = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'second' })
    const third = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'third' })

    expect(first).not.toBeNull()
    expect(second).not.toBeNull()
    expect(third).not.toBeNull()

    expect(promoteQueuedPrompt(SESSION_KEY, third!.id)).toBe(true)
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.text)).toEqual(['third', 'first', 'second'])
    expect(promoteQueuedPrompt(SESSION_KEY, third!.id)).toBe(false)
  })

  it('updates queued text and attachment snapshot', () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [attachment('f-1')], text: 'draft one' })
    const editedAttachments = [attachment('f-2'), attachment('f-3', 'image')]

    expect(first).not.toBeNull()
    expect(
      updateQueuedPrompt(SESSION_KEY, first!.id, {
        attachments: editedAttachments,
        text: 'edited text'
      })
    ).toBe(true)

    const queue = getQueuedPrompts(SESSION_KEY)
    expect(queue[0]?.text).toBe('edited text')
    expect(queue[0]?.attachments).toEqual(editedAttachments)
    expect(queue[0]?.attachments[0]).not.toBe(editedAttachments[0])
  })

  it('clears queue state for a session', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [attachment('img-1', 'image')], text: 'queued' })

    clearQueuedPrompts(SESSION_KEY)

    expect(getQueuedPrompts(SESSION_KEY)).toEqual([])
    expect($queuedPromptsBySession.get()[SESSION_KEY]).toBeUndefined()
    expect(window.localStorage.getItem(QUEUE_STORAGE_KEY)).toBeNull()
  })

  it('persists queue entries into local storage', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'persist me' })

    const raw = window.localStorage.getItem(QUEUE_STORAGE_KEY)
    expect(raw).toBeTruthy()

    const parsed = JSON.parse(String(raw)) as Record<string, { text: string }[]>
    expect(parsed[SESSION_KEY]?.[0]?.text).toBe('persist me')
  })
})

describe('migrateQueuedPrompts', () => {
  beforeEach(() => {
    window.localStorage.removeItem(QUEUE_STORAGE_KEY)
    $queuedPromptsBySession.set({})
  })

  it('moves entries from a dead runtime key onto the live one', () => {
    enqueueQueuedPrompt('rt-old', { attachments: [], text: 'stranded' })

    expect(migrateQueuedPrompts('rt-old', 'rt-new')).toBe(true)
    expect(getQueuedPrompts('rt-old')).toEqual([])
    expect(getQueuedPrompts('rt-new').map(e => e.text)).toEqual(['stranded'])
    // The dead key is dropped from the store entirely.
    expect($queuedPromptsBySession.get()['rt-old']).toBeUndefined()
  })

  it('appends after existing target entries (FIFO preserved)', () => {
    enqueueQueuedPrompt('rt-new', { attachments: [], text: 'already here' })
    enqueueQueuedPrompt('rt-old', { attachments: [], text: 'migrated' })

    migrateQueuedPrompts('rt-old', 'rt-new')

    expect(getQueuedPrompts('rt-new').map(e => e.text)).toEqual(['already here', 'migrated'])
  })

  it('is a no-op when source is empty or keys match', () => {
    expect(migrateQueuedPrompts('rt-old', 'rt-new')).toBe(false)
    expect(migrateQueuedPrompts('rt-x', 'rt-x')).toBe(false)
  })
})

/**
 * Simulate an app restart: everything in memory is thrown away and the store is
 * rebuilt from what actually survived in localStorage. This is the check that a
 * naive in-memory retry ledger cannot pass.
 */
function restartFromStorage(now?: number) {
  const raw = window.localStorage.getItem(QUEUE_STORAGE_KEY)
  $queuedPromptsBySession.set(normalizeLoadedQueueState(raw ? JSON.parse(raw) : null, now))
}

describe('queue dead-lettering', () => {
  beforeEach(() => {
    window.localStorage.removeItem(QUEUE_STORAGE_KEY)
    $queuedPromptsBySession.set({})
  })

  it('dead-letters an entry once it exhausts the auto-drain budget', () => {
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'poison' })!

    for (let attempt = 1; attempt < MAX_AUTO_DRAIN_ATTEMPTS; attempt += 1) {
      const outcome = recordDrainFailure(SESSION_KEY, entry.id, 'boom')
      expect(outcome).toEqual({ attempts: attempt, becameStuck: false })
      expect(isQueuedPromptStuck(getQueuedPrompts(SESSION_KEY)[0]!)).toBe(false)
    }

    expect(recordDrainFailure(SESSION_KEY, entry.id, 'boom')).toEqual({
      attempts: MAX_AUTO_DRAIN_ATTEMPTS,
      becameStuck: true
    })
    expect(isQueuedPromptStuck(getQueuedPrompts(SESSION_KEY)[0]!)).toBe(true)
  })

  it('reports the stuck transition exactly once, so the toast cannot fire per launch', () => {
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'poison' })!

    const transitions = Array.from({ length: MAX_AUTO_DRAIN_ATTEMPTS + 3 }, () =>
      Boolean(recordDrainFailure(SESSION_KEY, entry.id)?.becameStuck)
    )

    expect(transitions.filter(Boolean)).toHaveLength(1)
    // Past the budget the entry is inert: no further bookkeeping at all.
    expect(recordDrainFailure(SESSION_KEY, entry.id)).toBeNull()
  })

  it('KEEPS the entry dead-lettered across a restart', () => {
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'poison' })!

    for (let attempt = 0; attempt < MAX_AUTO_DRAIN_ATTEMPTS; attempt += 1) {
      recordDrainFailure(SESSION_KEY, entry.id, 'image not found: /Users/me/Library/Application Support/x.png')
    }

    restartFromStorage()

    // The whole bug: an in-memory ledger hands the entry a fresh budget on
    // every launch, so it replays its failure forever. Attempts and the stuck
    // flag must both survive the trip through localStorage.
    const revived = getQueuedPrompts(SESSION_KEY)[0]!
    expect(isQueuedPromptStuck(revived)).toBe(true)
    expect(revived.attempts).toBeGreaterThanOrEqual(MAX_AUTO_DRAIN_ATTEMPTS)
    expect(nextDrainableQueuedPrompt(getQueuedPrompts(SESSION_KEY))).toBeUndefined()
    expect(recordDrainFailure(SESSION_KEY, revived.id)).toBeNull()
    // The untruncated failure detail rides along for the panel to show.
    expect(revived.lastError).toContain('/Users/me/Library/Application Support/x.png')
  })

  it('marks a known-permanent failure stuck immediately, without spending the budget', () => {
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'has a dead attachment' })!
    const missing = '/Users/me/Library/Application Support/Hermes/images/clip_1.png'

    expect(markQueuedPromptStuck(SESSION_KEY, entry.id, 'attachment-missing', missing)).toBe(true)
    // Second call is not a transition — the caller must not toast twice.
    expect(markQueuedPromptStuck(SESSION_KEY, entry.id, 'attachment-missing', missing)).toBe(false)

    restartFromStorage()

    const revived = getQueuedPrompts(SESSION_KEY)[0]!
    expect(revived.state).toBe('stuck')
    expect(revived.stuckReason).toBe('attachment-missing')
    expect(revived.lastError).toBe(missing)
  })

  it('restores a stuck entry to pending on retry, with a fresh budget', () => {
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'retry me' })!
    markQueuedPromptStuck(SESSION_KEY, entry.id, 'attachment-missing', '/gone.png')

    expect(retryQueuedPrompt(SESSION_KEY, entry.id)).toBe(true)

    restartFromStorage()

    const revived = getQueuedPrompts(SESSION_KEY)[0]!
    expect(revived.state).toBe('pending')
    expect(revived.attempts).toBe(0)
    expect(revived.lastError).toBeUndefined()
    expect(revived.stuckReason).toBeUndefined()
    expect(nextDrainableQueuedPrompt(getQueuedPrompts(SESSION_KEY))?.id).toBe(entry.id)
  })

  it('editing an entry takes it out of the dead-letter box', () => {
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'bad' })!
    markQueuedPromptStuck(SESSION_KEY, entry.id, 'attachment-missing', '/gone.png')

    updateQueuedPrompt(SESSION_KEY, entry.id, { attachments: [attachment('fresh')], text: 'fixed' })

    const edited = getQueuedPrompts(SESSION_KEY)[0]!
    expect(edited.state).toBe('pending')
    expect(edited.attempts).toBe(0)
    expect(edited.lastError).toBeUndefined()
  })

  it('lets a stuck entry be skipped without blocking the ones behind it', () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'poison' })!
    const second = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'still good' })!

    markQueuedPromptStuck(SESSION_KEY, first.id, 'drain-failed')

    const entries = getQueuedPrompts(SESSION_KEY)
    expect(nextDrainableQueuedPrompt(entries)?.id).toBe(second.id)
    expect(drainableQueuedPromptCount(entries)).toBe(1)
    // An entry being edited is skipped too, which can leave nothing drainable.
    expect(nextDrainableQueuedPrompt(entries, second.id)).toBeUndefined()
  })

  it('treats an all-stuck queue as having no pending work', () => {
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'poison' })!
    markQueuedPromptStuck(SESSION_KEY, entry.id, 'drain-failed')

    const entries = getQueuedPrompts(SESSION_KEY)
    expect(entries).toHaveLength(1)
    expect(shouldAutoDrain({ isBusy: false, queueLength: drainableQueuedPromptCount(entries) })).toBe(false)
  })
})

describe('normalizeLoadedQueueState', () => {
  const now = 1_800_000_000_000

  it('dead-letters fossils older than the max age instead of auto-sending them', () => {
    const june = now - QUEUE_ENTRY_MAX_AGE_MS - 1

    const state = normalizeLoadedQueueState(
      { 'stored-june': [{ id: 'q1', text: 'what about the season posters?', attachments: [], queuedAt: june }] },
      now
    )

    const entry = state['stored-june']![0]!
    expect(entry.state).toBe('stuck')
    expect(entry.stuckReason).toBe('expired')
    // Never deleted — the user decides from the panel.
    expect(entry.text).toBe('what about the season posters?')
  })

  it('leaves a recently queued entry pending', () => {
    const state = normalizeLoadedQueueState(
      { sid: [{ id: 'q1', text: 'yesterday', attachments: [], queuedAt: now - 60_000 }] },
      now
    )

    expect(state.sid![0]!.state).toBe('pending')
    expect(state.sid![0]!.attempts).toBe(0)
  })

  it('backfills the dead-letter fields onto pre-existing entries written without them', () => {
    const state = normalizeLoadedQueueState(
      { sid: [{ id: 'q1', text: 'legacy', attachments: [], queuedAt: now }] },
      now
    )

    expect(state.sid![0]).toMatchObject({ attempts: 0, id: 'q1', state: 'pending' })
  })

  it('preserves an already-stuck entry and its reason', () => {
    const state = normalizeLoadedQueueState(
      {
        sid: [
          {
            id: 'q1',
            text: 'dead',
            attachments: [],
            queuedAt: now,
            attempts: 4,
            state: 'stuck',
            stuckReason: 'attachment-missing',
            lastError: '/Users/me/Library/Application Support/x.png'
          }
        ]
      },
      now
    )

    expect(state.sid![0]).toMatchObject({
      lastError: '/Users/me/Library/Application Support/x.png',
      state: 'stuck',
      stuckReason: 'attachment-missing'
    })
  })

  it('drops structurally broken rows rather than throwing on load', () => {
    const state = normalizeLoadedQueueState(
      { sid: [null, 'nope', { text: 'no id' }, { id: 'ok', text: 'fine', attachments: [], queuedAt: now }], '': [] },
      now
    )

    expect(state.sid!.map(e => e.id)).toEqual(['ok'])
    expect(state['']).toBeUndefined()
  })

  it('returns an empty state for junk input', () => {
    expect(normalizeLoadedQueueState(null)).toEqual({})
    expect(normalizeLoadedQueueState([1, 2, 3])).toEqual({})
    expect(normalizeLoadedQueueState('nope')).toEqual({})
  })
})

describe('shouldAutoDrain', () => {
  it('drains whenever idle with a non-empty queue', () => {
    expect(shouldAutoDrain({ isBusy: false, queueLength: 1 })).toBe(true)
  })

  it('drains on mount/reconnect with no observed busy edge', () => {
    // The whole point of dropping the edge: a remount resets the busy ref, so an
    // edge-gated drain would strand the entry. Idle + non-empty must still fire.
    expect(shouldAutoDrain({ isBusy: false, queueLength: 2 })).toBe(true)
  })

  it('does not drain mid-turn', () => {
    expect(shouldAutoDrain({ isBusy: true, queueLength: 1 })).toBe(false)
  })

  it('does not drain an empty queue', () => {
    expect(shouldAutoDrain({ isBusy: false, queueLength: 0 })).toBe(false)
  })
})

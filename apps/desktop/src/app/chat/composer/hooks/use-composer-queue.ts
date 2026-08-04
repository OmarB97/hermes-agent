import { useStore } from '@nanostores/react'
import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { desktopPathProbe, firstMissingAttachmentPath } from '@/lib/queued-attachment-preflight'
import { useSessionSlice } from '@/lib/use-session-slice'
import { type ComposerAttachment } from '@/store/composer'
import { resetBrowseState } from '@/store/composer-input-history'
import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  drainableQueuedPromptCount,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  markQueuedPromptStuck,
  migrateQueuedPrompts,
  nextDrainableQueuedPrompt,
  promoteQueuedPrompt,
  type QueuedPromptEntry,
  recordDrainFailure,
  removeQueuedPrompt,
  retryQueuedPrompt,
  shouldAutoDrain,
  unparkQueuedPrompts,
  updateQueuedPrompt
} from '@/store/composer-queue'
import { notify } from '@/store/notifications'

import { cloneAttachments, type QueueEditState } from '../composer-utils'
import { useComposerScope } from '../scope'
import type { ChatBarProps } from '../types'

interface UseComposerQueueArgs {
  activeQueueSessionKey: string | null
  attachments: ComposerAttachment[]
  busy: boolean
  clearDraft: () => void
  draftRef: RefObject<string>
  focusInput: () => void
  loadIntoComposer: (text: string, attachments: ComposerAttachment[]) => void
  onCancel: ChatBarProps['onCancel']
  onSubmit: ChatBarProps['onSubmit']
  queueEditRef: RefObject<QueueEditState | null>
  queueSessionKey: ChatBarProps['queueSessionKey']
  sessionId: string | null | undefined
}

/**
 * The composer's queue engine — everything about queued turns: the per-session
 * queue store binding, in-place queued-prompt editing (begin/step/exit), the
 * shared drain lock + send-then-remove sequence, manual send-now, and the
 * edge-independent auto-drain with bounded retries. It consumes the draft API
 * (draftRef/clearDraft/loadIntoComposer/focusInput) and writes the
 * coordinator-owned `queueEditRef` so the draft engine can read the edit state
 * without a back-reference. Behaviour-identical to the inline original.
 */
export function useComposerQueue({
  activeQueueSessionKey,
  attachments,
  busy,
  clearDraft,
  draftRef,
  focusInput,
  loadIntoComposer,
  onCancel,
  onSubmit,
  queueEditRef,
  queueSessionKey,
  sessionId
}: UseComposerQueueArgs) {
  const { t } = useI18n()
  const scope = useComposerScope()

  // Per-session slice (edge): re-renders only when THIS session's queue changes,
  // not on cross-session queue churn (the plain atom's map ref changes on every
  // write; the keyed array does not).
  const queuedPrompts = useSessionSlice($queuedPromptsBySession, activeQueueSessionKey)

  // Parked = the user explicitly halted this session (Stop/Esc) while prompts
  // were queued. The map is tiny (only halted sessions) so a plain subscribe
  // is fine; the auto-drain effect below reads it as a gate.
  const parkedSessions = useStore($parkedQueueSessions)
  const queueParked = Boolean(activeQueueSessionKey && parkedSessions[activeQueueSessionKey])

  const [queueEdit, setQueueEdit] = useState<QueueEditState | null>(null)
  queueEditRef.current = queueEdit

  const setQueueEditSnapshot = useCallback(
    (next: QueueEditState | null) => {
      queueEditRef.current = next
      setQueueEdit(next)
    },
    [queueEditRef]
  )

  const editingQueuedPrompt = queueEdit ? (queuedPrompts.find(entry => entry.id === queueEdit.entryId) ?? null) : null

  const prevQueueKeyRef = useRef(activeQueueSessionKey)
  const drainingQueueRef = useRef(false)
  const pathProbe = desktopPathProbe()

  const notifyStuck = useCallback(() => {
    notify({
      id: 'composer-queue-stuck',
      kind: 'error',
      title: t.composer.queueStuckTitle,
      message: t.composer.queueStuckBody
    })
  }, [t])

  const beginQueuedEdit = (entry: QueuedPromptEntry) => {
    if (!activeQueueSessionKey || queueEdit) {
      return
    }

    setQueueEditSnapshot({
      attachments: cloneAttachments(attachments),
      draft: draftRef.current,
      entryId: entry.id,
      sessionKey: activeQueueSessionKey
    })
    // Edit what the panel SHOWS. A queued `/skill` entry's text is the
    // expanded skill body — never drop that into the composer.
    loadIntoComposer(entry.displayText ?? entry.text, entry.attachments)
    triggerHaptic('selection')
    focusInput()
  }

  // Walk queued entries while editing (ArrowUp = older, ArrowDown = newer),
  // saving the in-progress edit on each step. Stepping newer past the last
  // entry exits edit mode and restores the pre-edit draft.
  const stepQueuedEdit = (direction: -1 | 1) => {
    if (!queueEdit) {
      return false
    }

    const index = queuedPrompts.findIndex(e => e.id === queueEdit.entryId)
    const target = index + direction

    if (index < 0 || target < 0) {
      return index >= 0 // at the oldest: swallow; missing entry: let it fall through
    }

    const saved = updateQueuedPrompt(queueEdit.sessionKey, queueEdit.entryId, {
      attachments: cloneAttachments(attachments),
      text: draftRef.current
    })

    const next = queuedPrompts[target]

    if (next) {
      setQueueEditSnapshot({ ...queueEdit, entryId: next.id })
      loadIntoComposer(next.displayText ?? next.text, next.attachments)
    } else {
      setQueueEditSnapshot(null)
      loadIntoComposer(queueEdit.draft, queueEdit.attachments)
    }

    triggerHaptic(saved ? 'success' : 'selection')
    focusInput()

    return true
  }

  const exitQueuedEdit = (action: 'cancel' | 'save'): boolean => {
    if (!queueEdit) {
      return false
    }

    if (action === 'save') {
      const text = draftRef.current
      const next = cloneAttachments(attachments)

      if (!text.trim() && next.length === 0) {
        return false
      }

      const saved = updateQueuedPrompt(queueEdit.sessionKey, queueEdit.entryId, { attachments: next, text })
      triggerHaptic(saved ? 'success' : 'selection')
    } else {
      triggerHaptic('cancel')
    }

    setQueueEditSnapshot(null)
    loadIntoComposer(queueEdit.draft, queueEdit.attachments)
    focusInput()

    return true
  }

  const queueCurrentDraft = useCallback(() => {
    const text = draftRef.current

    if (!activeQueueSessionKey || (!text.trim() && attachments.length === 0)) {
      return false
    }

    if (!enqueueQueuedPrompt(activeQueueSessionKey, { text, attachments })) {
      return false
    }

    clearDraft()
    scope.attachments.clear()
    triggerHaptic('selection')

    return true
  }, [activeQueueSessionKey, attachments, clearDraft, draftRef, scope.attachments])

  // All queue drain paths share one lock + send-then-remove sequence, and one
  // place that books failures against the entry. `pickEntry` lets each caller
  // choose head, by-id, or skip-edited. Resolves false on failure and NEVER
  // rejects — every caller fires it with `void`.
  const runDrain = useCallback(
    async (pickEntry: (entries: QueuedPromptEntry[]) => QueuedPromptEntry | undefined): Promise<boolean> => {
      if (drainingQueueRef.current || !activeQueueSessionKey) {
        return false
      }

      const drainQueueSessionKey = activeQueueSessionKey
      const drainRuntimeSessionId = sessionId ?? null
      const entry = pickEntry(getQueuedPrompts(drainQueueSessionKey))

      if (!entry) {
        return false
      }

      drainingQueueRef.current = true

      // Book the attempt on the PERSISTED entry, so a prompt that can never
      // succeed is dead-lettered for good instead of spending a fresh budget on
      // every app launch. Toast only on the pending → stuck edge.
      const fail = (detail?: string) => {
        if (recordDrainFailure(drainQueueSessionKey, entry.id, detail)?.becameStuck) {
          notifyStuck()
        }

        return false
      }

      try {
        // Preflight before the send: a queued turn references its attachments
        // by local path, and a file that has gone away can never be staged.
        // Dead-letter it here (with the full path) rather than letting the
        // submit pipeline paint an optimistic bubble it then has to roll back.
        const missingPath = await firstMissingAttachmentPath(entry.attachments, pathProbe)

        if (missingPath) {
          if (markQueuedPromptStuck(drainQueueSessionKey, entry.id, 'attachment-missing', missingPath)) {
            notifyStuck()
          }

          return false
        }

        const accepted = await Promise.resolve(
          onSubmit(entry.text, {
            attachments: entry.attachments,
            ...(entry.displayText ? { displayText: entry.displayText } : {}),
            fromQueue: true,
            sessionId: drainRuntimeSessionId,
            storedSessionId: drainQueueSessionKey
          })
        )

        if (accepted === false) {
          // A soft "not now" (session busy, context drifted) — no reason to
          // record beyond the attempt itself.
          return fail()
        }

        removeQueuedPrompt(drainQueueSessionKey, entry.id)
        resetBrowseState(drainRuntimeSessionId)
        // A successful drain means the queue is flowing again — lift any park
        // so the remaining entries follow. Manual drains (Enter on an empty
        // composer, the per-row send arrow) are exactly the resume gestures a
        // parked queue waits for; the auto path only reaches here unparked.
        unparkQueuedPrompts(drainQueueSessionKey)

        return true
      } catch (err) {
        // A hard failure carries its reason (e.g. the gateway's own message)
        // onto the entry, so the queue panel can show it verbatim.
        return fail((err instanceof Error ? err.message.trim() : '') || undefined)
      } finally {
        drainingQueueRef.current = false
      }
    },
    [activeQueueSessionKey, notifyStuck, onSubmit, pathProbe, sessionId]
  )

  const pickDrainHead = useCallback(
    (entries: QueuedPromptEntry[]) => nextDrainableQueuedPrompt(entries, queueEditRef.current?.entryId),
    [queueEditRef] // reads the edit id off a ref so the lock-holder always sees the latest
  )

  const drainNextQueued = useCallback(() => runDrain(pickDrainHead), [pickDrainHead, runDrain])

  const sendQueuedNow = useCallback(
    (id: string) => {
      if (!activeQueueSessionKey || id === queueEdit?.entryId) {
        return false
      }

      // Tapping send/retry clears the dead-letter first, so the entry gets a
      // fresh attempt budget whichever branch below runs — otherwise a stuck
      // entry promoted while busy would be skipped again by the auto-drain.
      retryQueuedPrompt(activeQueueSessionKey, id)

      if (busy) {
        // Promote to the head, then interrupt. The gateway always emits a
        // settle (message.complete + session.info running:false) when the
        // turn unwinds, and the busy→false auto-drain below sends this entry.
        // Unpark first: this interrupt exists to REACH the queue, so the
        // settle drain must flow — unlike a Stop/Esc halt, which parks.
        promoteQueuedPrompt(activeQueueSessionKey, id)
        unparkQueuedPrompts(activeQueueSessionKey)
        triggerHaptic('selection')
        void Promise.resolve(onCancel())

        return true
      }

      return runDrain(entries => entries.find(e => e.id === id))
    },
    [activeQueueSessionKey, busy, onCancel, queueEdit, runDrain]
  )

  // Edge-independent auto-drain: send the head whenever the session is idle and
  // the queue is non-empty, bounding retries so a thrown/rejected onSubmit (e.g.
  // a stale-session 404) can't strand the entry permanently nor spin-loop. The
  // drain lock serializes sends; a remount/reconnect resets the failure counts.
  const autoDrainNext = useCallback(() => {
    if (busy || queueParked || drainingQueueRef.current || !activeQueueSessionKey) {
      return
    }

    const entry = pickDrainHead(queuedPrompts)

    if (!entry) {
      return
    }

    // runDrain owns the retry bookkeeping (and never rejects), so there is
    // nothing left to do here but fire it.
    void runDrain(() => entry)
  }, [activeQueueSessionKey, busy, pickDrainHead, queueParked, queuedPrompts, runDrain])

  // Re-key on a runtime session-id change. A stable stored id (queueSessionKey)
  // never churns, so a change there is a real session switch and must NOT
  // migrate; only the runtime-derived key (queueSessionKey falsy → key is
  // sessionId) churns on a backend bounce/resume of the same conversation.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const prev = prevQueueKeyRef.current
    prevQueueKeyRef.current = activeQueueSessionKey

    if (queueSessionKey || !prev || !activeQueueSessionKey || prev === activeQueueSessionKey) {
      return
    }

    migrateQueuedPrompts(prev, activeQueueSessionKey)
  }, [activeQueueSessionKey, queueSessionKey])

  // Queued turns flow whenever the session is idle — on the busy→false settle
  // edge, on mount/reconnect, and after a re-key — so a swallowed edge can't
  // strand them. A park (explicit Stop/Esc) is the one gate: those entries wait
  // for the user. Dead-lettered entries are the other: they are not drainable,
  // so a queue holding only those is idle and must not re-arm this effect.
  // To cancel queued turns, the user deletes them from the panel.
  const drainableCount = drainableQueuedPromptCount(queuedPrompts)

  useEffect(() => {
    if (shouldAutoDrain({ isBusy: busy, parked: queueParked, queueLength: drainableCount })) {
      autoDrainNext()
    }
  }, [autoDrainNext, busy, drainableCount, queueParked])

  // Queue-edit cleanup: on session swap the scope effect already stashed the
  // edit snapshot; only restore into the composer when still on the same scope.
  useEffect(() => {
    if (!queueEdit) {
      return
    }

    if (queueEdit.sessionKey === activeQueueSessionKey) {
      if (editingQueuedPrompt) {
        return
      }

      setQueueEditSnapshot(null)
      loadIntoComposer(queueEdit.draft, queueEdit.attachments)

      return
    }

    setQueueEditSnapshot(null)
  }, [activeQueueSessionKey, editingQueuedPrompt, queueEdit, setQueueEditSnapshot]) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    beginQueuedEdit,
    drainNextQueued,
    editingQueuedPrompt,
    exitQueuedEdit,
    queueCurrentDraft,
    queueEdit,
    queueParked,
    queuedPrompts,
    sendQueuedNow,
    stepQueuedEdit
  }
}

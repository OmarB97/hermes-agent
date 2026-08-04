import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { desktopPathProbe, firstMissingAttachmentPath, type PathProbe } from '@/lib/queued-attachment-preflight'
import { resetBrowseState } from '@/store/composer-input-history'
import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  drainableQueuedPromptCount,
  getQueuedPrompts,
  markQueuedPromptStuck,
  nextDrainableQueuedPrompt,
  type QueuedPromptEntry,
  recordDrainFailure,
  removeQueuedPrompt,
  shouldAutoDrain
} from '@/store/composer-queue'
import { notify } from '@/store/notifications'
import { $sessions, idsShareLineage } from '@/store/session'
import { $workingSessionIds } from '@/store/session-states'

import type { SubmitTextOptions } from './use-prompt-actions/utils'

type SubmitQueuedPrompt = (text: string, options?: SubmitTextOptions) => Promise<boolean> | boolean

interface BackgroundQueueDrainOptions {
  enabled: boolean
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  selectedStoredSessionId: string | null
  submitText: SubmitQueuedPrompt
  /** Local file-existence probe for the attachment preflight. Defaults to the
   *  desktop bridge; injectable for tests. */
  pathProbe?: PathProbe
}

const BACKGROUND_DRAIN_RETRY_MS = 750

const errorDetail = (error: unknown): string => {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return message.trim()
}

/**
 * Drain queued prompts for sessions that are not currently rendered by ChatBar.
 *
 * The visible ChatBar owns the interactive queue panel for the selected session.
 * Without this background drain, a prompt queued in Session A can sit forever
 * after the user switches to Session B: the only auto-drain effect lives inside
 * the mounted ChatBar, so Session A's queue is not observed when A is offscreen.
 *
 * The retry ledger lives ON THE PERSISTED ENTRY, not in a ref. A ref is reset by
 * every app launch, so an entry that can never succeed (its attachment was
 * deleted months ago) spends its whole retry budget again on every single
 * start — forever. Entries that exhaust the budget are dead-lettered in
 * localStorage and skipped from then on, until the user retries or deletes them
 * from the queue panel.
 */
export function useBackgroundQueueDrain({
  enabled,
  pathProbe = desktopPathProbe(),
  runtimeIdByStoredSessionIdRef,
  selectedStoredSessionId,
  submitText
}: BackgroundQueueDrainOptions) {
  const { t } = useI18n()
  const queuedPromptsBySession = useStore($queuedPromptsBySession)
  const submitTextRef = useRef(submitText)
  const drainingSessionIdsRef = useRef(new Set<string>())
  const retryTimersRef = useRef<number[]>([])
  const [retryTick, setRetryTick] = useState(0)
  const parkedQueueSessions = useStore($parkedQueueSessions)
  const workingSessionIds = useStore($workingSessionIds)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    submitTextRef.current = submitText
  }, [submitText])

  const scheduleRetry = useCallback(() => {
    if (typeof window === 'undefined') {
      return
    }

    const timer = window.setTimeout(() => {
      retryTimersRef.current = retryTimersRef.current.filter(id => id !== timer)
      setRetryTick(tick => tick + 1)
    }, BACKGROUND_DRAIN_RETRY_MS)

    retryTimersRef.current.push(timer)
  }, [])

  useEffect(
    () => () => {
      for (const timer of retryTimersRef.current) {
        window.clearTimeout(timer)
      }

      retryTimersRef.current = []
    },
    []
  )

  // One toast per entry that transitions into the dead-letter box — never one
  // per launch. The notification id is the entry id, so a re-notify for the
  // same entry replaces rather than stacks.
  const notifyStuck = useCallback(
    (entry: QueuedPromptEntry) => {
      notify({
        id: `composer-background-queue-stuck-${entry.id}`,
        kind: 'error',
        title: t.composer.queueStuckTitle,
        message: t.composer.queueStuckBody
      })
    },
    [t]
  )

  const drainSessionQueue = useCallback(
    (sessionKey: string, entry: QueuedPromptEntry) => {
      if (drainingSessionIdsRef.current.has(sessionKey)) {
        return
      }

      drainingSessionIdsRef.current.add(sessionKey)

      // Book the attempt on the persisted entry. Retrying is the `finally`
      // block's job, so this only records — and toasts once, on the edge.
      const onFail = (error: unknown) => {
        if (recordDrainFailure(sessionKey, entry.id, errorDetail(error) || undefined)?.becameStuck) {
          notifyStuck(entry)
        }
      }

      void Promise.resolve()
        .then(async () => {
          const liveEntry = getQueuedPrompts(sessionKey).find(candidate => candidate.id === entry.id)

          if (!liveEntry) {
            return true
          }

          // Preflight BEFORE submitting: the submit pipeline paints its
          // optimistic bubble as soon as it has a session id, so a dead
          // attachment discovered mid-send is what leaves transcript residue.
          // A vanished file is permanent — dead-letter it now with the full,
          // untruncated path rather than burning the retry budget on it.
          const missingPath = await firstMissingAttachmentPath(liveEntry.attachments, pathProbe)

          if (missingPath) {
            if (markQueuedPromptStuck(sessionKey, liveEntry.id, 'attachment-missing', missingPath)) {
              notifyStuck(liveEntry)
            }

            return true
          }

          const runtimeSessionId = runtimeIdByStoredSessionIdRef.current.get(sessionKey) ?? null

          const accepted = await Promise.resolve(
            submitTextRef.current(liveEntry.text, {
              attachments: liveEntry.attachments,
              fromQueue: true,
              sessionId: runtimeSessionId,
              storedSessionId: sessionKey
            })
          )

          if (accepted === false) {
            return false
          }

          removeQueuedPrompt(sessionKey, liveEntry.id)
          resetBrowseState(runtimeSessionId)

          return true
        })
        .then(accepted => {
          if (!accepted) {
            onFail(null)
          }
        })
        .catch(onFail)
        .finally(() => {
          drainingSessionIdsRef.current.delete(sessionKey)

          // Re-poll if this session still has sendable work. The store write
          // that ends a drain (entry removed, or dead-lettered) can re-run the
          // effect while this lock is still held, and nothing schedules another
          // pass once it clears — which strands every entry queued behind the
          // one we just finished. One bounded tick per completed drain: it
          // either makes progress or finds nothing drainable and stops.
          if (nextDrainableQueuedPrompt(getQueuedPrompts(sessionKey))) {
            scheduleRetry()
          }
        })
    },
    [notifyStuck, pathProbe, runtimeIdByStoredSessionIdRef, scheduleRetry]
  )

  useEffect(() => {
    if (!enabled) {
      return
    }

    // Queue keys prefer the lineage root (resolveComposerSessionKey) while
    // $workingSessionIds / selection may hold the compression tip. Strict
    // equality then mis-classifies a busy or selected chat as idle/offscreen.
    const sessions = $sessions.get()
    const working = [...workingSessionIds]

    for (const [sessionKey, entries] of Object.entries(queuedPromptsBySession)) {
      const isSelected =
        Boolean(selectedStoredSessionId) && idsShareLineage(sessionKey, selectedStoredSessionId!, sessions)

      const isBusy = working.some(workingId => idsShareLineage(sessionKey, workingId, sessions))

      if (isSelected || drainingSessionIdsRef.current.has(sessionKey)) {
        continue
      }

      const entry = nextDrainableQueuedPrompt(entries)

      if (
        !entry ||
        !shouldAutoDrain({
          isBusy,
          parked: Boolean(parkedQueueSessions[sessionKey]),
          queueLength: drainableQueuedPromptCount(entries)
        })
      ) {
        continue
      }

      drainSessionQueue(sessionKey, entry)
    }
  }, [
    drainSessionQueue,
    enabled,
    parkedQueueSessions,
    queuedPromptsBySession,
    retryTick,
    selectedStoredSessionId,
    workingSessionIds
  ])
}

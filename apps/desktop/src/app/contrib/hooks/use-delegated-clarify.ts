import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { type Translations, useI18n } from '@/i18n'
import { textPart } from '@/lib/chat-messages'
import { DELEGATED_CLARIFY_ANSWER, formatDelegatedWait } from '@/lib/delegated-spawn'
import { $clarifyRequests, type ClarifyRequest, clearClarifyRequest } from '@/store/clarify'
import { $delegatedTimeoutByRuntimeId } from '@/store/session-states'

import type { GatewayRequest } from '../../session/hooks/use-prompt-actions/utils'
import type { ClientSessionState } from '../../types'

type UpdateSessionState = (
  sessionId: string,
  updater: (state: ClientSessionState) => ClientSessionState,
  storedSessionId?: string | null
) => ClientSessionState

interface AnswerForNobodyDeps {
  request: ClarifyRequest
  timeoutMs: number
  requestGateway: GatewayRequest
  updateSessionState: UpdateSessionState
  copy: Translations['assistant']['clarify']
}

/**
 * Answer a clarify prompt nobody is there to answer, and leave a record.
 *
 * Split out of the hook so the decision — what we send, what we write down,
 * what we do when the gateway has already moved on — can be exercised without
 * mounting React or arming a real timer.
 *
 * Returns what actually happened, which is not always what we intended.
 */
export async function answerClarifyForNobody({
  request,
  timeoutMs,
  requestGateway,
  updateSessionState,
  copy
}: AnswerForNobodyDeps): Promise<'answered' | 'skipped' | 'stale'> {
  const sessionId = request.sessionId

  if (!sessionId) {
    return 'skipped'
  }

  try {
    await requestGateway('clarify.respond', {
      request_id: request.requestId,
      answer: DELEGATED_CLARIFY_ANSWER
    })
  } catch {
    // The gateway rejects an answer it is no longer waiting for — its own 300s
    // clarify timeout got there first, or the turn was interrupted. Either way
    // the session is not stuck, it already moved on, so drop the dead card and
    // write nothing rather than claim an answer that never landed.
    clearClarifyRequest(request.requestId, sessionId)

    return 'stale'
  }

  clearClarifyRequest(request.requestId, sessionId)

  // Say it happened in the transcript, next to the question it answered. A
  // toast would be the wrong place: the person who needs to know a decision
  // was made without them is reading this chat tomorrow, not watching it now.
  updateSessionState(sessionId, state => ({
    ...state,
    needsInput: false,
    messages: [
      ...state.messages,
      {
        id: `system-delegated-${request.requestId}`,
        role: 'system' as const,
        parts: [textPart(copy.autoAnswered(formatDelegatedWait(timeoutMs)))]
      }
    ]
  }))

  return 'answered'
}

interface DelegatedClarifyParams {
  requestGateway: GatewayRequest
  updateSessionState: UpdateSessionState
}

/**
 * The half of delegated mode that does not depend on the model cooperating.
 *
 * `hermes desktop spawn --delegated` tells the session up front not to ask
 * anything (see `@/lib/delegated-spawn`), and most of the time that is the end
 * of it. But a preamble is advice, and an unattended run cannot rest on advice:
 * if the agent raises a clarify prompt anyway there is nobody to answer it, and
 * the chat stops there. That is the exact failure this exists for — a spawned
 * session that quite correctly asked where to put its output, and quite
 * correctly waited forever, because the brief had never said.
 *
 * So for a delegated session, and only there, a pending clarify gets a
 * deadline. When it passes the app answers, records that in the transcript, and
 * the turn carries on.
 *
 * Deliberately narrow:
 *
 * - **Delegated sessions only.** An ordinary chat has someone in front of it;
 *   answering their question for them would be a bug, not a feature.
 * - **Clarify only.** Approvals already fail closed on the backend after
 *   `approvals.timeout` — denied, on the grounds that silence is not consent —
 *   so they cannot park a session forever and must not be auto-allowed here.
 * - **One timer per request, dropped the moment the request goes.** Someone who
 *   is nearby can still take the question; delegated means unattended, not
 *   locked out.
 */
export function useDelegatedClarify({ requestGateway, updateSessionState }: DelegatedClarifyParams): void {
  const { t } = useI18n()
  const clarifyRequests = useStore($clarifyRequests)
  // Reference-stable projection, NOT $sessionStates — that republishes on every
  // streamed token and would re-run this effect at delta rates.
  const delegatedTimeouts = useStore($delegatedTimeoutByRuntimeId)

  // Read at fire time, so a re-render mid-wait never restarts the clock.
  const copyRef = useRef(t.assistant.clarify)
  const requestGatewayRef = useRef(requestGateway)
  const updateSessionStateRef = useRef(updateSessionState)
  const timersRef = useRef(new Map<string, ReturnType<typeof setTimeout>>())

  useEffect(() => {
    copyRef.current = t.assistant.clarify
    requestGatewayRef.current = requestGateway
    updateSessionStateRef.current = updateSessionState
  }, [requestGateway, t, updateSessionState])

  useEffect(() => {
    const timers = timersRef.current
    const armed = new Set<string>()

    for (const request of Object.values(clarifyRequests)) {
      const timeoutMs = request.sessionId ? delegatedTimeouts[request.sessionId] : undefined

      if (timeoutMs === undefined) {
        continue
      }

      armed.add(request.requestId)

      if (timers.has(request.requestId)) {
        continue
      }

      timers.set(
        request.requestId,
        setTimeout(() => {
          timers.delete(request.requestId)
          void answerClarifyForNobody({
            copy: copyRef.current,
            request,
            requestGateway: requestGatewayRef.current,
            timeoutMs,
            updateSessionState: updateSessionStateRef.current
          })
        }, timeoutMs)
      )
    }

    // A request that has gone — answered by a person, interrupted, or ended
    // with its turn — must not still be on a clock.
    for (const [requestId, timer] of timers) {
      if (!armed.has(requestId)) {
        clearTimeout(timer)
        timers.delete(requestId)
      }
    }
  }, [clarifyRequests, delegatedTimeouts])

  // A torn-down window must not answer on its way out.
  useEffect(() => {
    const timers = timersRef.current

    return () => {
      for (const timer of timers.values()) {
        clearTimeout(timer)
      }

      timers.clear()
    }
  }, [])
}

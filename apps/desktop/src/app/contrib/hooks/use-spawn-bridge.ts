import { useEffect, useRef } from 'react'

import { applyDelegationContract, normalizeDelegatedTimeoutMs } from '@/lib/delegated-spawn'
import { $freshDraftReady } from '@/store/session'
import { isSecondaryWindow } from '@/store/windows'

import type { SubmitTextOptions } from '../../session/hooks/use-prompt-actions/utils'
import type { SessionCreateOverrides } from '../../session/session-overrides'
import type { ClientSessionState } from '../../types'

/** How long to wait for a new-chat draft to settle before submitting anyway. */
export const SPAWN_DRAFT_SETTLE_TIMEOUT_MS = 5000

/**
 * Resolve once the new-chat draft has settled.
 *
 * `startFreshSession` navigates to the new-chat route, and that navigation lands
 * asynchronously. Submitting in the same tick captures the PREVIOUS route token,
 * so when the navigation completes mid-submit the submit sees the route move
 * under it, treats it as the user switching away, and aborts — silently, having
 * already told the caller 202. Measured 2026-08-02: the first spawn after boot
 * ran with `freshDraftReady=false` and returned false; the next one, by which
 * point the draft had settled, ran fine.
 *
 * Resolving on a timeout rather than hanging keeps a spawn best-effort if the
 * flag never arrives: a submit that then fails is visible, a spawn that never
 * happens is not.
 */
function whenFreshDraftReady(timeoutMs = SPAWN_DRAFT_SETTLE_TIMEOUT_MS): Promise<void> {
  if ($freshDraftReady.get()) {
    return Promise.resolve()
  }

  return new Promise(resolve => {
    let settled = false

    const finish = () => {
      if (settled) {
        return
      }

      settled = true
      unbind()
      clearTimeout(timer)
      resolve()
    }

    const unbind = $freshDraftReady.listen(ready => {
      if (ready) {
        finish()
      }
    })

    const timer = setTimeout(finish, timeoutMs)
  })
}

interface SpawnPayload {
  prompt: string
  model?: string
  provider?: string
  profile?: string
  toolsets?: string[]
  delegated?: boolean
  delegatedTimeoutMs?: number
}

interface SpawnBridgeParams {
  startFreshSession: (options?: boolean | { replaceRoute?: boolean }) => void
  submitText: (rawText: string, options?: SubmitTextOptions) => Promise<boolean>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

/**
 * Run chats that a local CLI (`hermes desktop spawn`) asked this app to start.
 *
 * The whole point is that the spawned session is OURS. Electron main hands us
 * the request; we then walk the same two steps a person does — start a fresh
 * chat, submit the text — so the session is created on this window's gateway
 * socket. That matters concretely: the gateway routes streaming events by the
 * transport stored on the session, and `prompt.submit` re-pins that transport
 * to the caller, so a session created by anything other than this renderer
 * would stream its tokens somewhere we never see. Going through `submitText`
 * means the transcript, tool cards, titles and sidebar all behave exactly as
 * they do for a typed message, with no parallel streaming path to maintain.
 *
 * Model/provider/profile ride along as per-session overrides rather than being
 * written into the composer's selection stores, which persist — a spawn must
 * not silently change the model the user gets on their next chat.
 *
 * A `--delegated` spawn additionally gets the unattended contract prepended to
 * its prompt, and its session is marked so `useDelegatedClarify` will answer a
 * clarify prompt rather than let the chat sit waiting for nobody. See
 * `@/lib/delegated-spawn` for why both halves exist.
 */
export function useSpawnBridge({ startFreshSession, submitText, updateSessionState }: SpawnBridgeParams): void {
  const startFreshSessionRef = useRef(startFreshSession)
  const submitTextRef = useRef(submitText)
  const updateSessionStateRef = useRef(updateSessionState)
  // Tail of the serial spawn chain — see the comment at the enqueue site.
  const chainRef = useRef<Promise<void>>(Promise.resolve())

  useEffect(() => {
    startFreshSessionRef.current = startFreshSession
    submitTextRef.current = submitText
    updateSessionStateRef.current = updateSessionState
  }, [startFreshSession, submitText, updateSessionState])

  useEffect(() => {
    // Secondary windows render a single chat and own no new-chat route.
    if (isSecondaryWindow()) {
      return
    }

    const unsubscribe = window.hermesDesktop?.onSpawnSession?.((payload: SpawnPayload) => {
      const prompt = typeof payload?.prompt === 'string' ? payload.prompt : ''

      if (!prompt.trim()) {
        return
      }

      const overrides: SessionCreateOverrides = {}

      if (payload.model) {
        overrides.model = payload.model
      }

      if (payload.provider) {
        overrides.provider = payload.provider
      }

      if (payload.profile) {
        overrides.profile = payload.profile
      }

      // The contract goes in the prompt itself so the model reads it before it
      // does anything; the timeout is hung on the session so the auto-answer
      // still fires if it ignores the contract and asks anyway.
      const delegated = payload.delegated === true
      const delegatedTimeoutMs = delegated ? normalizeDelegatedTimeoutMs(payload.delegatedTimeoutMs) : null
      const text = delegated ? applyDelegationContract(prompt) : prompt

      // Spawns run ONE AT A TIME, chained on the previous one's submit.
      //
      // They cannot be fired concurrently: `createBackendSessionForSend` guards
      // against the user navigating away mid-create by comparing the route token
      // and active-session refs it started with, and closing the session it just
      // made if they moved. A second spawn calling `startFreshSession` while the
      // first was still awaiting `session.create` tripped exactly that guard —
      // the first spawn silently closed its own session and never ran, even
      // though its caller had already been told 202. Measured 2026-08-02: two
      // spawns 2s apart produced only the second session.
      //
      // Chaining costs nothing in the common case: `submitText` resolves once
      // the turn has been submitted (`prompt.submit` returns "streaming"
      // immediately), not when generation finishes — so the first spawn keeps
      // streaming while the second starts.
      chainRef.current = chainRef.current
        .then(async () => {
          // Clear any half-typed draft's session binding first, so the submit
          // below creates a NEW session instead of appending to whatever chat
          // happens to be open. `replaceRoute` keeps the user's history from
          // filling with new-chat entries they never visited.
          startFreshSessionRef.current({ replaceRoute: true })

          // Let the new-chat route land before submitting — see the helper.
          await whenFreshDraftReady()

          await submitTextRef.current(text, {
            attachments: [],
            ...(overrides.model || overrides.provider || overrides.profile ? { sessionOverrides: overrides } : {}),
            ...(delegatedTimeoutMs !== null
              ? {
                  onSessionCreated: (sessionId: string) =>
                    void updateSessionStateRef.current(sessionId, state => ({ ...state, delegatedTimeoutMs }))
                }
              : {})
          })
        })
        // One spawn failing must not poison the chain for the next.
        .catch(() => undefined)
    })

    // Tell main we're listening so a spawn that raced app boot gets flushed.
    void window.hermesDesktop?.signalSpawnReady?.()

    return () => unsubscribe?.()
  }, [])
}

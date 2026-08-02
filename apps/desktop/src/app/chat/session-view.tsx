import type { ReadableAtom } from 'nanostores'
import { createContext, useContext } from 'react'

import type { ChatMessage } from '@/lib/chat-messages'
import {
  $activeSessionId,
  $awaitingResponse,
  $busy,
  $currentCwd,
  $lastVisibleMessageIsUser,
  $messages,
  $messagesEmpty,
  $primaryFastMode,
  $primaryModel,
  $primaryProvider,
  $primaryReasoningEffort,
  $selectedStoredSessionId
} from '@/store/session'

/**
 * SESSION VIEW — the store surface a ChatView renders from. The PRIMARY view
 * is the app's classic global atoms (route-driven active session, untouched
 * fast path). A session TILE provides the same shape computed from its
 * session's slice of `$sessionStates`, so the identical ChatView tree renders
 * either — one chat surface, N sessions on screen.
 *
 * Everything is atoms (not values) so subscription granularity survives:
 * ChatView subscribes only to the coarse edges; `$messages` stays boundary-
 * only exactly like the primary view's perf contract.
 */
export interface SessionView {
  kind: 'primary' | 'tile'
  $runtimeId: ReadableAtom<string | null>
  $storedId: ReadableAtom<string | null>
  $messages: ReadableAtom<ChatMessage[]>
  $busy: ReadableAtom<boolean>
  $awaitingResponse: ReadableAtom<boolean>
  $messagesEmpty: ReadableAtom<boolean>
  $lastVisibleIsUser: ReadableAtom<boolean>
  $cwd: ReadableAtom<string>
  $model: ReadableAtom<string>
  $provider: ReadableAtom<string>
  $fast: ReadableAtom<boolean>
  $reasoningEffort: ReadableAtom<string>
}

export const PRIMARY_SESSION_VIEW: SessionView = {
  kind: 'primary',
  $awaitingResponse,
  $busy,
  $cwd: $currentCwd,
  // The OPEN session's fast mode, not the composer's persisted pick — those
  // are the same value only on a fresh draft (see $primaryModel).
  $fast: $primaryFastMode,
  $lastVisibleIsUser: $lastVisibleMessageIsUser,
  $messages,
  $messagesEmpty,
  // The OPEN session's model, not the composer's persisted pick — those are
  // the same value only on a fresh draft (see $primaryModel).
  $model: $primaryModel,
  $provider: $primaryProvider,
  $reasoningEffort: $primaryReasoningEffort,
  $runtimeId: $activeSessionId,
  $storedId: $selectedStoredSessionId
}

const SessionViewContext = createContext<SessionView>(PRIMARY_SESSION_VIEW)

export const SessionViewProvider = SessionViewContext.Provider

export const useSessionView = (): SessionView => useContext(SessionViewContext)

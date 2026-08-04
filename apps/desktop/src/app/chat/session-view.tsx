import { computed, type ReadableAtom } from 'nanostores'
import { createContext, useContext } from 'react'

import type { ClientSessionState } from '@/app/types'
import type { ChatMessage } from '@/lib/chat-messages'
import {
  $activeSessionId,
  $awaitingResponse,
  $busy,
  $currentCwd,
  $messages,
  $primaryFastMode,
  $primaryModel,
  $primaryProvider,
  $primaryReasoningEffort,
  $selectedStoredSessionId
} from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { lastVisibleMessageIsUser } from './thread-loading'

/**
 * SESSION VIEW — the store surface a ChatView renders from. Every session,
 * including the one in the workspace pane, renders from ITS OWN slice of
 * `$sessionStates`. The workspace pane is just the first tab: a session
 * surface with no privileged state of its own.
 *
 * That symmetry is load-bearing. The pane used to render off the global
 * `$messages`/`$busy` atoms — a mirror of whichever session was active — so
 * with two turns in flight (⌘T tabs made that routine), navigating away from
 * a still-streaming session left it painting into the surface now showing a
 * different conversation. Reading the per-session slice makes that
 * structurally impossible rather than merely guarded.
 *
 * The global atoms stay the DRAFT surface: a new chat has no runtime id, and
 * therefore no slice, until its first turn creates one.
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

/** The active session's own slice, or `undefined` while it's a draft. */
const $primaryState = computed([$activeSessionId, $sessionStates], (runtimeId, states) =>
  runtimeId ? states[runtimeId] : undefined
)

/**
 * Read one field from the active session's slice, falling back to the global
 * draft atom while no runtime exists yet. Once a session HAS a slice, that
 * slice is authoritative — a background session publishing its own state can
 * never reach this view.
 */
function primaryField<T>(select: (state: ClientSessionState) => T, $draft: ReadableAtom<T>): ReadableAtom<T> {
  const $field: ReadableAtom<T> = computed([$primaryState, $draft], (state, draft: T) =>
    state ? select(state) : draft
  )

  return $field
}

const $primaryMessages = primaryField<ChatMessage[]>(state => state.messages, $messages)

export const PRIMARY_SESSION_VIEW: SessionView = {
  kind: 'primary',
  $awaitingResponse: primaryField<boolean>(state => state.awaitingResponse, $awaitingResponse),
  $busy: primaryField<boolean>(state => state.busy, $busy),
  $cwd: primaryField<string>(state => state.cwd, $currentCwd),
  // Draft fallback is the OPEN session's fast mode ($primaryFastMode), not the
  // composer's persisted pick — those are the same value only on a fresh draft.
  // A session mid-resume has an id but no slice yet, and must still paint its
  // own fast mode rather than whatever the composer last selected (#318/#322).
  $fast: primaryField<boolean>(state => state.fast, $primaryFastMode),
  $lastVisibleIsUser: computed($primaryMessages, lastVisibleMessageIsUser),
  $messages: $primaryMessages,
  $messagesEmpty: computed($primaryMessages, messages => messages.length === 0),
  // Same rule as $fast above: the OPEN session's model/provider/effort, falling
  // back to the composer's sticky pick only when nothing owns the display.
  $model: primaryField<string>(state => state.model, $primaryModel),
  $provider: primaryField<string>(state => state.provider, $primaryProvider),
  $reasoningEffort: primaryField<string>(state => state.reasoningEffort, $primaryReasoningEffort),
  $runtimeId: $activeSessionId,
  $storedId: $selectedStoredSessionId
}

const SessionViewContext = createContext<SessionView>(PRIMARY_SESSION_VIEW)

export const SessionViewProvider = SessionViewContext.Provider

export const useSessionView = (): SessionView => useContext(SessionViewContext)

/**
 * Delegated spawns — the rules that make an unattended chat finish by itself.
 *
 * `hermes desktop spawn --delegated` says "nobody is watching this one". That
 * changes what correct behaviour is: an agent that stops to ask "where should I
 * put this?" is doing the right thing for a person sitting in front of the app
 * and the wrong thing for a script that walked away. A delegated run has to
 * reach an answer on its own or say plainly that it could not.
 *
 * Two layers, deliberately, because the first one is advice and the second one
 * is a guarantee:
 *
 *  1. The contract below is prepended to the prompt, so the model knows up
 *     front not to ask and what to do with an underspecified brief. This is the
 *     path we want taken — the turn simply never raises a card.
 *  2. If it asks anyway, `delegatedClarifyAnswer` decides the reply and a timer
 *     sends it. Prompting is not enforcement; a model that ignores the preamble
 *     must not be able to park the session forever.
 *
 * Everything here is pure so both layers can be tested without a gateway.
 */

/**
 * How long a delegated session waits for a person before answering a clarify
 * card itself.
 *
 * Long enough that a human who *is* nearby can still take the question — the
 * flag means unattended, not "refuse human input" — and short enough that an
 * abandoned run resolves in a coffee break rather than overnight.
 */
export const DELEGATED_CLARIFY_TIMEOUT_MS = 120_000

/** Bounds for the caller-supplied timeout. Below the floor the card would be
 *  gone before a person could read it; above the ceiling "unattended" stops
 *  meaning anything. */
export const DELEGATED_CLARIFY_TIMEOUT_MIN_MS = 5_000
export const DELEGATED_CLARIFY_TIMEOUT_MAX_MS = 60 * 60_000

/**
 * Clamp an untrusted timeout to something sane, falling back to the default
 * for anything non-finite.
 */
export function normalizeDelegatedTimeoutMs(value: unknown): number {
  const ms = typeof value === 'number' ? value : Number.NaN

  if (!Number.isFinite(ms) || ms <= 0) {
    return DELEGATED_CLARIFY_TIMEOUT_MS
  }

  return Math.min(DELEGATED_CLARIFY_TIMEOUT_MAX_MS, Math.max(DELEGATED_CLARIFY_TIMEOUT_MIN_MS, Math.round(ms)))
}

/**
 * The contract prepended to a delegated prompt.
 *
 * Written as instructions to the agent, in plain language, because that is what
 * it is. The order matters: the reason comes first so the rest reads as
 * consequences rather than arbitrary rules.
 *
 * The delivery rule is the one that earns its place. The case this was built
 * for was a spawned session that stopped to ask where to put its output — a
 * fair question that simply had no one to answer it. Naming the default (say it
 * here, in the chat) removes the question instead of suppressing it.
 */
export const DELEGATION_CONTRACT = [
  'This chat was started by a script. No one is watching it, so nobody can answer a question or click a button.',
  '',
  'How to run this turn:',
  '- Do not ask anything. Do not raise a prompt, a card, or a confirmation — there is nobody there to respond, and the session will just sit.',
  '- Where the brief leaves a choice open, take the safest option you can undo, note it, and keep going.',
  '- Where the brief does not say what to do with the result, put the result in this chat, in your reply. Do not create or change files, and do not send anything to anyone outside this chat, unless the brief asked for that.',
  '- Say what you assumed. A short "Assumptions" list near the end is enough.',
  '- Finish with DONE on its own line if you got there, or BLOCKER: followed by the one thing that stopped you.',
  '',
  'The brief follows.'
].join('\n')

/**
 * Prepend the contract to a spawned prompt.
 *
 * The contract goes into the user message rather than the system prompt on
 * purpose: the system prompt has to stay byte-stable for the life of a
 * conversation or per-conversation caching is lost (root AGENTS.md), and skill
 * slash commands already inject this way for the same reason. It also means the
 * operator can read exactly what the run was told, in the transcript.
 */
export function applyDelegationContract(prompt: string): string {
  const brief = prompt.trim()

  if (!brief) {
    return brief
  }

  return `${DELEGATION_CONTRACT}\n\n${brief}`
}

/**
 * Render a wait as people say it — "2m", "45s", "1m 30s" — for the transcript
 * note. Locale-independent on purpose: the surrounding sentence is translated,
 * this is the number in it.
 */
export function formatDelegatedWait(ms: number): string {
  const totalSeconds = Math.max(1, Math.round(ms / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60

  if (!minutes) {
    return `${seconds}s`
  }

  return seconds ? `${minutes}m ${seconds}s` : `${minutes}m`
}

/**
 * What a delegated session sends back when it answers a clarify card itself.
 *
 * This lands verbatim in the tool result as `user_response`, so it is read by
 * the agent, not by a person. Two things it deliberately is not:
 *
 * - **Not the empty string.** Empty is what the Skip button sends, and the
 *   agent cannot tell an empty answer from a person who chose not to say. It
 *   learns nothing and is free to ask again.
 * - **Not one of the offered choices.** At this layer every choice is just a
 *   string; "keep it" and "delete it" are indistinguishable to code that cannot
 *   read the question. Picking one would be a guess with consequences attached.
 *
 * Handing the decision back with the rule to apply is the safest move available
 * — and it is what every other Hermes surface already does when nobody answers
 * (the CLI's clarify timeout, the gateway's, one-shot mode).
 */
export const DELEGATED_CLARIFY_ANSWER = [
  'No one is available to answer — this chat was started by a script and nobody is watching it.',
  'Choose the safest option you can undo, say which one you chose and why, and carry on.',
  'If the question was about where the result should go, put it in this chat.',
  'Do not ask again.'
].join(' ')

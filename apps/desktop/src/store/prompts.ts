import { atom, computed, type ReadableAtom } from 'nanostores'

import { $activeSessionId } from './session'

// Blocking interactive prompts the gateway raises mid-turn. Each maps to a
// `*.request` event the Python side emits while it blocks the agent thread
// waiting for a `*.respond` RPC. Without a renderer for these, the agent
// silently stalls until its timeout (default 5 min) and the tool is BLOCKED.
//
// Like clarify, every prompt is parked under the runtime session id that raised
// it (not one shared slot), so a *background* session running concurrently can
// raise an approval/sudo/secret prompt and have it wait — surfaced via the
// sidebar "needs input" badge — until the user switches to that chat. The
// exported $*Request view is scoped to the active session, so a background
// prompt never hijacks the foreground.

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export type PromptAudienceKind = 'any_member' | 'originator' | 'owner_admin' | 'principal' | 'role' | 'unknown'

export interface PromptEntityRef {
  alias?: string
  deviceId?: string
  displayName?: string
  endpoint?: string
  meshId?: string
  orgId?: string
  participantId?: string
  platform?: string
  platformUserId?: string
  principalId?: string
}

export interface PromptAudienceRef {
  kind: PromptAudienceKind
  label?: string
  principalIds?: string[]
  roleIds?: string[]
}

export interface PromptRequestContext {
  answeredBy?: PromptEntityRef
  answeredVia?: string
  claimedBy?: PromptEntityRef
  meshId?: string
  orgId?: string
  requestedBy?: PromptEntityRef
  requestedVia?: string
  sessionParticipantId?: string
  source?: PromptEntityRef
  targetAudience?: PromptAudienceRef
}

interface KeyedPrompt {
  context?: PromptRequestContext
  sessionId: string | null
}

interface PromptStore<T extends KeyedPrompt> {
  $active: ReadableAtom<null | T>
  $all: ReadableAtom<Record<string, T>>
  clear: (sessionId?: string | null, requestId?: string) => void
  reset: () => void
  set: (request: T) => void
}

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}

function readString(row: Record<string, unknown>, ...keys: string[]): string | undefined {
  for (const key of keys) {
    const value = row[key]

    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }

  return undefined
}

function readStringList(row: Record<string, unknown>, ...keys: string[]): string[] | undefined {
  for (const key of keys) {
    const value = row[key]

    if (Array.isArray(value)) {
      const out = value
        .filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
        .map(item => item.trim())

      if (out.length) {
        return out
      }
    }
  }

  return undefined
}

function readEntityRef(value: unknown): PromptEntityRef | undefined {
  if (typeof value === 'string' && value.trim()) {
    return { displayName: value.trim() }
  }

  const row = asRecord(value)

  if (Object.keys(row).length === 0) {
    return undefined
  }

  const ref: PromptEntityRef = {
    alias: readString(row, 'alias', 'nickname', 'nick'),
    deviceId: readString(row, 'deviceId', 'device_id'),
    displayName: readString(row, 'displayName', 'display_name', 'name', 'label'),
    endpoint: readString(row, 'endpoint', 'via'),
    meshId: readString(row, 'meshId', 'mesh_id'),
    orgId: readString(row, 'orgId', 'org_id', 'organizationId', 'organization_id'),
    participantId: readString(row, 'participantId', 'participant_id'),
    platform: readString(row, 'platform', 'source'),
    platformUserId: readString(row, 'platformUserId', 'platform_user_id', 'userId', 'user_id'),
    principalId: readString(row, 'principalId', 'principal_id', 'humanId', 'human_id')
  }

  return Object.values(ref).some(Boolean) ? ref : undefined
}

function normalizeAudienceKind(value: unknown): PromptAudienceKind {
  if (typeof value !== 'string') {
    return 'unknown'
  }

  const normalized = value.trim().toLowerCase().replace(/[-\s]+/g, '_')

  if (
    normalized === 'any_member' ||
    normalized === 'originator' ||
    normalized === 'owner_admin' ||
    normalized === 'principal' ||
    normalized === 'role'
  ) {
    return normalized
  }

  if (normalized === 'admin' || normalized === 'owner' || normalized === 'owners' || normalized === 'owner_or_admin') {
    return 'owner_admin'
  }

  if (normalized === 'anyone' || normalized === 'member') {
    return 'any_member'
  }

  return 'unknown'
}

function readAudienceRef(value: unknown): PromptAudienceRef | undefined {
  if (typeof value === 'string' && value.trim()) {
    const kind = normalizeAudienceKind(value)

    return kind === 'unknown' ? { kind, label: value.trim() } : { kind }
  }

  const row = asRecord(value)

  if (Object.keys(row).length === 0) {
    return undefined
  }

  const kind = normalizeAudienceKind(row.kind ?? row.type ?? row.audience)
  const label = readString(row, 'label', 'displayName', 'display_name', 'name')
  const principalIds = readStringList(row, 'principalIds', 'principal_ids')
  const roleIds = readStringList(row, 'roleIds', 'role_ids')

  if (kind === 'unknown' && !label && !principalIds?.length && !roleIds?.length) {
    return undefined
  }

  return { kind, label, principalIds, roleIds }
}

export function normalizePromptContext(
  payload: unknown,
  defaults: { targetAudience?: PromptAudienceRef } = {}
): PromptRequestContext | undefined {
  const row = asRecord(payload)
  const source = readEntityRef(row.source_actor ?? row.sourceActor ?? row.source)
  const requestedBy = readEntityRef(row.requested_by ?? row.requestedBy ?? row.actor ?? source)
  const claimedBy = readEntityRef(row.claimed_by ?? row.claimedBy)
  const answeredBy = readEntityRef(row.answered_by ?? row.answeredBy)

  const targetAudience =
    readAudienceRef(row.target_audience ?? row.targetAudience ?? row.target ?? row.audience) ??
    defaults.targetAudience

  const context: PromptRequestContext = {
    answeredBy,
    answeredVia: readString(row, 'answeredVia', 'answered_via'),
    claimedBy,
    meshId: readString(row, 'meshId', 'mesh_id') ?? requestedBy?.meshId ?? source?.meshId,
    orgId: readString(row, 'orgId', 'org_id', 'organizationId', 'organization_id') ?? requestedBy?.orgId ?? source?.orgId,
    requestedBy,
    requestedVia: readString(row, 'requestedVia', 'requested_via', 'via') ?? requestedBy?.endpoint ?? source?.platform,
    sessionParticipantId: readString(row, 'sessionParticipantId', 'session_participant_id', 'participantId', 'participant_id'),
    source,
    targetAudience
  }

  return Object.values(context).some(Boolean) ? context : undefined
}

// One per-session prompt kind: a map keyed by session, plus an active-session
// view for the overlays. `clear` drops one session's entry (a request-id
// mismatch is a no-op so a stale resolve can't wipe a newer prompt); with no
// session hint it drops every entry, optionally filtered by request id.
function keyedPromptStore<T extends KeyedPrompt>(): PromptStore<T> {
  const $all = atom<Record<string, T>>({})
  const idOf = (value: T): string | undefined => (value as { requestId?: string }).requestId

  return {
    $all,
    $active: computed([$all, $activeSessionId], (all, activeId) => all[keyFor(activeId)] ?? null),
    reset: () => $all.set({}),
    set: request => $all.set({ ...$all.get(), [keyFor(request.sessionId)]: request }),
    clear(sessionId, requestId) {
      const all = $all.get()

      if (sessionId !== undefined) {
        const key = keyFor(sessionId)
        const current = all[key]

        if (current && !(requestId && idOf(current) !== requestId)) {
          const next = { ...all }
          delete next[key]
          $all.set(next)
        }

        return
      }

      const next = Object.fromEntries(Object.entries(all).filter(([, v]) => requestId && idOf(v) !== requestId))

      if (Object.keys(next).length !== Object.keys(all).length) {
        $all.set(next as Record<string, T>)
      }
    }
  }
}

// Approval is session-keyed on the backend (one in-flight approval per session,
// resolved via approval.respond {choice, session_id}). It carries no request_id,
// unlike sudo/secret which are _block()-style request/response.
export interface ApprovalRequest extends KeyedPrompt {
  // false when the backend won't honor a permanent allow (tirith warning) → hide "Always allow".
  allowPermanent?: boolean
  command: string
  description: string
}

export interface SudoRequest extends KeyedPrompt {
  requestId: string
}

export interface SecretRequest extends KeyedPrompt {
  envVar: string
  prompt: string
  requestId: string
}

const approval = keyedPromptStore<ApprovalRequest>()
const sudo = keyedPromptStore<SudoRequest>()
const secret = keyedPromptStore<SecretRequest>()

export const $approvalRequests = approval.$all
export const $approvalRequest = approval.$active
export const setApprovalRequest = approval.set
export const clearApprovalRequest = approval.clear

export const $sudoRequests = sudo.$all
export const $sudoRequest = sudo.$active
export const setSudoRequest = sudo.set
export const clearSudoRequest = sudo.clear

export const $secretRequests = secret.$all
export const $secretRequest = secret.$active
export const setSecretRequest = secret.set
export const clearSecretRequest = secret.clear

// Drop in-flight prompts for `sessionId` (a turn ended) across all three kinds —
// or every parked prompt when no session is given (global reset / tests).
export function clearAllPrompts(sessionId?: string | null): void {
  if (sessionId === undefined) {
    approval.reset()
    sudo.reset()
    secret.reset()

    return
  }

  approval.clear(sessionId)
  sudo.clear(sessionId)
  secret.clear(sessionId)
}

import type { SessionInfo } from '@/types/hermes'

const ACTIVE_GRACE_SECONDS = 300

export interface SessionEligibilitySummary {
  total: number
  eligible: number
  protected: number
}

interface SessionArchivePreserveOptions {
  activeSessionId?: null | string
  pinnedSessionIds?: Iterable<string>
  selectedSessionId?: null | string
  workingSessionIds?: Iterable<string>
}

function addPreserveId(ids: Set<string>, value: null | string | undefined) {
  const id = String(value ?? '').trim()

  if (id) {ids.add(id)}
}

function aliases(session: SessionInfo): string[] {
  return [session.id, session._lineage_root_id].filter(
    (id): id is string => typeof id === 'string' && id.length > 0
  )
}

export function sessionArchivePreserveIds(
  sessions: SessionInfo[],
  {
    activeSessionId,
    pinnedSessionIds = [],
    selectedSessionId,
    workingSessionIds = []
  }: SessionArchivePreserveOptions
): Set<string> {
  const preserveIds = new Set<string>()

  for (const id of pinnedSessionIds) {addPreserveId(preserveIds, id)}

  for (const id of workingSessionIds) {addPreserveId(preserveIds, id)}
  addPreserveId(preserveIds, selectedSessionId)
  addPreserveId(preserveIds, activeSessionId)

  for (const session of sessions) {
    const sessionAliases = aliases(session)

    if (sessionAliases.some(id => preserveIds.has(id))) {
      sessionAliases.forEach(id => addPreserveId(preserveIds, id))
    }
  }

  return preserveIds
}

export function computeSessionEligibility(
  sessions: SessionInfo[],
  preserveIds: Set<string>,
  now = Date.now() / 1000
): SessionEligibilitySummary {
  let eligible = 0
  let protectedCount = 0
  const seenTargets = new Set<string>()

  for (const session of sessions) {
    const sid = String(session.id).trim()

    if (!sid) {continue}

    const targetId = String(session._lineage_root_id ?? sid).trim() || sid

    if (seenTargets.has(targetId)) {continue}
    seenTargets.add(targetId)

    const startedAt = Number(session.started_at) || 0
    const lastActive = Number(session.last_active) || startedAt
    const recentlyActive = session.ended_at === null && now - lastActive < ACTIVE_GRACE_SECONDS

    if (preserveIds.has(sid) || preserveIds.has(targetId) || recentlyActive) {
      protectedCount += 1
    } else {
      eligible += 1
    }
  }

  return { eligible, protected: protectedCount, total: eligible + protectedCount }
}

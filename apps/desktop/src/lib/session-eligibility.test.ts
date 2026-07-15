import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { computeSessionEligibility, sessionArchivePreserveIds } from './session-eligibility'

const NOW = 2_000_000

function session(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd: null,
    ended_at: NOW - 400,
    id: 'session',
    input_tokens: 0,
    is_active: false,
    last_active: NOW - 600,
    message_count: 2,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: NOW - 1200,
    title: 'Test',
    tool_call_count: 0,
    ...overrides
  }
}

describe('session archive eligibility', () => {
  it('protects pinned, working, selected, and active ids plus their lineage aliases', () => {
    const preserve = sessionArchivePreserveIds(
      [session({ id: 'tip', _lineage_root_id: 'root' })],
      {
        activeSessionId: 'tip',
        pinnedSessionIds: ['pinned'],
        selectedSessionId: 'selected',
        workingSessionIds: ['working']
      }
    )

    expect([...preserve].sort()).toEqual(['pinned', 'root', 'selected', 'tip', 'working'])
  })

  it('matches the backend grace rule and deduplicates compression lineages', () => {
    const summary = computeSessionEligibility(
      [
        session({ id: 'old', _lineage_root_id: 'root' }),
        session({ id: 'tip', _lineage_root_id: 'root' }),
        session({ id: 'recent', ended_at: null, last_active: NOW - 60 }),
        session({ id: 'pinned' })
      ],
      new Set(['pinned']),
      NOW
    )

    expect(summary).toEqual({ eligible: 1, protected: 2, total: 3 })
  })
})

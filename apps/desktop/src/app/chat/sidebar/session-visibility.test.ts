import { describe, expect, it } from 'vitest'

import { deriveSidebarSessionVisibility, type SidebarSessionVisibilityInput } from './session-visibility'

// All inputs default to "nothing loaded, nothing fetched"; each test overrides
// only the fields it cares about so the varied signal stays obvious.
const input = (over: Partial<SidebarSessionVisibilityInput> = {}): SidebarSessionVisibilityInput => ({
  hasArchived: false,
  hasCronJobs: false,
  hasPinned: false,
  sessionCount: 0,
  sessionsInitialLoadComplete: false,
  sessionsLoading: false,
  ...over
})

describe('deriveSidebarSessionVisibility', () => {
  it('shows skeletons during the genuine first load (fetching, nothing yet)', () => {
    expect(deriveSidebarSessionVisibility(input({ sessionsLoading: true }))).toEqual({
      recentsEmptyState: 'skeletons',
      showSessionSections: true,
      showSessionSkeletons: true
    })
  })

  it('settles to the minimal sidebar once the first load completes with zero content', () => {
    expect(deriveSidebarSessionVisibility(input({ sessionsInitialLoadComplete: true }))).toEqual({
      recentsEmptyState: 'empty',
      showSessionSections: false,
      showSessionSkeletons: false
    })
  })

  // The regression this module exists for: a periodic background refresh flips
  // sessionsLoading back to true on an empty-recents account. It must NOT
  // re-show skeletons or re-mount the section, or the whole area flashes in/out.
  it('stays silent on a background refresh after the first load (the flash bug)', () => {
    expect(
      deriveSidebarSessionVisibility(input({ sessionsInitialLoadComplete: true, sessionsLoading: true }))
    ).toEqual({
      recentsEmptyState: 'empty',
      showSessionSections: false,
      showSessionSkeletons: false
    })
  })

  it('keeps the section mounted (no skeletons) while real sessions are loaded', () => {
    expect(deriveSidebarSessionVisibility(input({ sessionCount: 5, sessionsInitialLoadComplete: true }))).toEqual({
      recentsEmptyState: 'all-pinned',
      showSessionSections: true,
      showSessionSkeletons: false
    })
  })

  it('does not flash to skeletons when a refresh runs with sessions already present', () => {
    expect(
      deriveSidebarSessionVisibility(
        input({ sessionCount: 5, sessionsInitialLoadComplete: true, sessionsLoading: true })
      )
    ).toEqual({
      recentsEmptyState: 'all-pinned',
      showSessionSections: true,
      showSessionSkeletons: false
    })
  })

  it('does not strand a skeleton when the first load fails (not loading, never completed)', () => {
    expect(deriveSidebarSessionVisibility(input())).toEqual({
      recentsEmptyState: 'empty',
      showSessionSections: false,
      showSessionSkeletons: false
    })
  })

  it('re-shows skeletons while a failed initial load is being retried', () => {
    expect(deriveSidebarSessionVisibility(input({ sessionsLoading: true }))).toEqual({
      recentsEmptyState: 'skeletons',
      showSessionSections: true,
      showSessionSkeletons: true
    })
  })

  // The bug this change fixes: cron / archived / pinned sections used to vanish
  // whenever recents was empty, because the whole area was gated on recents
  // count alone. Each must now mount the area on its own content.
  it('mounts the area for cron jobs even with zero recent sessions', () => {
    expect(
      deriveSidebarSessionVisibility(input({ hasCronJobs: true, sessionsInitialLoadComplete: true }))
    ).toEqual({
      recentsEmptyState: 'empty',
      showSessionSections: true,
      showSessionSkeletons: false
    })
  })

  it('mounts the area for archived sessions even with zero recent sessions', () => {
    expect(
      deriveSidebarSessionVisibility(input({ hasArchived: true, sessionsInitialLoadComplete: true }))
    ).toEqual({
      recentsEmptyState: 'empty',
      showSessionSections: true,
      showSessionSkeletons: false
    })
  })

  it('mounts the area for a pinned row even with zero recent sessions', () => {
    expect(deriveSidebarSessionVisibility(input({ hasPinned: true, sessionsInitialLoadComplete: true }))).toEqual({
      recentsEmptyState: 'empty',
      showSessionSections: true,
      showSessionSkeletons: false
    })
  })

  // recentsEmptyState discriminates the two non-skeleton empty cases:
  //  - zero recents (area up only for cron/archived/pinned) → neutral "empty",
  //    NOT "all pinned" (a pinned cron row is not a recent to unpin into).
  it('uses the neutral empty copy, not all-pinned, when recents is genuinely empty', () => {
    expect(
      deriveSidebarSessionVisibility(
        input({ hasPinned: true, sessionCount: 0, sessionsInitialLoadComplete: true })
      ).recentsEmptyState
    ).toBe('empty')
  })

  //  - recents exist but every one is pinned (unpinned list empty) → "all-pinned".
  it('uses the all-pinned copy only when recents exist but are all pinned', () => {
    expect(
      deriveSidebarSessionVisibility(
        input({ hasPinned: true, sessionCount: 3, sessionsInitialLoadComplete: true })
      ).recentsEmptyState
    ).toBe('all-pinned')
  })

  it('prefers skeletons over the empty/all-pinned copy during the first load', () => {
    expect(deriveSidebarSessionVisibility(input({ sessionsLoading: true })).recentsEmptyState).toBe('skeletons')
  })
})

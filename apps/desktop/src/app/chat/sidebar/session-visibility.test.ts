import { describe, expect, it } from 'vitest'

import { deriveSidebarSessionVisibility } from './session-visibility'

describe('deriveSidebarSessionVisibility', () => {
  it('shows skeletons during the genuine first load (fetching, nothing yet)', () => {
    expect(
      deriveSidebarSessionVisibility({
        sessionCount: 0,
        sessionsInitialLoadComplete: false,
        sessionsLoading: true
      })
    ).toEqual({ showSessionSkeletons: true, showSessionSections: true })
  })

  it('settles to the minimal sidebar once the first load completes with zero sessions', () => {
    expect(
      deriveSidebarSessionVisibility({
        sessionCount: 0,
        sessionsInitialLoadComplete: true,
        sessionsLoading: false
      })
    ).toEqual({ showSessionSkeletons: false, showSessionSections: false })
  })

  // The regression this module exists for: a periodic background refresh flips
  // sessionsLoading back to true on an empty-recents account. It must NOT
  // re-show skeletons or re-mount the section, or the whole area flashes in/out.
  it('stays silent on a background refresh after the first load (the flash bug)', () => {
    expect(
      deriveSidebarSessionVisibility({
        sessionCount: 0,
        sessionsInitialLoadComplete: true,
        sessionsLoading: true
      })
    ).toEqual({ showSessionSkeletons: false, showSessionSections: false })
  })

  it('keeps the section mounted (no skeletons) while real sessions are loaded', () => {
    expect(
      deriveSidebarSessionVisibility({
        sessionCount: 5,
        sessionsInitialLoadComplete: true,
        sessionsLoading: false
      })
    ).toEqual({ showSessionSkeletons: false, showSessionSections: true })
  })

  it('does not flash to skeletons when a refresh runs with sessions already present', () => {
    expect(
      deriveSidebarSessionVisibility({
        sessionCount: 5,
        sessionsInitialLoadComplete: true,
        sessionsLoading: true
      })
    ).toEqual({ showSessionSkeletons: false, showSessionSections: true })
  })

  it('does not strand a skeleton when the first load fails (not loading, never completed)', () => {
    expect(
      deriveSidebarSessionVisibility({
        sessionCount: 0,
        sessionsInitialLoadComplete: false,
        sessionsLoading: false
      })
    ).toEqual({ showSessionSkeletons: false, showSessionSections: false })
  })

  it('re-shows skeletons while a failed initial load is being retried', () => {
    expect(
      deriveSidebarSessionVisibility({
        sessionCount: 0,
        sessionsInitialLoadComplete: false,
        sessionsLoading: true
      })
    ).toEqual({ showSessionSkeletons: true, showSessionSections: true })
  })
})

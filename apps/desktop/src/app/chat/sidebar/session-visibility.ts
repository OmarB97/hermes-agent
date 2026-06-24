// Visibility of the sidebar's session area, derived from load state.
//
// The skeletons are an *initial-load* affordance only. Once the first fetch has
// settled, every later refresh keeps flipping $sessionsLoading true→false again
// (gateway reconnect, post-turn refetch, cross-window sync, periodic
// revalidation). Re-showing skeletons on those background refreshes makes the
// whole session area flash in and out — especially jarring on an empty-recents
// account, where the section mounts and unmounts wholesale on every refresh.
// Gating skeletons on "initial load complete" makes background refreshes silent.

export interface SidebarSessionVisibilityInput {
  /** A session fetch is in flight right now. */
  sessionsLoading: boolean
  /** The first session fetch has succeeded at least once this run. */
  sessionsInitialLoadComplete: boolean
  /** Number of (recent) sessions currently loaded. */
  sessionCount: number
}

export interface SidebarSessionVisibility {
  /** Render the placeholder skeleton rows (genuine first load only). */
  showSessionSkeletons: boolean
  /** Mount the session area (search, pinned, sessions, cron, archived). */
  showSessionSections: boolean
}

export function deriveSidebarSessionVisibility({
  sessionsLoading,
  sessionsInitialLoadComplete,
  sessionCount
}: SidebarSessionVisibilityInput): SidebarSessionVisibility {
  // Skeletons only during the genuine first load: still fetching, never
  // completed a load, and nothing to show yet. After the first success this is
  // permanently false, so a background refresh never re-flashes the section.
  const showSessionSkeletons = sessionsLoading && !sessionsInitialLoadComplete && sessionCount === 0

  // The session area shows while first-load skeletons are up or once there are
  // real rows. Because skeletons no longer toggle on background refreshes, this
  // stays stable instead of flashing in/out on an empty-recents account.
  const showSessionSections = showSessionSkeletons || sessionCount > 0

  return { showSessionSkeletons, showSessionSections }
}

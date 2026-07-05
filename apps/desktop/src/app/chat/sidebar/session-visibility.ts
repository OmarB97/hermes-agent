// Visibility of the sidebar's session area, derived from load state + content.
//
// The skeletons are an *initial-load* affordance only. Once the first fetch has
// settled, every later refresh keeps flipping $sessionsLoading true→false again
// (gateway reconnect, post-turn refetch, cross-window sync, periodic
// revalidation). Re-showing skeletons on those background refreshes makes the
// whole session area flash in and out — especially jarring on an empty-recents
// account, where the section mounts and unmounts wholesale on every refresh.
// Gating skeletons on "initial load complete" makes background refreshes silent.
//
// The area also must not hinge on recents *alone*. Pinned chats, cron jobs, and
// archived sessions all live in this same area (one DndContext wraps pinned,
// recents, and archived), and each has to appear on its own content — even when
// there are zero recent sessions. Gating the whole area on recents count made
// cron/archived/pinned vanish on an empty-recents account; deriving visibility
// from the union of every section's content fixes that.

export interface SidebarSessionVisibilityInput {
  /** A session fetch is in flight right now. */
  sessionsLoading: boolean
  /** The first session fetch has succeeded at least once this run. */
  sessionsInitialLoadComplete: boolean
  /** Number of (recent) sessions currently loaded. */
  sessionCount: number
  /** At least one pinned row resolves right now (may be a pinned cron/messaging
   * row that is not itself a recent session). */
  hasPinned: boolean
  /** At least one cron job is available to list. */
  hasCronJobs: boolean
  /** At least one archived session is known (loaded, or reported by the server
   * total before the list has been fetched). */
  hasArchived: boolean
}

/** Which placeholder the recents (SESSIONS) list shows when it has no rows. */
export type SidebarRecentsEmptyState = 'all-pinned' | 'empty' | 'skeletons'

export interface SidebarSessionVisibility {
  /** Render the placeholder skeleton rows (genuine first load only). */
  showSessionSkeletons: boolean
  /** Mount the session area (search, pinned, sessions, cron, archived). */
  showSessionSections: boolean
  /** Placeholder to render in the recents list when it has no rows. Only
   * consumed when that list is actually empty; the component decides emptiness. */
  recentsEmptyState: SidebarRecentsEmptyState
}

export function deriveSidebarSessionVisibility({
  sessionsLoading,
  sessionsInitialLoadComplete,
  sessionCount,
  hasPinned,
  hasCronJobs,
  hasArchived
}: SidebarSessionVisibilityInput): SidebarSessionVisibility {
  // Skeletons only during the genuine first load: still fetching, never
  // completed a load, and nothing to show yet. After the first success this is
  // permanently false, so a background refresh never re-flashes the section.
  const showSessionSkeletons = sessionsLoading && !sessionsInitialLoadComplete && sessionCount === 0

  // Mount the session area while first-load skeletons are up, or whenever ANY of
  // its sections has content. Recents is no longer the sole gate: pinned, cron,
  // and archived each keep the area alive on their own, so a recents-empty
  // account still sees them (they used to be hidden because the whole area was
  // gated on recents count alone). Each inner section keeps its own render
  // condition, so widening this only decides whether the area mounts at all.
  const showSessionSections =
    showSessionSkeletons || sessionCount > 0 || hasPinned || hasCronJobs || hasArchived

  // The recents list always renders inside the mounted area, so when it has no
  // rows it needs the right placeholder:
  //  - skeletons  → genuine first load
  //  - all-pinned → there ARE recent sessions but every one is pinned, so the
  //                 (unpinned) recents list is empty: "unpin a chat to show it"
  //  - empty      → there are genuinely no recent sessions; the area is only up
  //                 because pinned/cron/archived have content, so the all-pinned
  //                 copy would be wrong (nothing to unpin into recents)
  const recentsEmptyState: SidebarRecentsEmptyState = showSessionSkeletons
    ? 'skeletons'
    : sessionCount === 0
      ? 'empty'
      : 'all-pinned'

  return { recentsEmptyState, showSessionSections, showSessionSkeletons }
}

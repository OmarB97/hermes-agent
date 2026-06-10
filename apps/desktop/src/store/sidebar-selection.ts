import { atom } from 'nanostores'

/** Section identity for sidebar multi-select. A selection lives in exactly one
 * section at a time: every section shares the same bulk verbs (pin, archive,
 * delete, restore) but with section-specific meaning, so a cross-section
 * selection would have no honest answer for what its action bar applies. */
export type SidebarSectionKey = 'archived' | 'pinned' | 'results' | 'sessions' | `messaging:${string}`

export interface SidebarSelection {
  section: null | SidebarSectionKey
  /** Selected live session ids in click order; the LAST id is the range
   * anchor for the next shift-click. */
  ids: string[]
}

const EMPTY_SELECTION: SidebarSelection = { ids: [], section: null }

export const $sidebarSelection = atom<SidebarSelection>(EMPTY_SELECTION)

export function clearSidebarSelection() {
  const current = $sidebarSelection.get()

  if (current.section !== null || current.ids.length > 0) {
    $sidebarSelection.set(EMPTY_SELECTION)
  }
}

/** Toggle one row's membership. Selecting in a different section restarts the
 * selection there rather than silently merging two sections' rows. */
export function toggleSessionSelected(section: SidebarSectionKey, sessionId: string) {
  const current = $sidebarSelection.get()

  if (current.section !== section) {
    $sidebarSelection.set({ ids: [sessionId], section })

    return
  }

  const ids = current.ids.includes(sessionId)
    ? current.ids.filter(id => id !== sessionId)
    : [...current.ids, sessionId]

  $sidebarSelection.set(ids.length ? { ids, section } : EMPTY_SELECTION)
}

/** Shift-click: select the contiguous run between the current anchor (last
 * selected id) and `sessionId`, in the section's rendered order. Falls back to
 * a plain toggle when there is no usable anchor (fresh section, or the anchor
 * row left the list). The clicked id becomes the new anchor. */
export function rangeSelectSessions(section: SidebarSectionKey, sessionId: string, orderedIds: readonly string[]) {
  const current = $sidebarSelection.get()
  const anchor = current.section === section ? current.ids[current.ids.length - 1] : undefined
  const anchorIndex = anchor ? orderedIds.indexOf(anchor) : -1
  const targetIndex = orderedIds.indexOf(sessionId)

  if (anchorIndex < 0 || targetIndex < 0) {
    toggleSessionSelected(section, sessionId)

    return
  }

  const [from, to] = anchorIndex <= targetIndex ? [anchorIndex, targetIndex] : [targetIndex, anchorIndex]
  const range = orderedIds.slice(from, to + 1)
  const rangeSet = new Set(range)

  // Union with what's already selected; keep prior click order, append the
  // range in list order, and pin the clicked id to the end as the new anchor.
  const merged = [...current.ids.filter(id => !rangeSet.has(id)), ...range]

  $sidebarSelection.set({ ids: [...merged.filter(id => id !== sessionId), sessionId], section })
}

/** Drop selected ids that no longer exist in their section (archived away,
 * deleted elsewhere, paged out). Keeps the bar's count honest. */
export function pruneSidebarSelection(section: SidebarSectionKey, validIds: ReadonlySet<string>) {
  const current = $sidebarSelection.get()

  if (current.section !== section) {
    return
  }

  const ids = current.ids.filter(id => validIds.has(id))

  if (ids.length === current.ids.length) {
    return
  }

  $sidebarSelection.set(ids.length ? { ids, section: current.section } : EMPTY_SELECTION)
}

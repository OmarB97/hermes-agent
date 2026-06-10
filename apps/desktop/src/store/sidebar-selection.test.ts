import { beforeEach, describe, expect, it } from 'vitest'

import {
  $sidebarSelection,
  clearSidebarSelection,
  pruneSidebarSelection,
  rangeSelectSessions,
  toggleSessionSelected
} from './sidebar-selection'

const ORDER = ['a', 'b', 'c', 'd', 'e']

beforeEach(() => {
  $sidebarSelection.set({ ids: [], section: null })
})

describe('toggleSessionSelected', () => {
  it('starts a selection and toggles membership off back to empty', () => {
    toggleSessionSelected('sessions', 'a')
    expect($sidebarSelection.get()).toEqual({ ids: ['a'], section: 'sessions' })

    toggleSessionSelected('sessions', 'b')
    expect($sidebarSelection.get().ids).toEqual(['a', 'b'])

    toggleSessionSelected('sessions', 'a')
    expect($sidebarSelection.get().ids).toEqual(['b'])

    toggleSessionSelected('sessions', 'b')
    expect($sidebarSelection.get()).toEqual({ ids: [], section: null })
  })

  it('restarts in the new section instead of merging cross-section rows', () => {
    toggleSessionSelected('sessions', 'a')
    toggleSessionSelected('pinned', 'p1')

    expect($sidebarSelection.get()).toEqual({ ids: ['p1'], section: 'pinned' })
  })

  it('scopes messaging selections per platform', () => {
    toggleSessionSelected('messaging:telegram', 't1')
    toggleSessionSelected('messaging:discord', 'd1')

    expect($sidebarSelection.get()).toEqual({ ids: ['d1'], section: 'messaging:discord' })
  })
})

describe('rangeSelectSessions', () => {
  it('selects the contiguous run from the anchor, in either direction', () => {
    toggleSessionSelected('sessions', 'b')
    rangeSelectSessions('sessions', 'd', ORDER)
    expect($sidebarSelection.get().ids).toEqual(['b', 'c', 'd'])

    // New anchor is the clicked id; range back up the list unions in place.
    rangeSelectSessions('sessions', 'a', ORDER)
    expect(new Set($sidebarSelection.get().ids)).toEqual(new Set(['a', 'b', 'c', 'd']))
  })

  it('falls back to a plain toggle without a usable anchor', () => {
    rangeSelectSessions('sessions', 'c', ORDER)
    expect($sidebarSelection.get()).toEqual({ ids: ['c'], section: 'sessions' })

    // Anchor from another section doesn't leak into this one.
    toggleSessionSelected('pinned', 'p1')
    rangeSelectSessions('sessions', 'b', ORDER)
    expect($sidebarSelection.get()).toEqual({ ids: ['b'], section: 'sessions' })
  })

  it('keeps the clicked id as the next range anchor', () => {
    toggleSessionSelected('sessions', 'a')
    rangeSelectSessions('sessions', 'c', ORDER)
    rangeSelectSessions('sessions', 'e', ORDER)

    expect($sidebarSelection.get().ids).toEqual(['a', 'b', 'c', 'd', 'e'])
  })
})

describe('pruneSidebarSelection', () => {
  it('drops ids that left the section and clears when none remain', () => {
    toggleSessionSelected('sessions', 'a')
    toggleSessionSelected('sessions', 'b')

    pruneSidebarSelection('sessions', new Set(['b', 'c']))
    expect($sidebarSelection.get()).toEqual({ ids: ['b'], section: 'sessions' })

    pruneSidebarSelection('sessions', new Set(['c']))
    expect($sidebarSelection.get()).toEqual({ ids: [], section: null })
  })

  it('ignores prunes aimed at a different section', () => {
    toggleSessionSelected('archived', 'x')
    pruneSidebarSelection('sessions', new Set())

    expect($sidebarSelection.get()).toEqual({ ids: ['x'], section: 'archived' })
  })
})

describe('clearSidebarSelection', () => {
  it('resets to the empty selection', () => {
    toggleSessionSelected('archived', 'x')
    clearSidebarSelection()

    expect($sidebarSelection.get()).toEqual({ ids: [], section: null })
  })
})

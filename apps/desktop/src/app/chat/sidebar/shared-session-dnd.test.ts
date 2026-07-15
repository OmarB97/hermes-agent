import { describe, expect, it } from 'vitest'

import { moveSharedSessionDrag, type SharedSessionDrag } from './shared-session-dnd'

function drag(): SharedSessionDrag {
  return {
    activeId: 's2',
    from: 'sessions',
    pinned: ['p1', 'p2'],
    sessions: ['s1', 's2', 's3']
  }
}

describe('shared session drag working lists', () => {
  it('reorders organically within Sessions without an insertion-line model', () => {
    const next = moveSharedSessionDrag(drag(), 'sessions', 's3', true)

    expect(next.sessions).toEqual(['s1', 's3', 's2'])
    expect(next.pinned).toEqual(['p1', 'p2'])
  })

  it('keeps a Pinned preview order authoritative at release', () => {
    const current: SharedSessionDrag = {
      activeId: 'p2',
      from: 'pinned',
      pinned: ['p1', 'p2'],
      sessions: ['s1']
    }

    expect(moveSharedSessionDrag(current, 'pinned', 'p1', false).pinned).toEqual(['p2', 'p1'])
  })

  it('keeps one continuous drag coherent across Sessions and Pinned and back', () => {
    const inSessions = moveSharedSessionDrag(drag(), 'sessions', 's3', true)
    const inPinned = moveSharedSessionDrag(inSessions, 'pinned', 'p1', true)
    const movedWithinPinned = moveSharedSessionDrag(inPinned, 'pinned', 'p2', true)
    const backInSessions = moveSharedSessionDrag(movedWithinPinned, 'sessions', 's1', false)

    expect(inPinned).toMatchObject({
      pinned: ['p1', 's2', 'p2'],
      sessions: ['s1', 's3']
    })
    expect(movedWithinPinned).toMatchObject({
      pinned: ['p1', 'p2', 's2'],
      sessions: ['s1', 's3']
    })
    expect(backInSessions).toMatchObject({
      pinned: ['p1', 'p2'],
      sessions: ['s2', 's1', 's3']
    })
  })

  it('can enter an empty lane and remain movable', () => {
    const current = { ...drag(), pinned: [] }
    const inPinned = moveSharedSessionDrag(current, 'pinned', 'session-section:pinned', false)

    expect(inPinned.pinned).toEqual(['s2'])
    expect(inPinned.sessions).toEqual(['s1', 's3'])
  })
})

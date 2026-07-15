import { renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import {
  moveSharedSessionDrag,
  type SharedSessionDrag,
  sharedSessionReleaseAnchor,
  useSharedSessionDnd
} from './shared-session-dnd'

function drag(): SharedSessionDrag {
  return {
    activeId: 's2',
    from: 'sessions',
    pinned: ['p1', 'p2'],
    sessions: ['s1', 's2', 's3']
  }
}

describe('shared session drag working lists', () => {
  it('uses the physical release position even before the last preview frame renders', () => {
    const rows = [
      { height: 26, id: 'p1', top: 100 },
      { height: 26, id: 'p2', top: 127 },
      { height: 26, id: 's2', top: 154 }
    ]

    expect(sharedSessionReleaseAnchor(rows, 's2', 99)).toEqual({ after: false, overId: 'p1' })
    expect(sharedSessionReleaseAnchor(rows, 's2', 106)).toEqual({ after: false, overId: 'p1' })
    expect(sharedSessionReleaseAnchor(rows, 's2', 120)).toEqual({ after: true, overId: 'p1' })
  })

  it('updates the working preview on every drag move, not only when the anchor row changes', () => {
    const pinned = [{ id: 'p1' }, { id: 'p2' }] as SessionInfo[]
    const sessions = [{ id: 's1' }, { id: 's2' }, { id: 's3' }] as SessionInfo[]
    const sessionByAnyId = new Map([...pinned, ...sessions].map(session => [session.id, session]))

    const { result } = renderHook(() =>
      useSharedSessionDnd({
        enabled: true,
        pinnedSessionIds: pinned.map(session => session.id),
        pinnedSessions: pinned,
        sessionByAnyId,
        sessions
      })
    )

    expect(result.current.onDragMove).toBe(result.current.onDragOver)
  })

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

  it('keeps a cross-lane first-slot preview stable after the anchor index shifts', () => {
    const enteredAfterFirst = moveSharedSessionDrag(drag(), 'pinned', 'p1', true)
    const movedBeforeFirst = moveSharedSessionDrag(enteredAfterFirst, 'pinned', 'p1', false)
    const repeatedBeforeFirst = moveSharedSessionDrag(movedBeforeFirst, 'pinned', 'p1', false)

    expect(enteredAfterFirst.pinned).toEqual(['p1', 's2', 'p2'])
    expect(movedBeforeFirst.pinned).toEqual(['s2', 'p1', 'p2'])
    expect(repeatedBeforeFirst).toBe(movedBeforeFirst)
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

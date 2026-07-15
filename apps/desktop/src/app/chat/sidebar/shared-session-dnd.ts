import {
  closestCenter,
  type CollisionDetection,
  type DragMoveEvent,
  type DragOverEvent,
  type DragStartEvent,
  getFirstCollision,
  pointerWithin,
  rectIntersection
} from '@dnd-kit/core'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { SessionInfo } from '@/hermes'
import {
  $pinnedSessionIds,
  setSidebarPinsOpen,
  setSidebarRecentsOpen,
  setSidebarSessionOrderIds,
  setSidebarSessionOrderManual
} from '@/store/layout'
import { sessionPinId } from '@/store/session'

export type SharedSessionDndLane = 'pinned' | 'sessions'

const SECTION_PREFIX = 'session-section:'

export function sharedSessionSectionId(lane: SharedSessionDndLane): string {
  return `${SECTION_PREFIX}${lane}`
}

function parseSectionId(id: string): null | SharedSessionDndLane {
  if (!id.startsWith(SECTION_PREFIX)) {
    return null
  }

  const lane = id.slice(SECTION_PREFIX.length)

  return lane === 'pinned' || lane === 'sessions' ? lane : null
}

function sameIds(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((id, index) => id === b[index])
}

export interface SharedSessionDrag {
  activeId: string
  from: SharedSessionDndLane
  pinned: string[]
  sessions: string[]
}

export function moveSharedSessionDrag(
  current: SharedSessionDrag,
  targetLane: SharedSessionDndLane,
  overId: string,
  after: boolean
): SharedSessionDrag {
  const target = targetLane === 'pinned' ? current.pinned : current.sessions
  const other = (targetLane === 'pinned' ? current.sessions : current.pinned).filter(id => id !== current.activeId)
  const activeIndex = target.indexOf(current.activeId)
  const targetWithoutActive = target.filter(id => id !== current.activeId)
  const overIndex = targetWithoutActive.indexOf(overId)
  let nextTarget: string[]

  if (overIndex >= 0 && overId !== current.activeId) {
    // Always express the working order as an insertion relative to the row
    // under the pointer. Once a cross-lane preview has inserted the active row,
    // arrayMove's raw target index becomes unstable: excluding the active row
    // from collision detection makes the anchor's index change after every
    // preview shuffle, which can toggle a "before the first row" drop back to
    // one slot below it. Removing the active id first keeps before/after
    // semantics invariant across repeated hover frames and approach direction.
    const insertAt = overIndex + (after ? 1 : 0)

    nextTarget = [...targetWithoutActive.slice(0, insertAt), current.activeId, ...targetWithoutActive.slice(insertAt)]
  } else if (activeIndex >= 0) {
    nextTarget = target
  } else {
    nextTarget = [...target, current.activeId]
  }

  const nextPinned = targetLane === 'pinned' ? nextTarget : other
  const nextSessions = targetLane === 'sessions' ? nextTarget : other

  return sameIds(nextPinned, current.pinned) && sameIds(nextSessions, current.sessions)
    ? current
    : { ...current, pinned: nextPinned, sessions: nextSessions }
}

interface UseSharedSessionDndOptions {
  enabled: boolean
  pinnedSessionIds: string[]
  pinnedSessions: SessionInfo[]
  sessionByAnyId: Map<string, SessionInfo>
  sessions: SessionInfo[]
}

/**
 * One dnd-kit engine for both Pinned and Sessions. Keeping the active id in
 * exactly one working list avoids the dual native+dnd-kit activation race that
 * made packaged Electron builds appear draggable but never commit a drop.
 */
export function useSharedSessionDnd({
  enabled,
  pinnedSessionIds,
  pinnedSessions,
  sessionByAnyId,
  sessions
}: UseSharedSessionDndOptions) {
  const basePinnedIds = useMemo(() => pinnedSessions.map(session => session.id), [pinnedSessions])
  const baseSessionIds = useMemo(() => sessions.map(session => session.id), [sessions])
  const [drag, setDrag] = useState<null | SharedSessionDrag>(null)
  const dragRef = useRef<null | SharedSessionDrag>(null)
  const pointerYRef = useRef<null | number>(null)
  const releasedRef = useRef(false)

  useEffect(() => {
    const markReleased = () => {
      releasedRef.current = true
    }

    // Electron can cancel a dnd-kit drag after the physical release when the
    // animated preview has moved the last collision target away. Remember the
    // release in capture phase so that cancellation can still commit the exact
    // list the user was visibly holding.
    window.addEventListener('pointerup', markReleased, true)
    window.addEventListener('mouseup', markReleased, true)
    window.addEventListener('touchend', markReleased, true)

    return () => {
      window.removeEventListener('pointerup', markReleased, true)
      window.removeEventListener('mouseup', markReleased, true)
      window.removeEventListener('touchend', markReleased, true)
    }
  }, [])

  const commitDrag = useCallback((next: null | SharedSessionDrag) => {
    dragRef.current = next
    setDrag(next)
  }, [])

  const sessionForId = useCallback(
    (id: string) => sessionByAnyId.get(id) ?? sessions.find(session => session.id === id),
    [sessionByAnyId, sessions]
  )

  const mapIdsToSessions = useCallback(
    (ids: readonly string[]) => ids.map(sessionForId).filter((session): session is SessionInfo => Boolean(session)),
    [sessionForId]
  )

  const effectivePinnedSessions = useMemo(
    () => (drag ? mapIdsToSessions(drag.pinned) : pinnedSessions),
    [drag, mapIdsToSessions, pinnedSessions]
  )

  const effectiveSessions = useMemo(
    () => (drag ? mapIdsToSessions(drag.sessions) : sessions),
    [drag, mapIdsToSessions, sessions]
  )

  const containerForId = useCallback(
    (id: string, current: null | SharedSessionDrag): null | SharedSessionDndLane => {
      if (current?.pinned.includes(id) || (!current && basePinnedIds.includes(id))) {
        return 'pinned'
      }

      if (current?.sessions.includes(id) || (!current && baseSessionIds.includes(id))) {
        return 'sessions'
      }

      return null
    },
    [basePinnedIds, baseSessionIds]
  )

  const collisionDetection = useCallback<CollisionDetection>(
    args => {
      if (args.pointerCoordinates) {
        pointerYRef.current = args.pointerCoordinates.y
      }

      // The active sortable remains a droppable container and moves under the
      // pointer. Never let it win its own collision: doing so reports `over`
      // as the dragged row forever, so a same-lane move can animate without
      // ever recognizing the sibling underneath (and then snaps back).
      const activeId = String(args.active.id)
      const pointerHits = pointerWithin(args).filter(hit => String(hit.id) !== activeId)
      const rowUnderPointer = pointerHits.find(hit => !parseSectionId(String(hit.id)))

      if (rowUnderPointer) {
        return [{ id: rowUnderPointer.id }]
      }

      const intersections = (pointerHits.length ? pointerHits : rectIntersection(args)).filter(
        hit => String(hit.id) !== activeId
      )

      let overId = getFirstCollision(intersections, 'id')

      if (overId == null) {
        return []
      }

      const lane = parseSectionId(String(overId))

      if (lane) {
        const ids = lane === 'pinned' ? (drag?.pinned ?? basePinnedIds) : (drag?.sessions ?? baseSessionIds)
        const idSet = new Set(ids)

        const rows = args.droppableContainers.filter(
          container => String(container.id) !== activeId && idSet.has(String(container.id))
        )

        if (rows.length) {
          const closest = closestCenter({ ...args, droppableContainers: rows })

          if (closest.length) {
            overId = closest[0].id
          }
        }
      }

      return [{ id: overId }]
    },
    [basePinnedIds, baseSessionIds, drag]
  )

  const onDragStart = useCallback(
    (event: DragStartEvent) => {
      if (!enabled) {
        return
      }

      const activeId = String(event.active.id)
      const from = containerForId(activeId, null)

      if (!from) {
        return
      }

      releasedRef.current = false
      commitDrag({ activeId, from, pinned: [...basePinnedIds], sessions: [...baseSessionIds] })
    },
    [basePinnedIds, baseSessionIds, commitDrag, containerForId, enabled]
  )

  const updateDragPreview = useCallback(
    (event: DragMoveEvent | DragOverEvent) => {
      const current = dragRef.current

      if (!current || !event.over) {
        return
      }

      const overId = String(event.over.id)
      const targetLane = parseSectionId(overId) ?? containerForId(overId, current)

      if (!targetLane) {
        return
      }

      const pointerY = pointerYRef.current
      const rect = event.over.rect
      const after = pointerY != null && pointerY > rect.top + rect.height / 2
      const next = moveSharedSessionDrag(current, targetLane, overId, after)

      if (next === current) {
        return
      }

      commitDrag(next)
    },
    [commitDrag, containerForId]
  )

  const commitSettledDrag = useCallback(() => {
    const settled = dragRef.current

    commitDrag(null)

    if (!settled) {
      return
    }

    // The rendered working lists are the authoritative drop preview. At
    // release dnd-kit can report `over: null` because the animated preview
    // moved the row out from under the pointer. Requiring a final `over`
    // recreates the historical snap-back bug, so persist what the user was
    // visibly holding instead of recomputing from the release frame.
    // Translate visible live ids back to durable lineage-root ids while
    // preserving unloaded pins in their original slots.
    const reorderedVisiblePins = settled.pinned.map(id => {
      const session = sessionForId(id)

      return session ? sessionPinId(session) : id
    })

    const baseVisiblePins = new Set(pinnedSessions.map(sessionPinId))
    const nextPinnedIds: string[] = []
    let visibleIndex = 0

    for (const pinId of pinnedSessionIds) {
      if (baseVisiblePins.has(pinId)) {
        const replacement = reorderedVisiblePins[visibleIndex]

        if (replacement) {
          nextPinnedIds.push(replacement)
          visibleIndex += 1
        }
      } else {
        nextPinnedIds.push(pinId)
      }
    }

    nextPinnedIds.push(...reorderedVisiblePins.slice(visibleIndex))

    const dedupedPinnedIds = nextPinnedIds.filter((id, index, all) => all.indexOf(id) === index)

    if (!sameIds(dedupedPinnedIds, pinnedSessionIds)) {
      $pinnedSessionIds.set(dedupedPinnedIds)
    }

    if (!sameIds(settled.sessions, baseSessionIds)) {
      setSidebarSessionOrderManual(true)
      setSidebarSessionOrderIds(settled.sessions)
    }

    if (settled.pinned.includes(settled.activeId)) {
      setSidebarPinsOpen(true)
    } else if (settled.sessions.includes(settled.activeId)) {
      setSidebarRecentsOpen(true)
    }
  }, [baseSessionIds, commitDrag, pinnedSessionIds, pinnedSessions, sessionForId])

  const onDragCancel = useCallback(() => {
    if (releasedRef.current) {
      commitSettledDrag()

      return
    }

    commitDrag(null)
  }, [commitDrag, commitSettledDrag])

  const onDragEnd = useCallback(() => commitSettledDrag(), [commitSettledDrag])

  return {
    activeId: drag?.activeId ?? null,
    collisionDetection,
    effectivePinnedSessions,
    effectiveSessions,
    onDragCancel,
    onDragEnd,
    onDragMove: updateDragPreview,
    onDragOver: updateDragPreview,
    onDragStart
  }
}

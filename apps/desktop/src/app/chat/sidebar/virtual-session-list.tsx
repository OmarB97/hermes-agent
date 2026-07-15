import { useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useVirtualizer } from '@tanstack/react-virtual'
import { type FC, useCallback, useRef } from 'react'

import type { SessionDragPayload } from '@/app/chat/composer/inline-refs'
import type { SessionInfo } from '@/hermes'
import { type SidebarSessionEntry } from '@/lib/session-branch-tree'
import { cn } from '@/lib/utils'
import { sessionPinId } from '@/store/session'
import type { SidebarSectionKey } from '@/store/sidebar-selection'

import { SidebarSessionRow } from './session-row'

interface SessionRowCommonProps {
  branchStem?: string
  isPinned: boolean
  isSelected: boolean
  isWorking: boolean
  onArchive: () => void
  onBranch?: () => void
  onDelete: () => void
  onPin: () => void
  onResume: () => void
  onSessionDragEnd?: () => void
  onSessionDragStart?: (payload: SessionDragPayload) => void
  reorderable?: boolean
  selectable?: boolean
  selectionActive?: boolean
  checked?: boolean
  onToggleSelect?: (mode: 'range' | 'single') => void
  bulkSelectedSessionIds?: readonly string[]
  onArchiveSelectedSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onDeleteSelectedSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onHaltSelectedSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onPromptSelectedSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onSteerSelectedSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
}

interface VirtualSessionListProps {
  activeSessionId: null | string
  className?: string
  entries: SidebarSessionEntry[]
  onArchiveSession: (sessionId: string) => void
  onArchiveSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onBranchSession?: (sessionId: string, profile?: string) => void
  onDeleteSession: (sessionId: string) => void
  onDeleteSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onHaltSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onPromptSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onResumeSession: (sessionId: string) => void
  onSessionDragEnd?: () => void
  onSessionDragStart?: (payload: SessionDragPayload) => void
  onTogglePin: (sessionId: string) => void
  onToggleSelect?: (sessionId: string, mode: 'range' | 'single') => void
  pinned: boolean
  sectionKey?: SidebarSectionKey
  selectable?: boolean
  selectedIds?: ReadonlySet<string>
  selectedSessionIds?: readonly string[]
  selectionActive?: boolean
  sortable: boolean
  onSteerSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  workingSessionIdSet: Set<string>
}

const ROW_ESTIMATE_PX = 28
const OVERSCAN_ROWS = 12

export const VirtualSessionList: FC<VirtualSessionListProps> = ({
  activeSessionId,
  className,
  entries,
  onArchiveSession,
  onArchiveSessions,
  onBranchSession,
  onDeleteSession,
  onDeleteSessions,
  onHaltSessions,
  onPromptSessions,
  onResumeSession,
  onSessionDragEnd,
  onSessionDragStart,
  onTogglePin,
  onToggleSelect,
  pinned,
  sectionKey,
  selectable = false,
  selectedIds,
  selectedSessionIds,
  selectionActive = false,
  sortable,
  onSteerSessions,
  workingSessionIdSet
}) => {
  const scrollerRef = useRef<HTMLDivElement | null>(null)

  const virtualizer = useVirtualizer({
    count: entries.length,
    estimateSize: () => ROW_ESTIMATE_PX,
    getItemKey: index => entries[index]?.session.id ?? index,
    getScrollElement: () => scrollerRef.current,
    // jsdom-friendly default; the real rect takes over on first observe.
    initialRect: { height: 600, width: 240 },
    overscan: OVERSCAN_ROWS
  })

  const virtualItems = virtualizer.getVirtualItems()
  const totalSize = virtualizer.getTotalSize()
  const paddingTop = virtualItems[0]?.start ?? 0
  const paddingBottom = Math.max(0, totalSize - (virtualItems[virtualItems.length - 1]?.end ?? 0))

  const rows = virtualItems.map(virtualItem => {
    const entry = entries[virtualItem.index]

    if (!entry) {
      return null
    }

    const { branchStem, session } = entry
    const reorderable = sortable && !branchStem

    const commonProps: SessionRowCommonProps = {
      bulkSelectedSessionIds: selectedIds?.has(session.id) ? selectedSessionIds : undefined,
      checked: selectedIds?.has(session.id) ?? false,
      branchStem,
      isPinned: pinned,
      isSelected: session.id === activeSessionId,
      isWorking: workingSessionIdSet.has(session.id),
      onArchive: () => onArchiveSession(session.id),
      onArchiveSelectedSessions: onArchiveSessions,
      onBranch: onBranchSession ? () => onBranchSession(session.id, session.profile) : undefined,
      onDelete: () => onDeleteSession(session.id),
      onDeleteSelectedSessions: onDeleteSessions,
      onHaltSelectedSessions: onHaltSessions,
      onPin: () => onTogglePin(sessionPinId(session)),
      onPromptSelectedSessions: onPromptSessions,
      onResume: () => onResumeSession(session.id),
      onSessionDragEnd,
      onSessionDragStart,
      onSteerSelectedSessions: onSteerSessions,
      onToggleSelect: sectionKey && onToggleSelect ? mode => onToggleSelect(session.id, mode) : undefined,
      reorderable,
      selectable,
      selectionActive
    }

    return reorderable ? (
      <VirtualSortableRow
        index={virtualItem.index}
        key={session.id}
        measureRef={virtualizer.measureElement}
        rowProps={commonProps}
        session={session}
      />
    ) : (
      <SidebarSessionRow
        {...commonProps}
        data-index={virtualItem.index}
        key={session.id}
        ref={virtualizer.measureElement}
        session={session}
      />
    )
  })

  // When sortable, the caller wraps this in a ReorderableList that owns the
  // DndContext + SortableContext (keyed on the same ids); the virtualized rows
  // just consume that context via useSortable.
  return (
    <div
      className={cn('relative min-h-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain', className)}
      ref={scrollerRef}
    >
      <div className="grid gap-px" style={{ paddingBottom: `${paddingBottom}px`, paddingTop: `${paddingTop}px` }}>
        {rows}
      </div>
    </div>
  )
}

interface VirtualSortableRowProps {
  index: number
  measureRef: (node: Element | null) => void
  rowProps: SessionRowCommonProps
  session: SessionInfo
}

function VirtualSortableRow({ index, measureRef, rowProps, session }: VirtualSortableRowProps) {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ id: session.id })

  // Merge dnd-kit's setNodeRef with the virtualizer's measureElement so
  // the row participates in both DnD hit-testing and TanStack height
  // measurement.
  const refMerged = useCallback(
    (node: HTMLDivElement | null) => {
      setNodeRef(node)
      measureRef(node)
    },
    [measureRef, setNodeRef]
  )

  return (
    <SidebarSessionRow
      {...rowProps}
      data-index={index}
      dragging={isDragging}
      dragHandleProps={{ ...attributes, ...listeners }}
      ref={refMerged}
      reorderable
      session={session}
      style={{ transform: CSS.Transform.toString(transform), transition }}
    />
  )
}

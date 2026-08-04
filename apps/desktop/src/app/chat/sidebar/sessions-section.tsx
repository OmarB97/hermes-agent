import { useDroppable, type useSensors } from '@dnd-kit/core'
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable'
import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo } from 'react'

import type { SessionDragPayload } from '@/app/chat/composer/inline-refs'
import { SidebarPanelLabel } from '@/app/shell/sidebar-label'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { flattenSessionsWithBranches } from '@/lib/session-branch-tree'
import { groupEntriesByRecency, type SidebarListRow, toSessionRows } from '@/lib/session-date-groups'
import { sessionBucketLabel } from '@/lib/time'
import { cn } from '@/lib/utils'
import { sessionPinId } from '@/store/session'
import {
  $sidebarSelection,
  pruneSidebarSelection,
  rangeSelectSessions,
  type SidebarSectionKey,
  toggleSessionSelected
} from '@/store/sidebar-selection'

import { SidebarDateDivider, SidebarSectionMeta } from './chrome'
import {
  EnteredProjectContent,
  ProjectOverviewRow,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  SidebarWorkspaceGroup,
  type SidebarWorkspaceTree
} from './projects'
import { ReorderableList, useSortableBindings } from './reorderable-list'
import { SidebarSessionSkeletons } from './section-states'
import { SidebarSessionRow } from './session-row'
import { VirtualSessionList } from './virtual-session-list'

export const VIRTUALIZE_THRESHOLD = 25

interface SidebarSectionHeaderProps {
  label: string
  open: boolean
  onToggle: () => void
  action?: React.ReactNode
  meta?: React.ReactNode
  icon?: React.ReactNode
  // When false the section can't be collapsed: the label renders static (no
  // toggle, no caret) and the section is always open. Used for the single-
  // project view, where collapsing one project makes no sense.
  collapsible?: boolean
}

function SidebarSectionHeader({
  label,
  open,
  onToggle,
  action,
  meta,
  icon,
  collapsible = true
}: SidebarSectionHeaderProps) {
  const labelBody = (
    <>
      {icon}
      <SidebarPanelLabel>{label}</SidebarPanelLabel>
      {meta && <SidebarSectionMeta>{meta}</SidebarSectionMeta>}
    </>
  )

  return (
    <div className="group/section flex shrink-0 items-center justify-between gap-1 pb-1 pt-1.5">
      {collapsible ? (
        <button
          // min-w-0 lets the label truncate at narrow sidebar widths instead of
          // pushing the header's trailing action icons out of view.
          className="group/section-label flex w-fit min-w-0 items-center gap-1 bg-transparent text-left leading-none"
          onClick={onToggle}
          type="button"
        >
          {labelBody}
          <DisclosureCaret
            className="text-(--ui-text-tertiary) opacity-0 transition group-hover/section-label:opacity-100"
            open={open}
          />
        </button>
      ) : (
        <div className="flex w-fit min-w-0 items-center gap-1 leading-none">{labelBody}</div>
      )}
      {action}
    </div>
  )
}

interface SidebarSessionsSectionProps {
  label: string
  open: boolean
  onToggle: () => void
  sessions: SessionInfo[]
  activeSessionId: null | string
  workingSessionIdSet: Set<string>
  onResumeSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onDeleteSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onArchiveSession: (sessionId: string) => void
  onArchiveSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onHaltSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onPromptSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onSteerSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onBranchSession?: (sessionId: string, profile?: string) => void
  onTogglePin: (sessionId: string) => void
  onNewSessionInWorkspace?: (path: null | string) => void
  pinned: boolean
  sectionKey?: SidebarSectionKey
  rootClassName?: string
  contentClassName?: string
  emptyState: React.ReactNode
  forceEmptyState?: boolean
  headerAction?: React.ReactNode
  footer?: React.ReactNode
  groups?: SidebarSessionGroup[]
  tree?: SidebarWorkspaceTree[]
  // Project overview: when present, render a drill-in list of project rows
  // instead of sessions. Clicking a row enters that project (onEnterProject),
  // which then passes `projectContent` on the next render. Takes precedence
  // over `tree` / `groups`.
  projectOverview?: SidebarProjectTree[]
  // Per-project preview rows (from the backend tree), keyed by project path.
  projectOverviewPreviews?: Record<string, SessionInfo[]>
  // True while the backend project tree is loading (overview skeleton).
  projectsLoading?: boolean
  onEnterProject?: (id: string) => void
  // The entered project's flattened content: main-checkout sessions render
  // directly (no redundant repo/branch header); only linked worktrees nest.
  projectContent?: SidebarProjectTree
  // Live git lanes (`git worktree list`) for repos in the entered project —
  // a VISUAL enhancer only (empty lanes), never session membership.
  projectRepoWorktrees?: Record<string, HermesGitWorktree[]>
  // Live session cache used for optimistic placement inside entered-project lanes.
  liveSessions?: SessionInfo[]
  // Client-side optimistic eviction layer (deleted/archived ids).
  removedSessionIds?: ReadonlySet<string>
  activeProjectId?: null | string
  labelMeta?: React.ReactNode
  labelIcon?: React.ReactNode
  // When false the section header is static (no caret/toggle) and always open.
  collapsible?: boolean
  sortable?: boolean
  // The flat session list is the only hand-reorderable surface (grouped/project
  // views sort deterministically), so it owns the one ReorderableList.
  onReorderSessions?: (ids: string[]) => void
  // Drag-to-reorder for the project overview list (top-level projects).
  onReorderProjects?: (ids: string[]) => void
  // Rendered atop the entered-project body (a "back to overview" row).
  projectBackRow?: React.ReactNode
  dndSensors?: ReturnType<typeof useSensors>
  // Tag every row with its owning profile. Set on the flat cross-profile
  // lists (Pinned / search results) in the All-profiles view, where no group
  // header communicates ownership (#66003).
  showProfileTags?: boolean
  // Pinned + Sessions share one parent DndContext in the flat view. Each
  // section contributes its own droppable bucket + SortableContext.
  sharedSessionDnd?: boolean
  sessionDndId?: string
  draggingSessionId?: null | string
  disableVirtualization?: boolean
  // Native session-drag (drag-to-pin/unpin/reorder): the owner hands each row
  // body's drag start/end down so it can drive drop-zone previews in flight.
  onSessionDragEnd?: () => void
  onSessionDragStart?: (payload: SessionDragPayload) => void
  /** True while an acceptable row drag hovers this section (lights the frame). */
  dropActive?: boolean
  /** Native session-drag drop target handlers, spread onto the section frame. */
  dropHandlers?: Pick<React.DOMAttributes<HTMLDivElement>, 'onDragEnter' | 'onDragLeave' | 'onDragOver' | 'onDrop'>
  // Insert "Yesterday" / "Last week" date dividers into the chronological
  // session list (flat recents + entered-project lanes). Off for hand-ordered
  // lists, pinned, messaging groups, and the project overview, where the order
  // isn't strictly by recency so a date bucket would be misleading.
  dateGrouped?: boolean
}

export function SidebarSessionsSection({
  label,
  open,
  onToggle,
  sessions,
  activeSessionId,
  workingSessionIdSet,
  onResumeSession,
  onDeleteSession,
  onDeleteSessions,
  onArchiveSession,
  onArchiveSessions,
  onHaltSessions,
  onPromptSessions,
  onSteerSessions,
  onBranchSession,
  onTogglePin,
  onNewSessionInWorkspace,
  pinned,
  sectionKey,
  rootClassName,
  contentClassName,
  emptyState,
  forceEmptyState = false,
  headerAction,
  footer,
  groups,
  projectOverview,
  projectOverviewPreviews,
  projectsLoading = false,
  onEnterProject,
  projectContent,
  projectRepoWorktrees,
  liveSessions,
  removedSessionIds,
  activeProjectId,
  labelMeta,
  labelIcon,
  collapsible = true,
  sortable = false,
  onReorderSessions,
  onReorderProjects,
  projectBackRow,
  dndSensors,
  showProfileTags = false,
  sharedSessionDnd = false,
  sessionDndId,
  draggingSessionId,
  disableVirtualization = false,
  onSessionDragEnd,
  onSessionDragStart,
  dropActive = false,
  dropHandlers,
  dateGrouped = false
}: SidebarSessionsSectionProps) {
  const { isOver: sharedDropActive, setNodeRef: setSharedDropRef } = useDroppable({
    disabled: !sharedSessionDnd || !sessionDndId,
    id: sessionDndId ?? `disabled:${label}`
  })

  const { t } = useI18n()
  const dividerLabels = t.sidebar.dateDivider
  const sectionOpen = collapsible ? open : true
  const selection = useStore($sidebarSelection)
  const selectable = Boolean(sectionKey)
  const selectionActive = selectable && selection.section === sectionKey && selection.ids.length > 0
  const hasGroupedSessions = Boolean(groups?.some(group => group.sessions.length > 0))
  // A defined project list is itself content (even an empty project should
  // render as a drill-in row so the user can see it exists).
  const hasProjectOverview = Boolean(projectOverview?.length)

  // Lanes count as content even with no rows left in them: the backend only
  // emits a lane that has sessions, so a lane surviving with zero rows means
  // they were filtered out (pinned) — the branch is real and must still render.
  // A genuinely empty project has no lanes at all and keeps its empty state.
  const hasProjectContent = Boolean(
    projectContent && (projectContent.sessionCount > 0 || projectContent.repos.some(repo => repo.groups.length > 0))
  )

  const showEmptyState =
    forceEmptyState || (!hasGroupedSessions && !hasProjectOverview && !hasProjectContent && sessions.length === 0)

  // The flat recents/pinned list is the only place sessions reorder by hand;
  // grouped/tree views always sort by creation date and never drag. Under the
  // shared Pinned+Sessions DndContext the lane is draggable without owning an
  // `onReorderSessions` of its own.
  const handOrdered = sortable && (sharedSessionDnd || !!onReorderSessions)
  const sessionsDraggable = handOrdered && !selectionActive
  // Pinned and hand-dragged Sessions already arrive in their authoritative
  // persisted order. Branch flattening may nest children, but it must not
  // silently sort the top-level rows by last_active and undo a manual drop —
  // for pins that let a finishing turn float background tasks over the user's
  // fixed ranking. A date-grouped list is the exception: its dividers are only
  // truthful when the roots really are in recency order.
  const preserveInputOrder = pinned || (handOrdered && !dateGrouped)

  const displayEntries = useMemo(
    () => flattenSessionsWithBranches(sessions, { preserveOrder: preserveInputOrder }),
    [preserveInputOrder, sessions]
  )

  const orderedIds = useMemo(() => displayEntries.map(entry => entry.session.id), [displayEntries])

  const selectedIdSet = useMemo(
    () => (selectionActive ? new Set(selection.ids) : undefined),
    [selection.ids, selectionActive]
  )

  useEffect(() => {
    if (sectionKey) {
      pruneSidebarSelection(
        sectionKey,
        displayEntries.map(entry => entry.session)
      )
    }
  }, [displayEntries, sectionKey])

  const handleToggleSelect = useCallback(
    (sessionId: string, mode: 'range' | 'single') => {
      if (!sectionKey) {
        return
      }

      if (mode === 'range') {
        rangeSelectSessions(sectionKey, sessionId, orderedIds, activeSessionId)
      } else {
        toggleSessionSelected(sectionKey, sessionId)
      }
    },
    [activeSessionId, orderedIds, sectionKey]
  )

  const renderRow = useCallback(
    (session: SessionInfo, draggable: boolean, branchStem?: string) => {
      const checked = selectedIdSet?.has(session.id) ?? false

      const rowProps = {
        bulkSelectedSessionIds: checked && selection.ids.length > 1 ? selection.ids : undefined,
        branchStem,
        checked,
        isPinned: pinned,
        isSelected: session.id === activeSessionId,
        isWorking: workingSessionIdSet.has(session.id),
        dragging: session.id === draggingSessionId,
        onArchive: () => onArchiveSession(session.id),
        onArchiveSelectedSessions: onArchiveSessions,
        onBranch: onBranchSession ? () => onBranchSession(session.id, session.profile) : undefined,
        onDelete: () => onDeleteSession(session.id),
        onDeleteSelectedSessions: onDeleteSessions,
        onHaltSelectedSessions: onHaltSessions,
        onPin: () => onTogglePin(sessionPinId(session)),
        onPromptSelectedSessions: onPromptSessions,
        onResume: () => onResumeSession(session.id),
        onSteerSelectedSessions: onSteerSessions,
        onToggleSelect: sectionKey ? (mode: 'range' | 'single') => handleToggleSelect(session.id, mode) : undefined,
        reorderable: draggable && !branchStem,
        session,
        selectable,
        selectionActive,
        showProfile: showProfileTags
      }

      return draggable && !branchStem ? (
        <SortableSidebarSessionRow key={session.id} previewOwnsLayout={Boolean(sharedSessionDnd)} {...rowProps} />
      ) : (
        <SidebarSessionRow key={session.id} {...rowProps} />
      )
    },
    [
      activeSessionId,
      draggingSessionId,
      handleToggleSelect,
      onArchiveSession,
      onArchiveSessions,
      onBranchSession,
      onDeleteSession,
      onDeleteSessions,
      onHaltSessions,
      onPromptSessions,
      onResumeSession,
      onSteerSessions,
      onTogglePin,
      pinned,
      sectionKey,
      selectable,
      selectedIdSet,
      selection.ids,
      selectionActive,
      sharedSessionDnd,
      showProfileTags,
      workingSessionIdSet
    ]
  )

  // A single flat/virtual/lane list row — either a date divider or a session.
  const renderListRow = useCallback(
    (row: SidebarListRow, draggable: boolean) =>
      row.kind === 'divider' ? (
        <SidebarDateDivider key={row.key} label={sessionBucketLabel(row.bucket, dividerLabels)} />
      ) : (
        renderRow(row.entry.session, draggable, row.entry.branchStem)
      ),
    [dividerLabels, renderRow]
  )

  // Sessions inside repos/worktrees are date-ordered and static.
  const renderRows = useCallback(
    (items: SessionInfo[]) =>
      flattenSessionsWithBranches(items).map(({ branchStem, session }) => renderRow(session, false, branchStem)),
    [renderRow]
  )

  // Same as `renderRows`, but with date dividers folded in — used for
  // entered-project lanes so a lane spanning multiple days reads
  // chronologically, matching the flat recents list.
  const renderRowsDated = useCallback(
    (items: SessionInfo[]) => {
      const entries = flattenSessionsWithBranches(items)

      return (dateGrouped ? groupEntriesByRecency(entries) : toSessionRows(entries)).map(row => renderListRow(row, false))
    },
    [dateGrouped, renderListRow]
  )

  // Flat recents as list rows: grouped by recency when enabled, plain otherwise.
  const flatRows: SidebarListRow[] = useMemo(
    () => (dateGrouped ? groupEntriesByRecency(displayEntries) : toSessionRows(displayEntries)),
    [dateGrouped, displayEntries]
  )

  const flatVirtualized =
    !disableVirtualization &&
    !showEmptyState &&
    !groups?.length &&
    !projectOverview?.length &&
    !projectContent &&
    sessions.length >= VIRTUALIZE_THRESHOLD

  // First paint into the grouped view (e.g. the app restoring the Projects tab)
  // has flat recents in `sessions` but no tree yet. Show skeletons rather than
  // flashing the flat session list until the overview/content/groups resolve. A
  // background refresh keeps the prior tree, so this only fires when empty.
  const showProjectsSkeleton =
    projectsLoading && !hasProjectOverview && !hasProjectContent && !projectContent && !groups?.length

  let inner: React.ReactNode

  if (showProjectsSkeleton) {
    inner = <SidebarSessionSkeletons />
  } else if (projectContent) {
    // Entered a project: the back row is always present, then either the
    // (overlay-aware) content or a clean empty state — never a bare spinner or a
    // blank pane while lanes hydrate.
    inner = (
      <>
        {projectBackRow}
        {hasProjectContent ? (
          <EnteredProjectContent
            liveSessions={liveSessions}
            onNewSession={onNewSessionInWorkspace}
            project={projectContent}
            removedSessionIds={removedSessionIds}
            renderRows={renderRowsDated}
            repoWorktrees={projectRepoWorktrees}
          />
        ) : (
          emptyState
        )}
      </>
    )
  } else if (showEmptyState) {
    inner = emptyState
  } else if (projectOverview?.length) {
    // The model is already ordered (Home leads; then the default sort groups
    // explicit-before-auto, with a manual drag-order winning when present).
    // Render in that order and make rows drag-to-reorder when a handler is
    // wired — Home stays outside the sortable list, it's a fixture.
    const home = projectOverview[0]?.isNoProject ? projectOverview[0] : undefined
    const sortableProjects = home ? projectOverview.slice(1) : projectOverview
    const projectsDraggable = sortableProjects.length > 1 && !!onReorderProjects
    const Row = projectsDraggable ? SortableProjectOverviewRow : ProjectOverviewRow

    const projectRow = (project: SidebarProjectTree, Component: typeof ProjectOverviewRow) => (
      <Component
        activeProjectId={activeProjectId}
        key={project.id}
        onEnter={onEnterProject}
        onNewSession={onNewSessionInWorkspace}
        previewSessions={projectOverviewPreviews?.[project.id]}
        project={project}
        renderRows={renderRows}
      />
    )

    const rows = sortableProjects.map(project => projectRow(project, Row))

    inner = (
      <>
        {home && projectRow(home, ProjectOverviewRow)}
        {projectsDraggable && onReorderProjects ? (
          <ReorderableList
            ids={sortableProjects.map(project => project.id)}
            onReorder={onReorderProjects}
            sensors={dndSensors}
          >
            {rows}
          </ReorderableList>
        ) : (
          rows
        )}
      </>
    )
  } else if (groups?.length) {
    // Profile/source groups never reorder; render them flat with static rows.
    inner = groups.map(group => (
      <SidebarWorkspaceGroup
        group={group}
        key={group.id}
        onNewSession={onNewSessionInWorkspace}
        renderRows={renderRows}
      />
    ))
  } else if (flatVirtualized) {
    const virtual = (
      <VirtualSessionList
        activeSessionId={activeSessionId}
        className={contentClassName}
        onArchiveSession={onArchiveSession}
        onArchiveSessions={onArchiveSessions}
        onBranchSession={onBranchSession}
        onDeleteSession={onDeleteSession}
        onDeleteSessions={onDeleteSessions}
        onHaltSessions={onHaltSessions}
        onPromptSessions={onPromptSessions}
        onResumeSession={onResumeSession}
        onSessionDragEnd={onSessionDragEnd}
        onSessionDragStart={onSessionDragStart}
        onSteerSessions={onSteerSessions}
        onTogglePin={onTogglePin}
        onToggleSelect={sectionKey ? handleToggleSelect : undefined}
        pinned={pinned}
        rows={flatRows}
        sectionKey={sectionKey}
        selectable={selectable}
        selectedIds={selectedIdSet}
        selectedSessionIds={selection.ids}
        selectionActive={selectionActive}
        showProfileTags={showProfileTags}
        sortable={sessionsDraggable}
        workingSessionIdSet={workingSessionIdSet}
      />
    )

    inner =
      sharedSessionDnd && sessionsDraggable ? (
        <SortableContext items={sessions.map(s => s.id)} strategy={verticalListSortingStrategy}>
          {virtual}
        </SortableContext>
      ) : sessionsDraggable && onReorderSessions ? (
        <ReorderableList ids={sessions.map(s => s.id)} onReorder={onReorderSessions} sensors={dndSensors}>
          {virtual}
        </ReorderableList>
      ) : (
        virtual
      )
  } else if (sharedSessionDnd && sessionsDraggable) {
    inner = (
      <SortableContext items={sessions.map(s => s.id)} strategy={verticalListSortingStrategy}>
        {flatRows.map(row => renderListRow(row, true))}
      </SortableContext>
    )
  } else if (sessionsDraggable && onReorderSessions) {
    inner = (
      <ReorderableList ids={sessions.map(s => s.id)} onReorder={onReorderSessions} sensors={dndSensors}>
        {flatRows.map(row => renderListRow(row, true))}
      </ReorderableList>
    )
  } else {
    inner = flatRows.map(row => renderListRow(row, false))
  }

  // The virtualizer owns its own scroller, so suppress the wrapper's overflow
  // to avoid a double scroll container.
  const resolvedContentClassName = cn(contentClassName, flatVirtualized && 'overflow-y-visible')

  return (
    <SidebarGroup
      className={cn(
        rootClassName,
        // Light the whole section (header included — drops there count too, even
        // collapsed) while an acceptable row drag hovers it.
        (dropActive || sharedDropActive) &&
          'rounded-lg bg-(--ui-control-hover-background) ring-1 ring-inset ring-(--ui-stroke-tertiary)'
      )}
      data-session-dnd-lane={sharedSessionDnd ? (pinned ? 'pinned' : 'sessions') : undefined}
      data-sidebar-session-section={sectionKey}
      ref={sharedSessionDnd ? setSharedDropRef : undefined}
      {...dropHandlers}
    >
      <SidebarSectionHeader
        action={headerAction}
        collapsible={collapsible}
        icon={labelIcon}
        label={label}
        meta={labelMeta}
        onToggle={onToggle}
        open={sectionOpen}
      />
      {sectionOpen && (
        <SidebarGroupContent className={resolvedContentClassName}>
          {inner}
          {footer}
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}

interface SortableSessionRowProps {
  session: SessionInfo
  isPinned: boolean
  isSelected: boolean
  isWorking: boolean
  onArchive: () => void
  onDelete: () => void
  onPin: () => void
  onResume: () => void
  onSessionDragEnd?: () => void
  onSessionDragStart?: (payload: SessionDragPayload) => void
  previewOwnsLayout?: boolean
}

function SortableSidebarSessionRow({ previewOwnsLayout, ...props }: SortableSessionRowProps) {
  return <SidebarSessionRow {...props} {...useSortableBindings(props.session.id, { previewOwnsLayout })} />
}

function SortableProjectOverviewRow(props: React.ComponentProps<typeof ProjectOverviewRow>) {
  return <ProjectOverviewRow {...props} {...useSortableBindings(props.project.id)} />
}

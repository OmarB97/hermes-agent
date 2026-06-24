import {
  closestCenter,
  type CollisionDetection,
  DndContext,
  getFirstCollision,
  pointerWithin,
  rectIntersection,
  type DragEndEvent,
  type DragOverEvent,
  DragOverlay,
  type DragStartEvent,
  KeyboardSensor,
  MouseSensor,
  TouchSensor,
  useDroppable,
  useSensor,
  useSensors
} from '@dnd-kit/core'
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { PlatformAvatar } from '@/app/messaging/platform-icon'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { KbdGroup } from '@/components/ui/kbd'
import { SearchField } from '@/components/ui/search-field'
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem
} from '@/components/ui/sidebar'
import { Skeleton } from '@/components/ui/skeleton'
import { Tip } from '@/components/ui/tooltip'
import { searchSessions, type SessionInfo, type SessionSearchResult } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { comboTokens } from '@/lib/keybinds/combo'
import { profileColor } from '@/lib/profile-color'
import { sessionMatchesSearch } from '@/lib/session-search'
import { normalizeSessionSource, sessionSourceLabel } from '@/lib/session-source'
import { cn } from '@/lib/utils'
import { $cronJobs } from '@/store/cron'
import {
  $panesFlipped,
  $pinnedSessionIds,
  $sidebarAgentsGrouped,
  $sidebarArchivedOpen,
  $sidebarCronOpen,
  $sidebarMessagingOpenIds,
  $sidebarOpen,
  $sidebarOverlayMounted,
  $sidebarPinsOpen,
  $sidebarRecentsOpen,
  $sidebarSessionOrderIds,
  $sidebarWorkspaceOrderIds,
  pinSession,
  reorderPinnedSession,
  SESSION_SEARCH_FOCUS_EVENT,
  setSidebarAgentsGrouped,
  setSidebarArchivedOpen,
  setSidebarCronOpen,
  setSidebarPinsOpen,
  setSidebarRecentsOpen,
  setSidebarSessionOrderIds,
  setSidebarWorkspaceOrderIds,
  SIDEBAR_SESSIONS_PAGE_SIZE,
  toggleSidebarMessagingOpen,
  unpinSession
} from '@/store/layout'
import {
  $newChatProfile,
  $profiles,
  $profileScope,
  ALL_PROFILES,
  newSessionInProfile,
  normalizeProfileKey
} from '@/store/profile'
import {
  $archivedSessions,
  $archivedSessionsLoading,
  $archivedSessionsTotal,
  $cronSessions,
  $messagingPlatformTotals,
  $messagingSessions,
  $messagingTruncated,
  $selectedStoredSessionId,
  $sessionProfileTotals,
  $sessions,
  $sessionsInitialLoadComplete,
  $sessionsLoading,
  $sessionsTotal,
  $workingSessionIds,
  sessionPinId
} from '@/store/session'
import {
  $sidebarSelection,
  pruneSidebarSelection,
  rangeSelectSessions,
  type SidebarSectionKey,
  toggleSessionSelected
} from '@/store/sidebar-selection'

import { type AppView, ARTIFACTS_ROUTE, MESSAGING_ROUTE, SKILLS_ROUTE } from '../../routes'
import type { SidebarNavItem } from '../../types'
import { SidebarCount, SidebarSectionHeader } from './section-header'

import { CloudChannelsDialog } from './cloud-channels-dialog'
import { SidebarCronJobsSection } from './cron-jobs-section'
import { SidebarLoadMoreRow } from './load-more-row'
import { ProfileRail } from './profile-switcher'
import { SelectionActionBar } from './selection-action-bar'
import { SidebarSessionRow } from './session-row'
import { deriveSidebarSessionVisibility } from './session-visibility'
import { VirtualSessionList } from './virtual-session-list'

const VIRTUALIZE_THRESHOLD = 25

// Non-session groups (messaging platforms) stay compact: show a few rows up
// front, reveal more in larger steps on demand. Keeps a busy platform from
// dominating the sidebar before the user asks to see it.
const NON_SESSION_INITIAL_ROWS = 3
const NON_SESSION_LOAD_STEP = 10

const NEW_SESSION_KBD = comboTokens('mod+n')

const SIDEBAR_NAV: SidebarNavItem[] = [
  {
    id: 'new-session',
    label: '',
    icon: props => <Codicon name="robot" {...props} />,
    action: 'new-session'
  },
  {
    id: 'skills',
    label: '',
    icon: props => <Codicon name="symbol-misc" {...props} />,
    route: SKILLS_ROUTE
  },
  {
    id: 'cloud-channels',
    label: '',
    icon: props => <Codicon name="cloud" {...props} />,
    action: 'cloud-channels'
  },
  { id: 'messaging', label: '', icon: props => <Codicon name="comment" {...props} />, route: MESSAGING_ROUTE },
  { id: 'artifacts', label: '', icon: props => <Codicon name="files" {...props} />, route: ARTIFACTS_ROUTE }
]

const WORKSPACE_PAGE = 5
// ALL-profiles view: show only the latest N per profile up front to keep the
// unified list scannable, then reveal/fetch more in N-sized steps on demand.
const PROFILE_INITIAL_PAGE = 5
const GROUP_DND_ID_PREFIX = 'group:'
const SECTION_DND_ID_PREFIX = 'section:'
const SIDEBAR_SESSION_DROP_SECTIONS = new Set(['archived', 'pinned', 'sessions'])

// Two modes via the `compact` height variant (styles.css):
//   tall    → each section is shrink-0, capped, its own scroller; Sessions is flex-1.
//   compact → COMPACT_FLAT drops the caps so the whole stack scrolls as one.
// Sections stay shrink-0 so none can be squeezed below its content and bleed onto
// the next — the flexbox `min-height: auto` overlap trap that caused the bug.
const COMPACT_FLAT = 'compact:max-h-none compact:overflow-visible'

// Vertical scroll only — never a horizontal bar from glow bleed, long titles, etc.
const SCROLL_Y = 'overflow-y-auto overflow-x-hidden overscroll-contain'

// A non-session group's scroll body: own scroller when tall, flattened when compact.
const GROUP_BODY = cn(SCROLL_Y, COMPACT_FLAT)

const groupDndId = (id: string) => `${GROUP_DND_ID_PREFIX}${id}`

const parseGroupDndId = (id: string) =>
  id.startsWith(GROUP_DND_ID_PREFIX) ? id.slice(GROUP_DND_ID_PREFIX.length) : null

const sidebarSectionDndId = (sectionKey: string) => `${SECTION_DND_ID_PREFIX}${sectionKey}`

const parseSidebarSectionDndId = (id: string) =>
  id.startsWith(SECTION_DND_ID_PREFIX) ? id.slice(SECTION_DND_ID_PREFIX.length) : null

type SidebarSessionDropSectionKey = 'archived' | 'pinned' | 'sessions'

const isSidebarSessionDropSectionKey = (value: unknown): value is SidebarSessionDropSectionKey =>
  typeof value === 'string' && SIDEBAR_SESSION_DROP_SECTIONS.has(value)

const countLabel = (loaded: number, total: number) => (total > loaded ? `${loaded}/${total}` : String(loaded))
const sessionTime = (s: SessionInfo) => s.last_active || s.started_at || 0

function orderByIds<T>(items: T[], getId: (item: T) => string, orderIds: string[]): T[] {
  if (!orderIds.length) {
    return items
  }

  const byId = new Map(items.map(item => [getId(item), item]))
  const seen = new Set<string>()
  const ordered: T[] = []

  for (const id of orderIds) {
    const item = byId.get(id)

    if (item) {
      ordered.push(item)
      seen.add(id)
    }
  }

  // Items missing from the persisted order are new since it was last
  // reconciled. Callers pass recency-sorted lists (newest first), so surface
  // these at the TOP instead of burying them beneath the saved order —
  // otherwise a brand-new session sinks to the bottom of the sidebar and reads
  // as "my latest session never showed up".
  const fresh = items.filter(item => !seen.has(getId(item)))

  return fresh.length ? [...fresh, ...ordered] : ordered
}

function reconcileOrderIds(currentIds: string[], orderIds: string[]): string[] {
  if (!currentIds.length) {
    return []
  }

  if (!orderIds.length) {
    return currentIds
  }

  const current = new Set(currentIds)
  const retained = orderIds.filter(id => current.has(id))
  const retainedSet = new Set(retained)

  // New ids (absent from the saved order) are the newest sessions/groups; keep
  // them ahead of the persisted order so fresh activity surfaces at the top of
  // the sidebar rather than being appended to the bottom.
  const fresh = currentIds.filter(id => !retainedSet.has(id))

  return [...fresh, ...retained]
}

function sameIds(left: string[], right: string[]) {
  return left.length === right.length && left.every((item, index) => item === right[index])
}

const baseName = (path: string) =>
  path
    .replace(/[/\\]+$/, '')
    .split(/[/\\]/)
    .filter(Boolean)
    .pop()

// FTS results cover sessions that aren't in the loaded page; synthesize a
// minimal SessionInfo so they render in the same row component (resume works
// by id; the snippet stands in for the preview).
function searchResultToSession(result: SessionSearchResult): SessionInfo {
  const ts = result.session_started ?? Date.now() / 1000

  return {
    archived: false,
    cwd: null,
    ended_at: null,
    id: result.session_id,
    _lineage_root_id: result.lineage_root ?? null,
    input_tokens: 0,
    is_active: false,
    last_active: ts,
    message_count: 0,
    model: result.model ?? null,
    output_tokens: 0,
    preview: result.snippet?.trim() || null,
    source: result.source ?? null,
    started_at: ts,
    title: null,
    tool_call_count: 0
  }
}

function workspaceGroupsFor(
  sessions: SessionInfo[],
  noWorkspaceLabel: string,
  options: { preserveSessionOrder?: boolean } = {}
): SidebarSessionGroup[] {
  const groups = new Map<string, SidebarSessionGroup>()

  for (const session of sessions) {
    const path = session.cwd?.trim() || ''
    const id = path || '__no_workspace__'
    const label = baseName(path) || path || noWorkspaceLabel

    const group = groups.get(id) ?? { id, label, path: path || null, sessions: [] }
    group.sessions.push(session)
    groups.set(id, group)
  }

  if (!options.preserveSessionOrder) {
    // Groups keep recency order (Map insertion = first-seen in the recency-sorted
    // input, so an active project floats up), but rows *within* a group sort by
    // creation time so they don't reshuffle every time a message lands — keeps
    // muscle memory intact.
    for (const group of groups.values()) {
      group.sessions.sort((a, b) => b.started_at - a.started_at)
    }
  }

  return [...groups.values()]
}

function useSortableBindings(id: string, data?: Record<string, unknown>) {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ data, id })

  return {
    dragging: isDragging,
    dragHandleProps: { ...attributes, ...listeners },
    ref: setNodeRef,
    reorderable: true as const,
    style: {
      transform: CSS.Transform.toString(transform),
      transition: isDragging ? undefined : transition,
      willChange: isDragging ? 'transform' : undefined
    }
  }
}

interface ChatSidebarProps extends React.ComponentProps<typeof Sidebar> {
  currentView: AppView
  onNavigate: (item: SidebarNavItem) => void
  onLoadMoreSessions: () => void
  onLoadMoreProfileSessions?: (profile: string) => Promise<void> | void
  onLoadMoreMessaging?: (platform: string) => Promise<void> | void
  onResumeSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onArchiveSession: (sessionId: string) => void
  onArchiveAllSessions: () => Promise<unknown> | void
  onNewSessionInWorkspace: (path: null | string) => void
  onManageCronJob: (jobId: string) => void
  onTriggerCronJob: (jobId: string) => void
  onEnsureArchivedLoaded?: () => void
  onLoadMoreArchived?: () => void
  onRestoreSession?: (sessionId: string) => void
  onArchiveSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onRestoreSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onDeleteSessions?: (sessionIds: string[]) => Promise<unknown> | void
}

export function ChatSidebar({
  currentView,
  onNavigate,
  onLoadMoreSessions,
  onLoadMoreProfileSessions,
  onLoadMoreMessaging,
  onResumeSession,
  onDeleteSession,
  onArchiveSession,
  onArchiveAllSessions,
  onNewSessionInWorkspace,
  onManageCronJob,
  onTriggerCronJob,
  onEnsureArchivedLoaded,
  onLoadMoreArchived,
  onRestoreSession,
  onArchiveSessions,
  onRestoreSessions,
  onDeleteSessions
}: ChatSidebarProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const sidebarOpen = useStore($sidebarOpen)
  // Collapsed-but-overlay-mounted → render the full sidebar, not just the nav rail.
  const overlayMounted = useStore($sidebarOverlayMounted)
  const contentVisible = sidebarOpen || overlayMounted
  const panesFlipped = useStore($panesFlipped)
  // FLATTENED SIDEBAR: pure flat list — no workspace/agent grouping.
  const agentsGrouped = false
  const pinnedSessionIds = useStore($pinnedSessionIds)
  const pinsOpen = useStore($sidebarPinsOpen)
  const agentsOpen = useStore($sidebarRecentsOpen)
  const cronOpen = useStore($sidebarCronOpen)
  const selectedSessionId = useStore($selectedStoredSessionId)
  const sessions = useStore($sessions)
  const cronSessions = useStore($cronSessions)
  const cronJobs = useStore($cronJobs)
  const [cloudChannelsOpen, setCloudChannelsOpen] = useState(false)
  const [archiveAllOpen, setArchiveAllOpen] = useState(false)
  const [archiveAllSubmitting, setArchiveAllSubmitting] = useState(false)
  const archivedOpen = useStore($sidebarArchivedOpen)
  const archivedSessions = useStore($archivedSessions)
  const archivedTotal = useStore($archivedSessionsTotal)
  const archivedLoading = useStore($archivedSessionsLoading)
  const selection = useStore($sidebarSelection)
  const messagingSessions = useStore($messagingSessions)
  const messagingPlatformTotals = useStore($messagingPlatformTotals)
  const messagingTruncated = useStore($messagingTruncated)
  const sessionsLoading = useStore($sessionsLoading)
  const sessionsInitialLoadComplete = useStore($sessionsInitialLoadComplete)
  const sessionsTotal = useStore($sessionsTotal)
  const sessionProfileTotals = useStore($sessionProfileTotals)
  const workingSessionIds = useStore($workingSessionIds)
  const profiles = useStore($profiles)
  const profileScope = useStore($profileScope)
  // Only surface the profile switcher when more than one profile exists, so
  // single-profile users see the unchanged sidebar.
  const multiProfile = profiles.length > 1
  // Gate ALL-profiles grouping on multiProfile too: if a user drops back to one
  // profile while scope is still ALL (persisted), the rail is hidden and they'd
  // otherwise be stuck in the grouped view with no way out.
  // FLATTENED SIDEBAR: never group by device/profile. Sessions from every
  // profile/device render in one flat list with drag-reorder enabled.
  const showAllProfiles = false
  const agentOrderIds = useStore($sidebarSessionOrderIds)
  const workspaceOrderIds = useStore($sidebarWorkspaceOrderIds)
  const [searchQuery, setSearchQuery] = useState('')
  const [serverMatches, setServerMatches] = useState<SessionSearchResult[]>([])
  const [newSessionKbdFlash, setNewSessionKbdFlash] = useState(false)
  const [profileLoadMorePending, setProfileLoadMorePending] = useState<Record<string, boolean>>({})
  const [messagingLoadMorePending, setMessagingLoadMorePending] = useState<Record<string, boolean>>({})
  const messagingOpenIds = useStore($sidebarMessagingOpenIds)
  // Per-platform count of rows currently revealed (starts at NON_SESSION_INITIAL_ROWS).
  const [messagingVisible, setMessagingVisible] = useState<Record<string, number>>({})
  const searchInputRef = useRef<HTMLInputElement>(null)
  const trimmedQuery = searchQuery.trim()

  // Hotkey (session.focusSearch) → focus the field once it's mounted.
  useEffect(() => {
    const onFocus = () => searchInputRef.current?.focus({ preventScroll: true })

    window.addEventListener(SESSION_SEARCH_FOCUS_EVENT, onFocus)

    return () => window.removeEventListener(SESSION_SEARCH_FOCUS_EVENT, onFocus)
  }, [])

  // Flash the ⌘N hint full-opacity (no transition) for the press, so hitting
  // the shortcut visibly pings its affordance in the sidebar.
  useEffect(() => {
    let timeout: ReturnType<typeof setTimeout> | undefined

    const onShortcut = () => {
      setNewSessionKbdFlash(true)
      clearTimeout(timeout)
      timeout = setTimeout(() => setNewSessionKbdFlash(false), 140)
    }

    window.addEventListener('hermes:new-session-shortcut', onShortcut)

    return () => {
      window.removeEventListener('hermes:new-session-shortcut', onShortcut)
      clearTimeout(timeout)
    }
  }, [])

  const activeSidebarSessionId = currentView === 'chat' ? selectedSessionId : null

  const dndSensors = useSensors(
    useSensor(MouseSensor, { activationConstraint: { distance: 6 } }),
    useSensor(TouchSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  )

  // Profile scope = the "workspace switcher" context. Concrete scope shows only
  // that profile's sessions (clean rows, no per-row tags); ALL fans every
  // profile in, grouped by profile below. Single-profile users land here with
  // scope === their only profile, so nothing is filtered out.
  // FLATTENED SIDEBAR: every session from every profile/device, unified into
  // one flat list (device is an abstraction — all sessions belong to all
  // devices, synced under the hood).
  const visibleSessions = sessions

  const sortedSessions = useMemo(
    () => [...visibleSessions].sort((a, b) => sessionTime(b) - sessionTime(a)),
    [visibleSessions]
  )

  const workingSessionIdSet = useMemo(() => new Set(workingSessionIds), [workingSessionIds])

  // Index sessions by both their live id and their lineage-root id so a pin
  // stored as the pre-compression root resolves to the live continuation tip.
  const sessionByAnyId = useMemo(() => {
    const map = new Map<string, SessionInfo>()

    // Cron and messaging sessions are listed separately but can still be
    // pinned, so index them too — otherwise a pinned non-recents row can't
    // resolve into the Pinned section. Recents take precedence on collisions.
    for (const s of [...cronSessions, ...messagingSessions, ...visibleSessions]) {
      map.set(s.id, s)

      if (s._lineage_root_id && !map.has(s._lineage_root_id)) {
        map.set(s._lineage_root_id, s)
      }
    }

    return map
  }, [visibleSessions, cronSessions, messagingSessions])

  const pinnedSessions = useMemo(() => {
    const seen = new Set<string>()
    const out: SessionInfo[] = []

    for (const pinId of pinnedSessionIds) {
      const session = sessionByAnyId.get(pinId)

      if (session && !seen.has(session.id)) {
        seen.add(session.id)
        out.push(session)
      }
    }

    return out
  }, [pinnedSessionIds, sessionByAnyId])

  const pinnedRealIdSet = useMemo(() => new Set(pinnedSessions.map(s => s.id)), [pinnedSessions])

  // Full-text search across *all* sessions (not just the loaded page) so 699
  // sessions stay findable. Debounced; loaded sessions are matched instantly
  // client-side and merged ahead of the server hits.
  useEffect(() => {
    if (!trimmedQuery) {
      setServerMatches([])

      return
    }

    let cancelled = false

    const id = window.setTimeout(() => {
      void searchSessions(trimmedQuery)
        .then(res => {
          if (!cancelled) {
            setServerMatches(res.results)
          }
        })
        .catch(() => undefined)
    }, 200)

    return () => {
      cancelled = true
      window.clearTimeout(id)
    }
  }, [trimmedQuery])

  const searchResults = useMemo(() => {
    if (!trimmedQuery) {
      return []
    }

    const out = new Map<string, SessionInfo>()

    for (const s of sortedSessions) {
      if (sessionMatchesSearch(s, trimmedQuery)) {
        out.set(s.id, s)
      }
    }

    for (const match of serverMatches) {
      if (out.has(match.session_id)) {
        continue
      }

      const loaded = sessionByAnyId.get(match.session_id)
      out.set(match.session_id, loaded ?? searchResultToSession(match))
    }

    return [...out.values()]
  }, [trimmedQuery, sortedSessions, serverMatches, sessionByAnyId])

  const unpinnedAgentSessions = useMemo(
    () => sortedSessions.filter(s => !pinnedRealIdSet.has(s.id)),
    [sortedSessions, pinnedRealIdSet]
  )

  useEffect(() => {
    const next = reconcileOrderIds(
      unpinnedAgentSessions.map(s => s.id),
      agentOrderIds
    )

    if (!sameIds(next, agentOrderIds)) {
      setSidebarSessionOrderIds(next)
    }
  }, [agentOrderIds, unpinnedAgentSessions])

  const agentSessions = useMemo(
    () => orderByIds(unpinnedAgentSessions, s => s.id, agentOrderIds),
    [unpinnedAgentSessions, agentOrderIds]
  )

  // Recents are local-only: messaging-platform sessions are fetched as their
  // own slice ($messagingSessions) and rendered in self-managed per-platform
  // sections below, so there is no source-grouping magic to untangle here.
  const agentGroups = useMemo(
    () => orderByIds(workspaceGroupsFor(agentSessions, s.noWorkspace), g => g.id, workspaceOrderIds),
    [agentSessions, s.noWorkspace, workspaceOrderIds]
  )

  const loadMoreForProfileGroup = useCallback(
    (profile: string) => {
      if (!onLoadMoreProfileSessions) {
        return
      }

      setProfileLoadMorePending(prev => ({ ...prev, [profile]: true }))

      void Promise.resolve(onLoadMoreProfileSessions(profile))
        .catch(() => undefined)
        .finally(() => setProfileLoadMorePending(({ [profile]: _done, ...rest }) => rest))
    },
    [onLoadMoreProfileSessions]
  )

  const loadMoreForMessaging = useCallback(
    (platform: string) => {
      if (!onLoadMoreMessaging) {
        return
      }

      setMessagingLoadMorePending(prev => ({ ...prev, [platform]: true }))

      void Promise.resolve(onLoadMoreMessaging(platform))
        .catch(() => undefined)
        .finally(() => setMessagingLoadMorePending(({ [platform]: _done, ...rest }) => rest))
    },
    [onLoadMoreMessaging]
  )

  // Reveal another batch of a platform's rows; fetch from the backend too if we
  // run past what's loaded and more remain on disk.
  const revealMoreMessaging = (platform: string, loaded: number, hasMore: boolean) => {
    const next = (messagingVisible[platform] ?? NON_SESSION_INITIAL_ROWS) + NON_SESSION_LOAD_STEP

    setMessagingVisible(prev => ({ ...prev, [platform]: next }))

    if (next > loaded && hasMore) {
      loadMoreForMessaging(platform)
    }
  }

  // Each messaging platform is its own self-managed section: split the
  // separately-fetched messaging slice by source, newest platform first, rows
  // within a platform by recency. Per-platform totals (when a "load more" has
  // resolved them) drive the count + whether more remain on disk.
  const messagingGroups = useMemo<MessagingSection[]>(() => {
    if (!messagingSessions.length) {
      return []
    }

    const bySource = new Map<string, SessionInfo[]>()

    for (const session of messagingSessions) {
      const sourceId = normalizeSessionSource(session.source)

      if (!sourceId) {
        continue
      }

      const list = bySource.get(sourceId) ?? []
      list.push(session)
      bySource.set(sourceId, list)
    }

    return [...bySource.entries()]
      .map(([sourceId, list]) => {
        const ordered = [...list].sort((a, b) => sessionTime(b) - sessionTime(a))
        const known = messagingPlatformTotals[sourceId]
        const total = Math.max(ordered.length, known ?? 0)

        return {
          // Known exact total → more exist iff total exceeds loaded; otherwise
          // the seed fetch was capped, so assume more until a per-platform load
          // resolves the count.
          hasMore: known != null ? known > ordered.length : messagingTruncated,
          label: sessionSourceLabel(sourceId) ?? sourceId,
          sessions: ordered,
          sourceId,
          total
        }
      })
      .sort((a, b) => sessionTime(b.sessions[0]) - sessionTime(a.sessions[0]))
  }, [messagingSessions, messagingPlatformTotals, messagingTruncated])

  // ALL-profiles view: one collapsible group per profile, color on the header
  // (not on every row). Default profile floats to the top, the rest alpha.
  const profileGroups = useMemo<SidebarSessionGroup[] | undefined>(() => {
    if (!showAllProfiles) {
      return undefined
    }

    const groups = new Map<string, SidebarSessionGroup>()

    for (const session of agentSessions) {
      const key = normalizeProfileKey(session.profile)

      const group = groups.get(key) ?? {
        color: profileColor(key),
        id: key,
        label: key,
        mode: 'profile',
        path: null,
        sessions: []
      }

      group.sessions.push(session)

      groups.set(key, group)
    }

    return (
      [...groups.values()]
        .map(group => ({
          ...group,
          loadingMore: Boolean(profileLoadMorePending[group.id]),
          onLoadMore: onLoadMoreProfileSessions ? () => loadMoreForProfileGroup(group.id) : undefined,
          totalCount: Math.max(group.sessions.length, sessionProfileTotals[group.id] ?? 0)
        }))
        // default (root) first, then the rest alphabetically.
        .sort((a, b) => (a.id === 'default' ? -1 : b.id === 'default' ? 1 : a.label.localeCompare(b.label)))
    )
  }, [
    showAllProfiles,
    agentSessions,
    loadMoreForProfileGroup,
    onLoadMoreProfileSessions,
    profileLoadMorePending,
    sessionProfileTotals
  ])

  const displayAgentSessions = agentSessions

  const selectionSessionsById = useMemo(() => {
    const map = new Map<string, SessionInfo>()

    for (const session of [
      ...archivedSessions,
      ...searchResults,
      ...cronSessions,
      ...messagingSessions,
      ...visibleSessions
    ]) {
      map.set(session.id, session)
    }

    return map
  }, [archivedSessions, cronSessions, messagingSessions, searchResults, visibleSessions])

  const selectionSessions = useMemo(
    () =>
      selection.ids
        .map(id => selectionSessionsById.get(id))
        .filter((session): session is SessionInfo => Boolean(session)),
    [selection.ids, selectionSessionsById]
  )

  useEffect(() => {
    const section = selection.section

    if (!section || !selection.ids.length) {
      return
    }

    let rows: SessionInfo[]

    switch (section) {
      case 'archived':
        rows = archivedSessions

        break

      case 'pinned':
        rows = pinnedSessions

        break

      case 'results':
        rows = searchResults

        break

      case 'sessions':
        rows = agentSessions

        break
      default: {
        const sourceId = section.slice('messaging:'.length)
        rows = messagingGroups.find(group => group.sourceId === sourceId)?.sessions ?? []
      }
    }

    pruneSidebarSelection(section, rows)
  }, [selection, agentSessions, archivedSessions, messagingGroups, pinnedSessions, searchResults])

  // Pagination is scope-aware. In "All profiles" mode it tracks the global
  // unified set. When scoped to one profile it must compare that profile's own
  // loaded rows against that profile's total — otherwise a huge default profile
  // keeps "Load more" stuck on while you browse a small one (the aggregator's
  // total sums every profile). Per-profile totals come from the aggregator
  // (children excluded); fall back to the global total / loaded count.
  const loadedSessionCount = showAllProfiles ? sessions.length : visibleSessions.length
  const scopedProfileTotal = showAllProfiles ? undefined : sessionProfileTotals[profileScope]

  const knownSessionTotal = Math.max(
    showAllProfiles ? sessionsTotal : (scopedProfileTotal ?? loadedSessionCount),
    loadedSessionCount
  )

  const hasMoreSessions = knownSessionTotal > loadedSessionCount
  const remainingSessionCount = Math.max(0, knownSessionTotal - loadedSessionCount)

  const recentsMeta = countLabel(agentSessions.length, knownSessionTotal)
  const archiveAllDisabled = sessionsLoading || agentSessions.length === 0 || archiveAllSubmitting

  const handleArchiveAll = async () => {
    if (archiveAllSubmitting) {
      return
    }

    setArchiveAllSubmitting(true)

    try {
      await onArchiveAllSessions()
      setArchiveAllOpen(false)
      onEnsureArchivedLoaded?.()
    } catch {
      // The caller owns the error toast/rollback; keep the dialog open.
    } finally {
      setArchiveAllSubmitting(false)
    }
  }

  const displayAgentGroups = showAllProfiles ? profileGroups : agentsGrouped ? agentGroups : undefined
  const rendersGroupedAgentSessions = Boolean(displayAgentGroups?.length)

  // The recents list owns its own (virtualized) scroll container only when it's a
  // long flat list. In that case it must keep its scroller even in short mode, so
  // we don't flatten it (flattening would defeat virtualization). Short flat lists
  // and grouped views flatten into the single outer scroll instead.
  const recentsVirtualizes = !displayAgentGroups?.length && displayAgentSessions.length >= VIRTUALIZE_THRESHOLD

  useEffect(() => {
    if (!displayAgentGroups?.length || showAllProfiles) {
      return
    }

    const next = reconcileOrderIds(
      displayAgentGroups.map(g => g.id),
      workspaceOrderIds
    )

    if (!sameIds(next, workspaceOrderIds)) {
      setSidebarWorkspaceOrderIds(next)
    }
  }, [displayAgentGroups, showAllProfiles, workspaceOrderIds])

  // Skeletons are a first-load-only affordance; background refreshes keep
  // flipping $sessionsLoading and must not re-flash the section. See
  // ./session-visibility.
  const { showSessionSections, showSessionSkeletons } = deriveSidebarSessionVisibility({
    sessionCount: sortedSessions.length,
    sessionsInitialLoadComplete,
    sessionsLoading
  })

  // ──────────────────────────────────────────────────────────────────────────
  // Canonical dnd-kit multi-container engine.
  //
  // ONE parent <DndContext> wraps every section. Each reorderable section
  // (PINNED, SESSIONS, ARCHIVED) renders its own <SortableContext>. The bug we
  // are killing: during a cross-section drag the active row used to stay in its
  // source SortableContext while ALSO being previewed in the target, so dnd-kit
  // sorted it in the source against an `over` that lived in a different
  // container — the row shuffled all the way down the other list.
  //
  // The fix is the documented multi-container recipe: keep working copies of the
  // two reorderable id-lists and physically MOVE the active id between them on
  // every dragOver, so `active` and `over` are always co-resident in one
  // container. The rendered lists are derived from these working copies while a
  // drag is in flight, and persisted on dragEnd.
  // ──────────────────────────────────────────────────────────────────────────
  const basePinnedIds = useMemo(() => pinnedSessions.map(s => s.id), [pinnedSessions])
  const baseSessionIds = useMemo(() => agentSessions.map(s => s.id), [agentSessions])
  const archivedIdSet = useMemo(() => new Set(archivedSessions.map(s => s.id)), [archivedSessions])

  type SidebarDndContainer = 'archived' | 'pinned' | 'sessions'

  interface SidebarDndDrag {
    activeId: string
    from: SidebarDndContainer
    pinned: string[]
    sessions: string[]
    overArchived: boolean
    /** Width of the source row at grab time, so the floating overlay matches it
     * exactly (the overlay renders in a portal at the document root). */
    overlayWidth: null | number
  }

  const [dndDrag, setDndDrag] = useState<null | SidebarDndDrag>(null)
  const dndDragRef = useRef<null | SidebarDndDrag>(null)

  const commitDndDrag = useCallback((next: null | SidebarDndDrag) => {
    dndDragRef.current = next
    setDndDrag(next)
  }, [])

  // Resolve any id → a live SessionInfo. While a drag restores a row OUT of
  // Archived, that row is not in `sessionByAnyId`, so fall back to the archived
  // list.
  const sessionForDndId = useCallback(
    (id: string): SessionInfo | undefined =>
      sessionByAnyId.get(id) ?? archivedSessions.find(session => session.id === id),
    [archivedSessions, sessionByAnyId]
  )

  const mapIdsToSessions = useCallback(
    (ids: readonly string[]): SessionInfo[] => {
      const out: SessionInfo[] = []

      for (const id of ids) {
        const session = sessionForDndId(id)

        if (session) {
          out.push(session)
        }
      }

      return out
    },
    [sessionForDndId]
  )

  // Effective lists for rendering: the working copies while dragging, the base
  // lists otherwise.
  const effPinnedSessions = useMemo(
    () => (dndDrag ? mapIdsToSessions(dndDrag.pinned) : pinnedSessions),
    [dndDrag, mapIdsToSessions, pinnedSessions]
  )

  const effAgentSessions = useMemo(
    () => (dndDrag ? mapIdsToSessions(dndDrag.sessions) : agentSessions),
    [agentSessions, dndDrag, mapIdsToSessions]
  )

  const dndActiveSession = useMemo(
    () => (dndDrag ? (sessionForDndId(dndDrag.activeId) ?? null) : null),
    [dndDrag, sessionForDndId]
  )

  // Archived is a drop bucket + a source of restore drags, never a reorder lane,
  // so it keeps rendering the real archived list. The one exception: while an
  // archived row is being dragged INTO a live lane (it now lives in the pinned/
  // sessions working copy), drop it from the archived view so the same id is not
  // simultaneously a member of the Archived SortableContext and a live lane —
  // that double-membership is the cross-container conflict this rewrite kills.
  const effArchivedSessions = useMemo(() => {
    if (!dndDrag || dndDrag.from !== 'archived') {
      return archivedSessions
    }

    const placedInLiveLane = dndDrag.pinned.includes(dndDrag.activeId) || dndDrag.sessions.includes(dndDrag.activeId)

    return placedInLiveLane ? archivedSessions.filter(session => session.id !== dndDrag.activeId) : archivedSessions
  }, [archivedSessions, dndDrag])

  // Which reorderable container currently holds an id, reading the working lists
  // first (so a moved row resolves to where the drag put it), then the bases.
  const containerForId = useCallback(
    (id: string, drag: SidebarDndDrag | null): SidebarDndContainer | null => {
      if (drag) {
        if (drag.pinned.includes(id)) {
          return 'pinned'
        }

        if (drag.sessions.includes(id)) {
          return 'sessions'
        }
      }

      if (basePinnedIds.includes(id)) {
        return 'pinned'
      }

      if (baseSessionIds.includes(id)) {
        return 'sessions'
      }

      if (archivedIdSet.has(id)) {
        return 'archived'
      }

      return null
    },
    [archivedIdSet, baseSessionIds, basePinnedIds]
  )

  // Resolve the over-target into a container. An `over` is either a section
  // droppable (`section:<key>`) or a row id.
  const overContainerForEvent = useCallback(
    (event: DragOverEvent | DragEndEvent, drag: SidebarDndDrag | null): SidebarDndContainer | null => {
      const over = event.over

      if (!over) {
        return null
      }

      const overId = String(over.id)
      const section = parseSidebarSectionDndId(overId)

      if (isSidebarSessionDropSectionKey(section)) {
        return section
      }

      // A row id: its current container in the working lists / bases. Archived
      // rows are sources only — landing the pointer over one means "Archived".
      return containerForId(overId, drag)
    },
    [containerForId]
  )

  const sidebarDndSensors = dndSensors

  // Latest pointer Y during a drag (captured in collision detection, which runs
  // each move with the real cursor coords). onDragOver uses this — NOT the
  // dragged row's rect — to decide before/after an over-row, so the insertion
  // tracks the cursor instead of biasing one row low.
  const dndPointerY = useRef<null | number>(null)

  const archivedRowIds = useMemo(() => effArchivedSessions.map(session => session.id), [effArchivedSessions])

  // Multi-container collision: prefer the closest ROW inside the hovered
  // section. Without this, dragging over a section resolves the collision to
  // the section container itself, so a cross-section drop appends to the end
  // instead of landing where the pointer is. Narrowing to the section's rows
  // makes onDragOver insert at the hovered position.
  const collisionDetectionStrategy = useCallback<CollisionDetection>(
    args => {
      if (args.pointerCoordinates) {
        dndPointerY.current = args.pointerCoordinates.y
      }

      const pointerHits = pointerWithin(args)

      // Prefer the actual ROW the pointer is over. The closestCenter narrowing
      // below keys off the dragged row's rect (which sits a bit below the
      // cursor), so it resolves the over-row one too low and the drop lands one
      // row down. A real row under the pointer tracks the cursor exactly.
      const rowUnderPointer = pointerHits.find(hit => !parseSidebarSectionDndId(String(hit.id)))

      if (rowUnderPointer) {
        return [{ id: rowUnderPointer.id }]
      }

      const intersections = pointerHits.length > 0 ? pointerHits : rectIntersection(args)
      let overId = getFirstCollision(intersections, 'id')

      if (overId == null) {
        return []
      }

      const overSection = parseSidebarSectionDndId(String(overId))

      if (isSidebarSessionDropSectionKey(overSection)) {
        const rowIds =
          overSection === 'pinned'
            ? (dndDrag?.pinned ?? basePinnedIds)
            : overSection === 'sessions'
              ? (dndDrag?.sessions ?? baseSessionIds)
              : archivedRowIds
        const rowSet = new Set(rowIds)
        const within = args.droppableContainers.filter(container => rowSet.has(String(container.id)))

        if (within.length > 0) {
          const closest = closestCenter({ ...args, droppableContainers: within })

          if (closest.length > 0) {
            overId = closest[0].id
          }
        }
      }

      return [{ id: overId }]
    },
    [archivedRowIds, baseSessionIds, basePinnedIds, dndDrag]
  )

  const handleDndStart = useCallback(
    (event: DragStartEvent) => {
      const activeId = String(event.active.id)

      if (parseGroupDndId(activeId)) {
        commitDndDrag(null)

        return
      }

      const from = containerForId(activeId, null)

      if (!from) {
        commitDndDrag(null)

        return
      }

      commitDndDrag({
        activeId,
        from,
        overArchived: false,
        overlayWidth: event.active.rect.current.initial?.width ?? null,
        pinned: [...basePinnedIds],
        sessions: [...baseSessionIds]
      })
    },
    [baseSessionIds, basePinnedIds, commitDndDrag, containerForId]
  )

  const handleDndOver = useCallback(
    (event: DragOverEvent) => {
      const current = dndDragRef.current

      if (!current) {
        return
      }

      const overContainer = overContainerForEvent(event, current)

      if (!overContainer) {
        return
      }

      const { activeId } = current
      const overId = String(event.over?.id ?? '')

      // Archived is a drop bucket, not a reorder lane: collapse the source by
      // removing the active id from BOTH working lists and flag overArchived.
      if (overContainer === 'archived') {
        if (current.overArchived && !current.pinned.includes(activeId) && !current.sessions.includes(activeId)) {
          return
        }

        commitDndDrag({
          ...current,
          overArchived: true,
          pinned: current.pinned.filter(id => id !== activeId),
          sessions: current.sessions.filter(id => id !== activeId)
        })

        return
      }

      // Target is a reorder lane (pinned/sessions). Rebuild it with the active
      // id spliced in at the over-row position, and ensure the active id is
      // absent from the OTHER lane.
      const targetKey = overContainer === 'pinned' ? 'pinned' : 'sessions'
      const curTarget = targetKey === 'pinned' ? current.pinned : current.sessions
      // The other lane never keeps the active id while it's over this one.
      const otherFiltered = (targetKey === 'pinned' ? current.sessions : current.pinned).filter(id => id !== activeId)
      const activeIndexInTarget = curTarget.indexOf(activeId)
      const overIsRow =
        Boolean(overId) && !parseSidebarSectionDndId(overId) && overId !== activeId && curTarget.includes(overId)

      let newTarget: string[]

      if (overIsRow) {
        const overIndex = curTarget.indexOf(overId)

        if (activeIndexInTarget >= 0) {
          // Same lane: move the active to the hovered row's slot. arrayMove
          // accounts for removing the active first, so dropping ONTO a row lands
          // the active exactly at that row's slot (no off-by-one shift).
          newTarget = arrayMove(curTarget, activeIndexInTarget, overIndex)
        } else {
          // Entering this lane from the other one: insert before/after the
          // hovered row by the REAL pointer vs the row midpoint.
          const overRect = event.over?.rect ?? null
          const pointerY = dndPointerY.current
          const after = pointerY != null && overRect ? pointerY > overRect.top + overRect.height / 2 : false

          const insertAt = overIndex + (after ? 1 : 0)
          newTarget = [...curTarget.slice(0, insertAt), activeId, ...curTarget.slice(insertAt)]
        }
      } else if (activeIndexInTarget >= 0) {
        // Over its own slot, or the bare container while already in this lane:
        // leave it where it is.
        newTarget = curTarget
      } else {
        // Entering an empty lane / bare container from the other: append.
        newTarget = [...curTarget, activeId]
      }

      const resolvedPinned = targetKey === 'pinned' ? newTarget : otherFiltered
      const resolvedSessions = targetKey === 'sessions' ? newTarget : otherFiltered

      // Skip the state update when nothing changed (guards the render loop).
      if (
        !current.overArchived &&
        sameIds(resolvedPinned, current.pinned) &&
        sameIds(resolvedSessions, current.sessions)
      ) {
        return
      }

      commitDndDrag({
        ...current,
        overArchived: false,
        pinned: resolvedPinned,
        sessions: resolvedSessions
      })
    },
    [commitDndDrag, overContainerForEvent]
  )

  const handleDndCancel = useCallback(() => {
    commitDndDrag(null)
  }, [commitDndDrag])

  const handleDndEnd = useCallback(
    (event: DragEndEvent) => {
      const drag = dndDragRef.current

      commitDndDrag(null)

      if (!drag) {
        return
      }

      const { activeId, from } = drag

      // Released with no droppable under the pointer → cancel: discard the
      // working copies entirely, no persistence (matches Escape-to-cancel).
      if (!event.over) {
        return
      }

      // Released over the Archived bucket → archive (and, if it came out of
      // Archived, that's a no-op the resolver below ignores).
      if (drag.overArchived) {
        if (from !== 'archived') {
          onArchiveSession(activeId)
          setSidebarArchivedOpen(true)
          onEnsureArchivedLoaded?.()
        }

        return
      }

      const landedPinned = drag.pinned.includes(activeId)
      const landedSessions = drag.sessions.includes(activeId)

      // Persist the pinned lane: working ids are live session ids; the pin store
      // is keyed by durable (lineage-root) pin ids, so translate before saving.
      const newPinIds = drag.pinned
        .map(id => {
          const session = sessionForDndId(id)

          return session ? sessionPinId(session) : id
        })
        // De-dupe defensively (a lineage tip and its root could both map in).
        .filter((pinId, index, all) => all.indexOf(pinId) === index)

      if (!sameIds(newPinIds, pinnedSessionIds)) {
        $pinnedSessionIds.set(newPinIds)
      }

      if (!sameIds(drag.sessions, agentOrderIds)) {
        setSidebarSessionOrderIds(drag.sessions)
      }

      // Restoring a row OUT of Archived into a live lane.
      if (from === 'archived' && (landedPinned || landedSessions)) {
        onRestoreSession?.(activeId)
      }

      if (landedPinned) {
        setSidebarPinsOpen(true)
      }

      if (landedSessions) {
        setSidebarRecentsOpen(true)
      }
    },
    [
      agentOrderIds,
      commitDndDrag,
      onArchiveSession,
      onEnsureArchivedLoaded,
      onRestoreSession,
      pinnedSessionIds,
      sessionForDndId
    ]
  )

  return (
    <>
      <Sidebar
        className={cn(
          'relative h-full min-w-0 overflow-hidden border-t-0 border-b-0 text-foreground transition-none',
          panesFlipped ? 'border-l border-r-0' : 'border-r border-l-0',
          sidebarOpen
            ? 'border-(--sidebar-edge-border) bg-(--ui-sidebar-surface-background) opacity-100'
            : 'pointer-events-none border-transparent bg-transparent opacity-0',
          // While floated by PaneShell's hover-reveal, force visible + interactive
          // — on hover (group-hover/reveal) or when keyboard-pinned (data-forced).
          'in-data-[pane-hover-reveal=open]:pointer-events-auto in-data-[pane-hover-reveal=open]:border-(--sidebar-edge-border) in-data-[pane-hover-reveal=open]:bg-(--ui-sidebar-surface-background) in-data-[pane-hover-reveal=open]:opacity-100',
          'group-hover/reveal:pointer-events-auto group-hover/reveal:border-(--sidebar-edge-border) group-hover/reveal:bg-(--ui-sidebar-surface-background) group-hover/reveal:opacity-100'
        )}
        collapsible="none"
      >
        <SidebarContent className="gap-0 overflow-hidden bg-transparent px-2.5">
          <SidebarGroup className="shrink-0 p-0 pb-2 pt-[calc(var(--titlebar-height)+0.375rem)]">
            <SidebarGroupContent>
              <SidebarMenu className="gap-px">
                {SIDEBAR_NAV.map(item => {
                  const isInteractive = Boolean(item.action) || Boolean(item.route)

                  const active =
                    (item.id === 'skills' && currentView === 'skills') ||
                    (item.id === 'messaging' && currentView === 'messaging') ||
                    (item.id === 'artifacts' && currentView === 'artifacts')

                  const isNewSession = item.id === 'new-session'
                  const isCloudChannels = item.id === 'cloud-channels'

                  return (
                    <SidebarMenuItem key={item.id}>
                      <SidebarMenuButton
                        aria-disabled={!isInteractive}
                        className={cn(
                          // no-drag: these rows sit directly under the titlebar's
                          // [-webkit-app-region:drag] strips (app-shell.tsx), with only
                          // 6px of clearance. Drag regions win hit-testing over DOM
                          // (pointer-events can't override), and on Linux/WSLg the
                          // resolved region has been observed to swallow clicks on the
                          // top rows. Same carve-out as USER_BUBBLE_BASE_CLASS in
                          // thread.tsx.
                          'flex h-7 w-full justify-start gap-2 rounded-md border border-transparent px-2 text-left text-[0.8125rem] font-medium text-(--ui-text-secondary) transition-colors duration-100 ease-out [-webkit-app-region:no-drag] hover:bg-(--ui-control-hover-background) hover:text-foreground hover:transition-none',
                          active &&
                            'border-(--ui-stroke-tertiary) bg-(--ui-control-active-background) text-foreground shadow-none hover:border-(--ui-stroke-tertiary)!',
                          !isInteractive &&
                            'cursor-default hover:border-transparent hover:bg-transparent hover:text-inherit'
                        )}
                        onClick={() => {
                          // A plain new session lands in whatever profile the live
                          // gateway is on (= the active switcher context). null →
                          // no swap. The switcher header is the single place to
                          // change which profile that is.
                          if (isNewSession) {
                            $newChatProfile.set(null)
                          }

                          if (isCloudChannels) {
                            setCloudChannelsOpen(true)

                            return
                          }

                          onNavigate(item)
                        }}
                        tooltip={s.nav[item.id] ?? item.label}
                        type="button"
                      >
                        <item.icon className="size-4 shrink-0 text-[color-mix(in_srgb,currentColor_72%,transparent)]" />
                        {contentVisible && (
                          <>
                            <span className="min-w-0 flex-1 truncate">{s.nav[item.id] ?? item.label}</span>
                            {isNewSession && (
                              <KbdGroup
                                className={cn('ml-auto opacity-55', newSessionKbdFlash && 'opacity-100!')}
                                keys={[...NEW_SESSION_KBD]}
                                size="sm"
                              />
                            )}
                          </>
                        )}
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  )
                })}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>

          {contentVisible && showSessionSections && (
            <div className="shrink-0 px-2 pb-1 pt-1">
              <SearchField
                aria-label={s.searchAria}
                inputRef={searchInputRef}
                onChange={setSearchQuery}
                placeholder={s.searchPlaceholder}
                value={searchQuery}
              />
            </div>
          )}

          {contentVisible && showSessionSections && (
            <DndContext
              collisionDetection={collisionDetectionStrategy}
              onDragCancel={handleDndCancel}
              onDragEnd={handleDndEnd}
              onDragOver={handleDndOver}
              onDragStart={handleDndStart}
              sensors={sidebarDndSensors}
            >
              <div className={cn('flex min-h-0 flex-1 flex-col pb-1.75', SCROLL_Y)}>
                {trimmedQuery && (
                  <SidebarSessionsSection
                    activeSessionId={activeSidebarSessionId}
                    contentClassName={cn('flex min-h-0 flex-1 flex-col gap-px pb-1.75', SCROLL_Y)}
                    emptyState={
                      <div className="grid min-h-24 place-items-center rounded-lg px-2 text-center text-xs text-(--ui-text-tertiary)">
                        {s.noMatch(trimmedQuery)}
                      </div>
                    }
                    label={s.results}
                    labelMeta={String(searchResults.length)}
                    onArchiveSession={onArchiveSession}
                    onArchiveSessions={onArchiveSessions}
                    onDeleteSession={onDeleteSession}
                    onDeleteSessions={onDeleteSessions}
                    onRestoreSessions={onRestoreSessions}
                    onResumeSession={onResumeSession}
                    onToggle={() => undefined}
                    onTogglePin={pinSession}
                    open
                    pinned={false}
                    rootClassName="min-h-32 flex-1 overflow-hidden p-0"
                    sectionKey="results"
                    sessions={searchResults}
                    workingSessionIdSet={workingSessionIdSet}
                  />
                )}

                {!trimmedQuery && (
                  <SidebarSessionsSection
                    activeSessionId={activeSidebarSessionId}
                    contentClassName={cn('flex max-h-44 flex-col gap-px rounded-lg pb-2 pt-1', GROUP_BODY)}
                    dndSensors={dndSensors}
                    draggingSessionId={dndDrag?.activeId}
                    emptyState={<SidebarPinnedEmptyState />}
                    label={s.pinned}
                    onArchiveSession={onArchiveSession}
                    onArchiveSessions={onArchiveSessions}
                    onDeleteSession={onDeleteSession}
                    onDeleteSessions={onDeleteSessions}
                    onRestoreSessions={onRestoreSessions}
                    onResumeSession={onResumeSession}
                    onToggle={() => setSidebarPinsOpen(!pinsOpen)}
                    onTogglePin={unpinSession}
                    open={pinsOpen}
                    ownDndContext={false}
                    pinned
                    rootClassName="shrink-0 p-0 pb-1"
                    sectionKey="pinned"
                    sessions={effPinnedSessions}
                    sortable
                    workingSessionIdSet={workingSessionIdSet}
                  />
                )}

                {!trimmedQuery && (
                  <SidebarSessionsSection
                    activeSessionId={activeSidebarSessionId}
                    contentClassName={cn(
                      'flex min-h-0 flex-1 flex-col pb-1.75',
                      SCROLL_Y,
                      // Separate profile sections clearly in the ALL view; rows inside
                      // each group keep their own tight gap-px rhythm.
                      showAllProfiles ? 'gap-3' : 'gap-px',
                      // Flatten into the single scroll when compact — unless this is the
                      // virtualized long list, which must keep its own scroller.
                      !recentsVirtualizes && COMPACT_FLAT
                    )}
                    dndSensors={dndSensors}
                    draggingSessionId={dndDrag?.activeId}
                    emptyState={showSessionSkeletons ? <SidebarSessionSkeletons /> : <SidebarAllPinnedState />}
                    footer={
                      // Hide "load more" only when workspace-grouped (those groups page
                      // themselves). ALL-profiles now pages per-profile from each profile
                      // header; the global footer only applies to non-ALL views.
                      !showAllProfiles && !agentsGrouped && !showSessionSkeletons && hasMoreSessions ? (
                        <SidebarLoadMoreRow
                          loading={sessionsLoading}
                          onClick={onLoadMoreSessions}
                          step={Math.min(SIDEBAR_SESSIONS_PAGE_SIZE, remainingSessionCount)}
                        />
                      ) : null
                    }
                    forceEmptyState={showSessionSkeletons}
                    headerAction={
                      <div className="flex items-center gap-0.5">
                        <div className="grid size-6 shrink-0 place-items-center">
                          {!showAllProfiles && agentSessions.length > 0 ? (
                            <Tip label={s.archiveAllTitle}>
                              <Button
                                aria-label={s.archiveAllAria}
                                className="text-(--ui-text-tertiary) opacity-70 hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100 focus-visible:opacity-100"
                                disabled={archiveAllDisabled}
                                onClick={event => {
                                  event.stopPropagation()
                                  setSidebarRecentsOpen(true)
                                  setArchiveAllOpen(true)
                                }}
                                size="icon-xs"
                                variant="ghost"
                              >
                                <Codicon
                                  name={archiveAllSubmitting ? 'loading' : 'archive'}
                                  size="0.75rem"
                                  spinning={archiveAllSubmitting}
                                />
                              </Button>
                            </Tip>
                          ) : null}
                        </div>
                        <div className="grid size-6 shrink-0 place-items-center">
                          {!showAllProfiles && agentSessions.length > 0 ? (
                            <Tip label={agentsGrouped ? s.groupTitleGrouped : s.groupTitleUngrouped}>
                              <Button
                                aria-label={agentsGrouped ? s.groupAriaGrouped : s.groupAriaUngrouped}
                                className={cn(
                                  'text-(--ui-text-tertiary) opacity-70 hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100 focus-visible:opacity-100',
                                  agentsGrouped && 'bg-(--ui-control-active-background) text-foreground opacity-100'
                                )}
                                onClick={event => {
                                  event.stopPropagation()
                                  setSidebarRecentsOpen(true)
                                  setSidebarAgentsGrouped(!agentsGrouped)
                                }}
                                size="icon-xs"
                                variant="ghost"
                              >
                                <Codicon name={agentsGrouped ? 'list-unordered' : 'root-folder'} size="0.75rem" />
                              </Button>
                            </Tip>
                          ) : null}
                        </div>
                      </div>
                    }
                    // dnd-kit can only target mounted rows; force the
                    // non-virtualized path for the whole list while a drag is in
                    // flight so an off-screen drop target still exists.
                    disableVirtualization={Boolean(dndDrag)}
                    groups={undefined}
                    label={s.sessions}
                    labelMeta={recentsMeta}
                    onArchiveSession={onArchiveSession}
                    onArchiveSessions={onArchiveSessions}
                    onDeleteSession={onDeleteSession}
                    onDeleteSessions={onDeleteSessions}
                    onNewSessionInWorkspace={showAllProfiles ? undefined : onNewSessionInWorkspace}
                    onRestoreSessions={onRestoreSessions}
                    onResumeSession={onResumeSession}
                    onToggle={() => setSidebarRecentsOpen(!agentsOpen)}
                    onTogglePin={pinSession}
                    open={agentsOpen}
                    ownDndContext={false}
                    pinned={false}
                    rootClassName={cn(
                      'min-h-32 flex-1 overflow-hidden p-0',
                      !recentsVirtualizes && 'compact:min-h-0 compact:flex-none compact:overflow-visible'
                    )}
                    sectionKey="sessions"
                    sessions={effAgentSessions}
                    sortable
                    workingSessionIdSet={workingSessionIdSet}
                  />
                )}

                {!trimmedQuery &&
                  messagingGroups.map(group => {
                    const visible = messagingVisible[group.sourceId] ?? NON_SESSION_INITIAL_ROWS
                    const shownSessions = group.sessions.slice(0, visible)
                    // More to show if rows are hidden behind the cap, or the backend
                    // still has older threads on disk.
                    const canRevealMore = visible < group.sessions.length || group.hasMore

                    return (
                      <SidebarSessionsSection
                        activeSessionId={activeSidebarSessionId}
                        contentClassName={cn('flex max-h-56 flex-col gap-px pb-1.75', GROUP_BODY)}
                        emptyState={null}
                        footer={
                          canRevealMore ? (
                            <SidebarLoadMoreRow
                              loading={Boolean(messagingLoadMorePending[group.sourceId])}
                              onClick={() => revealMoreMessaging(group.sourceId, group.sessions.length, group.hasMore)}
                              step={Math.min(NON_SESSION_LOAD_STEP, Math.max(0, group.total - shownSessions.length))}
                            />
                          ) : null
                        }
                        key={group.sourceId}
                        label={group.label}
                        labelIcon={
                          <PlatformAvatar
                            className="size-4 rounded-[4px] text-[0.5625rem] [&_svg]:size-3"
                            platformId={group.sourceId}
                            platformName={group.label}
                          />
                        }
                        labelMeta={countLabel(group.sessions.length, group.total)}
                        onArchiveSession={onArchiveSession}
                        onArchiveSessions={onArchiveSessions}
                        onDeleteSession={onDeleteSession}
                        onDeleteSessions={onDeleteSessions}
                        onRestoreSessions={onRestoreSessions}
                        onResumeSession={onResumeSession}
                        onToggle={() => toggleSidebarMessagingOpen(group.sourceId)}
                        onTogglePin={pinSession}
                        open={messagingOpenIds.includes(group.sourceId)}
                        pinned={false}
                        rootClassName="shrink-0 p-0"
                        sectionKey={`messaging:${group.sourceId}`}
                        sessions={shownSessions}
                        workingSessionIdSet={workingSessionIdSet}
                      />
                    )
                  })}

                {!trimmedQuery && cronJobs.length > 0 && (
                  <SidebarCronJobsSection
                    jobs={cronJobs}
                    label={s.cronJobs}
                    onManageJob={onManageCronJob}
                    onOpenRun={onResumeSession}
                    onToggle={() => setSidebarCronOpen(!cronOpen)}
                    onTriggerJob={onTriggerCronJob}
                    open={cronOpen}
                  />
                )}

                {!trimmedQuery && (archivedTotal > 0 || archivedSessions.length > 0) && (
                  <SidebarSessionsSection
                    activeSessionId={activeSidebarSessionId}
                    archivedRows
                    contentClassName={cn('flex max-h-56 shrink-0 flex-col gap-px pb-1.75', GROUP_BODY)}
                    draggingSessionId={dndDrag?.activeId}
                    emptyState={
                      archivedLoading ? (
                        <SidebarSessionSkeletons />
                      ) : (
                        <div className="flex min-h-7 items-center rounded-lg pl-2 text-[0.75rem] text-(--ui-text-tertiary)">
                          {s.archivedEmpty}
                        </div>
                      )
                    }
                    footer={
                      archivedTotal > archivedSessions.length ? (
                        <SidebarLoadMoreRow
                          loading={archivedLoading}
                          onClick={() => onLoadMoreArchived?.()}
                          step={Math.min(SIDEBAR_SESSIONS_PAGE_SIZE, archivedTotal - archivedSessions.length)}
                        />
                      ) : null
                    }
                    label={s.archived}
                    labelMeta={
                      archivedOpen
                        ? countLabel(archivedSessions.length, Math.max(archivedTotal, archivedSessions.length))
                        : String(Math.max(archivedTotal, archivedSessions.length))
                    }
                    onArchiveSession={onArchiveSession}
                    onArchiveSessions={onArchiveSessions}
                    onDeleteSession={onDeleteSession}
                    onDeleteSessions={onDeleteSessions}
                    onRestoreSession={onRestoreSession}
                    onRestoreSessions={onRestoreSessions}
                    onResumeSession={onResumeSession}
                    onToggle={() => {
                      const next = !archivedOpen
                      setSidebarArchivedOpen(next)

                      if (next) {
                        onEnsureArchivedLoaded?.()
                      }
                    }}
                    onTogglePin={pinSession}
                    open={archivedOpen}
                    ownDndContext={false}
                    pinned={false}
                    rootClassName="shrink-0 p-0"
                    sectionKey="archived"
                    sessions={effArchivedSessions}
                    sortable={effArchivedSessions.length > 0}
                    workingSessionIdSet={workingSessionIdSet}
                  />
                )}
              </div>
              <DragOverlay adjustScale={false} dropAnimation={null}>
                {dndActiveSession && dndDrag ? (
                  <SidebarSessionDragOverlay
                    // Reflect the row's CURRENT lane so the floating overlay's
                    // pin/restore affordances match where it would land: pinned
                    // while over the Pinned lane, archived while over Archived
                    // (or still resting in its Archived source).
                    archived={
                      dndDrag.overArchived ||
                      (dndDrag.from === 'archived' &&
                        !dndDrag.pinned.includes(dndDrag.activeId) &&
                        !dndDrag.sessions.includes(dndDrag.activeId))
                    }
                    isPinned={dndDrag.pinned.includes(dndDrag.activeId)}
                    isWorking={workingSessionIdSet.has(dndActiveSession.id)}
                    session={dndActiveSession}
                    width={dndDrag.overlayWidth}
                  />
                ) : null}
              </DragOverlay>
            </DndContext>
          )}

          {contentVisible && !showSessionSections && <div className="min-h-0 flex-1" />}

          {contentVisible && selection.ids.length > 0 && (
            <SelectionActionBar
              onArchiveSessions={ids => onArchiveSessions?.(ids)}
              onDeleteSessions={ids => onDeleteSessions?.(ids)}
              onRestoreSessions={ids => onRestoreSessions?.(ids)}
              sessions={selectionSessions}
            />
          )}

          {contentVisible && (
            <div className="shrink-0 px-0.5 pb-1 pt-0.5">
              <ProfileRail />
            </div>
          )}
        </SidebarContent>
      </Sidebar>
      <ArchiveAllSessionsDialog
        count={knownSessionTotal}
        onConfirm={handleArchiveAll}
        onOpenChange={setArchiveAllOpen}
        open={archiveAllOpen}
        submitting={archiveAllSubmitting}
      />
      <CloudChannelsDialog onOpenChange={setCloudChannelsOpen} open={cloudChannelsOpen} />
    </>
  )
}

function SidebarSessionDragOverlay({
  archived,
  isPinned,
  isWorking,
  session,
  width
}: {
  archived: boolean
  isPinned: boolean
  isWorking: boolean
  session: SessionInfo
  width: null | number
}) {
  return (
    <div
      className="pointer-events-none rounded-md"
      style={{ width: width ? `${width}px` : undefined }}
    >
      <SidebarSessionRow
        archived={archived}
        className="bg-(--ui-row-hover-background) opacity-100! shadow-lg ring-1 ring-(--ui-stroke-tertiary)"
        isPinned={isPinned}
        isSelected={false}
        isWorking={isWorking}
        nativeDraggable={false}
        onArchive={() => undefined}
        onDelete={() => undefined}
        onPin={() => undefined}
        onRestore={() => undefined}
        onResume={() => undefined}
        reorderable
        session={session}
      />
    </div>
  )
}

interface ArchiveAllSessionsDialogProps {
  count: number
  open: boolean
  onConfirm: () => void | Promise<void>
  onOpenChange: (open: boolean) => void
  submitting: boolean
}

function ArchiveAllSessionsDialog({ count, open, onConfirm, onOpenChange, submitting }: ArchiveAllSessionsDialogProps) {
  const { t } = useI18n()
  const s = t.sidebar

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{s.archiveAllDialogTitle}</DialogTitle>
          <DialogDescription>{s.archiveAllDialogDesc}</DialogDescription>
        </DialogHeader>
        <div className="rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 py-2 text-xs text-(--ui-text-secondary)">
          {count > 0 ? s.archiveAllChecked(count) : s.archiveAllNone}
        </div>
        <DialogFooter>
          <Button disabled={submitting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {s.archiveAllCancel}
          </Button>
          <Button disabled={submitting} onClick={() => void onConfirm()} type="button" variant="destructive">
            {submitting ? s.archiveAllSubmitting : s.archiveAllConfirm}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function SidebarSessionSkeletons() {
  return (
    <div aria-hidden="true" className="grid gap-px">
      {['w-32', 'w-40', 'w-28', 'w-36', 'w-24'].map((width, i) => (
        <div className="grid min-h-7 grid-cols-[minmax(0,1fr)_1.5rem] items-center rounded-lg" key={`${width}-${i}`}>
          <Skeleton className={cn('h-3.5 rounded-full', width)} />
          <Skeleton className="mx-auto size-4 rounded-md opacity-60" />
        </div>
      ))}
    </div>
  )
}

function SidebarAllPinnedState() {
  const { t } = useI18n()

  return (
    <div className="grid min-h-24 place-items-center rounded-lg text-center text-xs text-(--ui-text-tertiary)">
      {t.sidebar.allPinned}
    </div>
  )
}

function SidebarPinnedEmptyState() {
  const { t } = useI18n()

  return (
    <div className="flex min-h-7 items-center gap-1.5 rounded-lg pl-2 text-[0.75rem] text-(--ui-text-tertiary)">
      <span className="grid w-3.5 shrink-0 place-items-center text-(--ui-text-quaternary)">
        <Codicon name="pin" size="0.75rem" />
      </span>
      <span>{t.sidebar.shiftClickHint}</span>
    </div>
  )
}

interface SidebarSessionGroup {
  id: string
  label: string
  path: null | string
  sessions: SessionInfo[]
  // Profile color for the ALL-profiles view; absent for workspace groups.
  color?: null | string
  loadingMore?: boolean
  mode?: 'profile' | 'source' | 'workspace'
  onLoadMore?: () => void
  sourceId?: string
  totalCount?: number
}

interface MessagingSection {
  sourceId: string
  label: string
  sessions: SessionInfo[]
  total: number
  hasMore: boolean
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
  onArchiveSession: (sessionId: string) => void
  onTogglePin: (sessionId: string) => void
  onNewSessionInWorkspace?: (path: null | string) => void
  pinned: boolean
  sectionKey?: SidebarSectionKey
  archivedRows?: boolean
  onRestoreSession?: (sessionId: string) => void
  rootClassName?: string
  contentClassName?: string
  emptyState: React.ReactNode
  forceEmptyState?: boolean
  headerAction?: React.ReactNode
  footer?: React.ReactNode
  groups?: SidebarSessionGroup[]
  labelMeta?: React.ReactNode
  labelIcon?: React.ReactNode
  sortable?: boolean
  draggingSessionId?: null | string
  onReorder?: (event: DragEndEvent) => void
  dndSensors?: ReturnType<typeof useSensors>
  ownDndContext?: boolean
  /** Force the non-virtualized render path (every row mounted). The parent sets
   * this while a cross-section drag is live so dnd-kit can target a row that
   * would otherwise be scrolled out of the virtual window. */
  disableVirtualization?: boolean
  onArchiveSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onRestoreSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onDeleteSessions?: (sessionIds: string[]) => Promise<unknown> | void
}

function SidebarSessionsSection({
  label,
  open,
  onToggle,
  sessions,
  activeSessionId,
  workingSessionIdSet,
  onResumeSession,
  onDeleteSession,
  onArchiveSession,
  onTogglePin,
  onNewSessionInWorkspace,
  pinned,
  sectionKey,
  archivedRows = false,
  onRestoreSession,
  rootClassName,
  contentClassName,
  emptyState,
  forceEmptyState = false,
  headerAction,
  footer,
  groups,
  labelMeta,
  labelIcon,
  sortable = false,
  draggingSessionId,
  onReorder,
  dndSensors,
  ownDndContext = true,
  disableVirtualization = false,
  onArchiveSessions,
  onRestoreSessions,
  onDeleteSessions
}: SidebarSessionsSectionProps) {
  const hasGroupedSessions = Boolean(groups?.some(group => group.sessions.length > 0))
  const showEmptyState = forceEmptyState || (!hasGroupedSessions && sessions.length === 0)
  const dndActive = sortable && (!ownDndContext || !!onReorder)
  const selection = useStore($sidebarSelection)
  const selectable = Boolean(sectionKey)
  const selectionActive = selectable && selection.section === sectionKey && selection.ids.length > 0

  const { isOver: dndDropActive, setNodeRef: setDroppableNodeRef } = useDroppable({
    data: {
      sectionKey
    },
    disabled: !sectionKey,
    id: sidebarSectionDndId(sectionKey ?? label)
  })

  const selectedIdSet = useMemo(
    () => (selectionActive ? new Set(selection.ids) : undefined),
    [selectionActive, selection.ids]
  )

  const orderedIds = useMemo(() => sessions.map(s => s.id), [sessions])

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

  const renderRow = (session: SessionInfo) => {
    const checked = selectedIdSet?.has(session.id) ?? false

    const rowProps = {
      archived: archivedRows,
      bulkSelectedSessionIds: checked && selection.ids.length > 1 ? selection.ids : undefined,
      checked,
      dragging: draggingSessionId === session.id,
      isPinned: pinned,
      isSelected: session.id === activeSessionId,
      isWorking: workingSessionIdSet.has(session.id),
      onArchive: () => onArchiveSession(session.id),
      onArchiveSelectedSessions: onArchiveSessions,
      onDelete: () => onDeleteSession(session.id),
      onDeleteSelectedSessions: onDeleteSessions,
      onPin: () => onTogglePin(sessionPinId(session)),
      onRestore: onRestoreSession ? () => onRestoreSession(session.id) : undefined,
      onRestoreSelectedSessions: onRestoreSessions,
      onResume: () => onResumeSession(session.id),
      onToggleSelect: sectionKey ? (mode: 'range' | 'single') => handleToggleSelect(session.id, mode) : undefined,
      selectable,
      selectionActive,
      session
    }

    return sortable ? (
      <SortableSidebarSessionRow key={session.id} sortableSectionKey={sectionKey} {...rowProps} />
    ) : (
      <SidebarSessionRow key={session.id} nativeDraggable {...rowProps} />
    )
  }

  const renderRows = (items: SessionInfo[]) => items.map(renderRow)

  const renderSessionList = (items: SessionInfo[]) =>
    dndActive ? (
      <SortableContext items={items.map(s => s.id)} strategy={verticalListSortingStrategy}>
        {renderRows(items)}
      </SortableContext>
    ) : (
      renderRows(items)
    )

  const renderNestedSessionList = (items: SessionInfo[]) => {
    const list = dndActive ? (
      <SortableContext items={items.map(s => s.id)} strategy={verticalListSortingStrategy}>
        {renderRows(items)}
      </SortableContext>
    ) : (
      renderRows(items)
    )

    return dndActive && ownDndContext ? (
      <DndContext collisionDetection={closestCenter} onDragEnd={onReorder} sensors={dndSensors}>
        {list}
      </DndContext>
    ) : (
      list
    )
  }

  const flatVirtualized =
    !disableVirtualization && !showEmptyState && !groups?.length && sessions.length >= VIRTUALIZE_THRESHOLD

  let inner: React.ReactNode
  let bodyOwnsDndContext = dndActive && ownDndContext && !showEmptyState

  if (showEmptyState) {
    inner = emptyState
    bodyOwnsDndContext = false
  } else if (groups?.length) {
    const groupNodes = groups.map(group =>
      dndActive ? (
        <SortableSidebarWorkspaceGroup
          group={group}
          key={group.id}
          onNewSession={onNewSessionInWorkspace}
          renderRows={renderNestedSessionList}
        />
      ) : (
        <SidebarWorkspaceGroup
          group={group}
          key={group.id}
          onNewSession={onNewSessionInWorkspace}
          renderRows={renderSessionList}
        />
      )
    )

    const groupedInner = dndActive ? (
      <SortableContext items={groups.map(g => groupDndId(g.id))} strategy={verticalListSortingStrategy}>
        {groupNodes}
      </SortableContext>
    ) : (
      groupNodes
    )

    inner =
      dndActive && ownDndContext ? (
        <DndContext collisionDetection={closestCenter} onDragEnd={onReorder} sensors={dndSensors}>
          {groupedInner}
        </DndContext>
      ) : (
        groupedInner
      )
    bodyOwnsDndContext = false
  } else if (flatVirtualized) {
    const virtualList = (
      <VirtualSessionList
        activeSessionId={activeSessionId}
        archived={archivedRows}
        className={contentClassName}
        draggingSessionId={draggingSessionId ?? undefined}
        onArchiveSession={onArchiveSession}
        onArchiveSessions={onArchiveSessions}
        onDeleteSession={onDeleteSession}
        onDeleteSessions={onDeleteSessions}
        onRestoreSession={onRestoreSession}
        onRestoreSessions={onRestoreSessions}
        onResumeSession={onResumeSession}
        onTogglePin={onTogglePin}
        onToggleSelect={sectionKey ? handleToggleSelect : undefined}
        pinned={pinned}
        sectionKey={sectionKey}
        selectable={selectable}
        selectedIds={selectedIdSet}
        selectedSessionIds={selection.ids}
        selectionActive={selectionActive}
        sessions={sessions}
        sortable={sortable}
        workingSessionIdSet={workingSessionIdSet}
      />
    )

    inner = dndActive ? (
      <SortableContext items={sessions.map(s => s.id)} strategy={verticalListSortingStrategy}>
        {virtualList}
      </SortableContext>
    ) : (
      virtualList
    )
  } else {
    inner = renderSessionList(sessions)
  }

  const body = bodyOwnsDndContext ? (
    <DndContext collisionDetection={closestCenter} onDragEnd={onReorder} sensors={dndSensors}>
      {inner}
    </DndContext>
  ) : (
    inner
  )

  // The virtualizer owns its own scroller, so suppress the wrapper's overflow
  // to avoid a double scroll container.
  const resolvedContentClassName = cn(contentClassName, flatVirtualized && 'overflow-y-visible')

  return (
    <SidebarGroup
      className={cn(
        rootClassName,
        dndDropActive &&
          'rounded-lg bg-(--ui-control-hover-background) ring-1 ring-inset ring-(--ui-stroke-tertiary)'
      )}
      data-sidebar-session-section={sectionKey}
      ref={setDroppableNodeRef}
    >
      <SidebarSectionHeader
        action={headerAction}
        icon={labelIcon}
        label={label}
        meta={labelMeta}
        onToggle={onToggle}
        open={open}
      />
      {open && (
        <SidebarGroupContent className={resolvedContentClassName}>
          {body}
          {footer}
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}

interface SidebarWorkspaceGroupProps extends React.ComponentProps<'div'> {
  group: SidebarSessionGroup
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
  reorderable?: boolean
  dragging?: boolean
  dragHandleProps?: React.HTMLAttributes<HTMLElement>
}

function SidebarWorkspaceGroup({
  group,
  renderRows,
  onNewSession,
  reorderable = false,
  dragging = false,
  dragHandleProps,
  className,
  style,
  ref,
  ...rest
}: SidebarWorkspaceGroupProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const isProfileGroup = group.mode === 'profile'
  const isSourceGroup = group.mode === 'source'
  const pageStep = isProfileGroup ? PROFILE_INITIAL_PAGE : WORKSPACE_PAGE
  const [open, setOpen] = useState(true)
  const [visibleCount, setVisibleCount] = useState(pageStep)

  const loadedCount = group.sessions.length
  // Profile groups know their on-disk total (children excluded); workspace
  // groups only ever page within what's already loaded.
  const totalCount = isProfileGroup ? Math.max(group.totalCount ?? loadedCount, loadedCount) : loadedCount
  const visibleSessions = group.sessions.slice(0, visibleCount)
  const hiddenCount = Math.max(0, totalCount - visibleSessions.length)
  const nextCount = Math.min(pageStep, hiddenCount)

  // Reveal already-loaded rows first; only hit the backend when the next page
  // crosses what's been fetched for this profile.
  const handleProfileLoadMore = () => {
    const target = visibleCount + pageStep

    setVisibleCount(target)

    if (target > loadedCount && loadedCount < totalCount) {
      group.onLoadMore?.()
    }
  }

  return (
    <div
      className={cn(
        'grid gap-px data-[dragging=true]:z-10 data-[dragging=true]:opacity-70 data-[dragging=true]:will-change-transform',
        className
      )}
      data-dragging={dragging ? 'true' : undefined}
      ref={ref}
      style={style}
      {...rest}
    >
      <div className="group/workspace flex min-h-6 items-center gap-1 px-2 pt-1 text-[0.6875rem] font-medium text-(--ui-text-tertiary)">
        <button
          className="flex min-w-0 items-center gap-1.5 bg-transparent text-left hover:text-(--ui-text-secondary)"
          onClick={() => setOpen(value => !value)}
          type="button"
        >
          {group.color ? (
            <span
              aria-hidden="true"
              className="size-2 shrink-0 rounded-full"
              style={{ backgroundColor: group.color }}
            />
          ) : null}
          {isSourceGroup && group.sourceId ? (
            <PlatformAvatar
              className="size-4 rounded-[4px] text-[0.5625rem] [&_svg]:size-3"
              platformId={group.sourceId}
              platformName={group.label}
            />
          ) : null}
          <span className="truncate">{group.label}</span>
          <SidebarCount>
            {isProfileGroup ? countLabel(visibleSessions.length, totalCount) : group.sessions.length}
          </SidebarCount>
          <DisclosureCaret
            className="text-(--ui-text-tertiary) opacity-0 transition group-hover/workspace:opacity-100"
            open={open}
          />
        </button>
        {(onNewSession || isProfileGroup) && (
          <Tip label={s.newSessionIn(group.label)}>
            <button
              aria-label={s.newSessionIn(group.label)}
              className="grid size-4 shrink-0 place-items-center rounded-sm bg-transparent text-(--ui-text-quaternary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/workspace:opacity-100"
              // Profile groups start a fresh session in that profile but keep the
              // all-profiles browse view (newSessionInProfile leaves the scope
              // alone); workspace groups seed the new session's cwd from the path.
              onClick={() => (isProfileGroup ? newSessionInProfile(group.id) : onNewSession?.(group.path))}
              type="button"
            >
              <Codicon name="add" size="0.75rem" />
            </button>
          </Tip>
        )}
        {reorderable && (
          <span
            {...dragHandleProps}
            aria-label={s.reorderWorkspace(group.label)}
            className="ml-auto -my-0.5 grid w-4 shrink-0 cursor-grab touch-none place-items-center self-stretch overflow-hidden active:cursor-grabbing"
            onClick={event => event.stopPropagation()}
          >
            <Codicon
              className={cn(
                'text-(--ui-text-quaternary) opacity-0 transition-opacity group-hover/workspace:opacity-80 hover:text-(--ui-text-secondary)',
                dragging && 'text-(--ui-text-secondary) opacity-100'
              )}
              name="grabber"
              size="0.75rem"
            />
          </span>
        )}
      </div>
      {open && (
        <>
          {renderRows(visibleSessions)}
          {hiddenCount > 0 &&
            (isProfileGroup ? (
              <SidebarLoadMoreRow
                loading={Boolean(group.loadingMore)}
                onClick={handleProfileLoadMore}
                step={nextCount}
              />
            ) : (
              <Tip label={s.showMoreIn(nextCount, group.label)}>
                <button
                  aria-label={s.showMoreIn(nextCount, group.label)}
                  className="ml-auto grid size-5 place-items-center rounded-sm bg-transparent text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground"
                  onClick={() => setVisibleCount(count => count + WORKSPACE_PAGE)}
                  type="button"
                >
                  <Codicon name="ellipsis" size="0.75rem" />
                </button>
              </Tip>
            ))}
        </>
      )}
    </div>
  )
}

interface SortableWorkspaceProps {
  group: SidebarSessionGroup
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
}

function SortableSidebarWorkspaceGroup(props: SortableWorkspaceProps) {
  return <SidebarWorkspaceGroup {...props} {...useSortableBindings(groupDndId(props.group.id))} />
}

type SortableSessionRowProps = React.ComponentProps<typeof SidebarSessionRow> & {
  sortableSectionKey?: SidebarSectionKey
}

function SortableSidebarSessionRow({ sortableSectionKey, ...props }: SortableSessionRowProps) {
  const payload = useMemo(
    () => ({
      archived: Boolean(props.archived),
      id: props.session.id,
      pinId: sessionPinId(props.session),
      pinned: props.isPinned,
      profile: props.session.profile || 'default',
      title: sessionTitle(props.session)
    }),
    [props.archived, props.isPinned, props.session]
  )

  const sortableBindings = useSortableBindings(props.session.id, {
    sessionDragPayload: payload,
    sessionId: props.session.id,
    sourceSectionKey: sortableSectionKey
  })

  return <SidebarSessionRow {...props} {...sortableBindings} />
}

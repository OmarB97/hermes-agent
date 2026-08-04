import { useStore } from '@nanostores/react'
import { memo } from 'react'
import type * as React from 'react'

import { ProfileTag } from '@/app/chat/profile-tag'
import { startSessionDrag } from '@/app/chat/session-drag'
import { PlatformAvatar } from '@/app/messaging/platform-icon'
import { openSession } from '@/app/open-session'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { middleClickHandlers } from '@/lib/middle-click'
import { handoffOriginSource, sessionSourceLabel } from '@/lib/session-source'
import { coarseElapsed } from '@/lib/time'
import { cn } from '@/lib/utils'
import { $attentionSessionIds, $delegatedSessionIds } from '@/store/session-states'

import { SessionStatusDot } from '../session-status-dot'

import { SidebarRowBody, SidebarRowGrab, SidebarRowLabel, SidebarRowLead, SidebarRowShell } from './chrome'
import { SessionActionsMenu, type SessionBulkContextActions, SessionContextMenu } from './session-actions-menu'
import { sessionShowsRunningArc } from './session-row-state'
import { useProfilePrewarm } from './use-profile-prewarm'

interface SidebarSessionRowProps extends React.ComponentProps<'div'> {
  session: SessionInfo
  /** TUI-style tree stem for branched sessions (`└─ ` / `├─ `). */
  branchStem?: string
  isPinned: boolean
  isSelected: boolean
  isWorking: boolean
  onArchive: () => void
  onBranch?: () => void
  onDelete: () => void
  onPin: () => void
  onResume: () => void
  reorderable?: boolean
  dragging?: boolean
  dragHandleProps?: React.HTMLAttributes<HTMLElement>
  /** Tag the row with its owning profile (initial chip + tooltip). Used by
   *  flat cross-profile lists — Pinned and search results in the All-profiles
   *  view — where no group header communicates ownership (#66003). */
  showProfile?: boolean
  /** Row participates in its section's multi-select. */
  selectable?: boolean
  /** At least one row in this section is selected. */
  selectionActive?: boolean
  /** This row is included in the current section selection. */
  checked?: boolean
  onToggleSelect?: (mode: 'range' | 'single') => void
  bulkSelectedSessionIds?: readonly string[]
  onArchiveSelectedSessions?: SessionBulkContextActions['onArchiveSessions']
  onDeleteSelectedSessions?: SessionBulkContextActions['onDeleteSessions']
  onHaltSelectedSessions?: SessionBulkContextActions['onHaltSessions']
  onPromptSelectedSessions?: SessionBulkContextActions['onPromptSessions']
  onSteerSelectedSessions?: SessionBulkContextActions['onSteerSessions']
}

const AGE_KEY = { day: 'ageDay', hour: 'ageHour', minute: 'ageMin' } as const

function formatAge(seconds: number, r: Translations['sidebar']['row']): string {
  const { unit, value } = coarseElapsed(Date.now() - seconds * 1000)

  // Under a minute reads as "now" — the sidebar never shows a seconds tick.
  return unit === 'second' ? r.ageNow : `${value}${r[AGE_KEY[unit]]}`
}

function SidebarSessionRowImpl({
  session,
  branchStem,
  isPinned,
  isSelected,
  isWorking,
  onArchive,
  onBranch,
  onDelete,
  onPin,
  onResume,
  reorderable = false,
  dragging = false,
  dragHandleProps,
  showProfile = false,
  selectable = false,
  selectionActive = false,
  checked = false,
  onToggleSelect,
  bulkSelectedSessionIds,
  onArchiveSelectedSessions,
  onDeleteSelectedSessions,
  onHaltSelectedSessions,
  onPromptSelectedSessions,
  onSteerSelectedSessions,
  className,
  style,
  ref,
  ...rest
}: SidebarSessionRowProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const { cancelPrewarm, startPrewarm } = useProfilePrewarm(session.profile)
  const title = sessionTitle(session)
  const age = formatAge(session.last_active || session.started_at, r)
  const handleLabel = `Reorder ${title}`
  // A handed-off session's live source is local, but it originated on a
  // messaging platform — surface that origin as a small badge so e.g. a
  // Telegram thread continued here still reads as Telegram.
  const handoffSource = handoffOriginSource(session.handoff_state, session.handoff_platform)
  const handoffLabel = handoffSource ? (sessionSourceLabel(handoffSource) ?? handoffSource) : null
  // True when a clarify prompt in this session is waiting on the user.
  const needsInput = useStore($attentionSessionIds).includes(session.id)
  // Started by `hermes desktop spawn --delegated` — it runs with nobody
  // watching and answers its own questions, which is worth knowing before you
  // read what it decided.
  const isDelegated = useStore($delegatedSessionIds).includes(session.id)

  const bulkContextActions =
    checked && bulkSelectedSessionIds && bulkSelectedSessionIds.length > 1
      ? {
          onArchiveSessions: onArchiveSelectedSessions,
          onDeleteSessions: onDeleteSelectedSessions,
          onHaltSessions: onHaltSelectedSessions,
          onPromptSessions: onPromptSelectedSessions,
          onSteerSessions: onSteerSelectedSessions,
          sessionIds: bulkSelectedSessionIds
        }
      : undefined

  const toggleSelect = (mode: 'range' | 'single') => {
    triggerHaptic('selection')
    onToggleSelect?.(mode)
  }

  return (
    <SessionContextMenu
      bulkActions={bulkContextActions}
      onArchive={onArchive}
      onBranch={onBranch}
      onDelete={onDelete}
      onPin={onPin}
      pinned={isPinned}
      profile={session.profile}
      sessionId={session.id}
      title={title}
    >
      <SidebarRowShell
        actions={
          <div className="relative z-2 grid w-[1.375rem] place-items-center" data-row-actions>
            {!isWorking && (
              <span className="pointer-events-none absolute right-6 top-1/2 min-w-6 -translate-y-1/2 text-right text-[0.625rem] leading-none text-(--ui-text-tertiary) opacity-0 transition-opacity group-hover:opacity-100">
                {age}
              </span>
            )}
            <SessionActionsMenu
              onArchive={onArchive}
              onBranch={onBranch}
              onDelete={onDelete}
              onPin={onPin}
              pinned={isPinned}
              profile={session.profile}
              sessionId={session.id}
              title={title}
            >
              <Button
                aria-label={r.sessionActions}
                className="size-5 rounded-[4px] bg-transparent text-transparent transition-colors duration-100 hover:bg-(--ui-control-active-background) hover:text-foreground focus-visible:bg-(--ui-control-active-background) focus-visible:text-foreground focus-visible:ring-0 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground group-hover:text-(--ui-text-tertiary) [&_svg]:size-3.5!"
                size="icon"
                variant="ghost"
              >
                <Codicon name="kebab-vertical" size="0.875rem" />
              </Button>
            </SessionActionsMenu>
          </div>
        }
        className={cn(
          'group row-hover relative',
          (isSelected || checked) && 'bg-(--ui-row-active-background)',
          isWorking && 'text-foreground',
          // Opaque surface while lifted so the dragged row erases what's under
          // it (translucency let the rows below bleed through).
          dragging && 'z-10 cursor-grabbing bg-(--ui-sidebar-surface-background)',
          className
        )}
        data-working={isWorking ? 'true' : undefined}
        onPointerDown={event => {
          // Reorder drags belong to dnd-kit (the grab handle); the ⋯ actions
          // cluster keeps its own gestures. Everything else on the row —
          // including the row-body BUTTON, the natural grab surface — is a
          // session drag source: a POINTER drag on the shared drag session
          // (never native HTML5 DnD: no macOS snap-back, Esc aborts
          // instantly). Sub-threshold releases stay ordinary clicks, so
          // resume / pin / open-in-window are untouched.
          //
          // While a section selection is live the row is a checkbox, not a drag
          // source — a pointer drag there would fight shift-range selection.
          if (selectionActive || (event.target as HTMLElement).closest('[data-reorder-handle], [data-row-actions]')) {
            return
          }

          startSessionDrag({ id: session.id, profile: session.profile || 'default', title }, event)
        }}
        // Hovering a row from another profile (the all-profiles view) telegraphs
        // a cross-profile resume — start that backend's spawn now so the click
        // doesn't pay the full cold boot. Same-profile rows no-op inside
        // prewarmProfileBackend.
        onPointerEnter={startPrewarm}
        onPointerLeave={cancelPrewarm}
        ref={ref}
        style={style}
        {...rest}
      >
        {sessionShowsRunningArc({ isWorking, needsInput }) && (
          <span aria-hidden="true" className="arc-border arc-row" />
        )}
        <SidebarRowBody
          className={cn('z-0 group-hover:pr-12', branchStem && 'pl-3.5')}
          data-session-row-main
          // Middle-click = open in a new tab (browser muscle memory).
          {...middleClickHandlers(() => {
            triggerHaptic('selection')
            openSession(session.id, () => undefined, 'tab')
          })}
          onClick={event => {
            const mod = event.metaKey || event.ctrlKey
            const canSelect = Boolean(selectable && onToggleSelect)

            // Finder-grade selection gestures come FIRST: ⌘/⌃/⌥-click toggles a
            // single row, ⇧-click extends a range (anchored on the open session
            // when the selection is still cold), and once a selection is live a
            // plain click keeps toggling instead of resuming.
            if (canSelect && (mod || event.altKey)) {
              event.preventDefault()
              event.stopPropagation()
              toggleSelect('single')

              return
            }

            if (canSelect && event.shiftKey) {
              event.preventDefault()
              event.stopPropagation()
              toggleSelect('range')

              return
            }

            if (canSelect && selectionActive) {
              event.preventDefault()
              event.stopPropagation()
              toggleSelect('single')

              return
            }

            // ⇧⌘-click → pop into its own window (needs standalone windows).
            if (mod && event.shiftKey) {
              event.preventDefault()
              event.stopPropagation()
              triggerHaptic('selection')
              openSession(session.id, () => undefined, 'window')

              return
            }

            // ⌘/⌃-click → open in a new tab (stack into main).
            if (mod) {
              event.preventDefault()
              event.stopPropagation()
              triggerHaptic('selection')
              openSession(session.id, () => undefined, 'tab')

              return
            }

            // ⇧-click → pin.
            if (event.shiftKey) {
              event.preventDefault()
              event.stopPropagation()
              triggerHaptic('selection')
              onPin()

              return
            }

            onResume()
          }}
        >
          {selectionActive ? (
            <SidebarRowLead>
              <span aria-checked={checked} className="grid size-3 place-items-center" role="checkbox">
                <span
                  className={cn(
                    'grid size-3 place-items-center rounded-[3px] border transition-colors',
                    checked
                      ? 'border-foreground/80 bg-foreground/90 text-(--ui-sidebar-surface-background,var(--background))'
                      : 'border-(--ui-stroke-secondary) bg-transparent'
                  )}
                >
                  {checked && <Codicon name="check" size="0.5rem" />}
                </span>
              </span>
            </SidebarRowLead>
          ) : reorderable ? (
            <SidebarRowGrab
              ariaLabel={handleLabel}
              dragging={dragging}
              dragHandleProps={dragHandleProps}
              leadClassName={needsInput ? 'overflow-visible' : undefined}
            >
              <SessionStatusDot
                branchStem={branchStem}
                className="transition-opacity group-hover/handle:opacity-0 group-focus-within/handle:opacity-0"
                session={session}
                storedSessionId={session.id}
              />
            </SidebarRowGrab>
          ) : (
            <SidebarRowLead className={needsInput ? 'overflow-visible' : 'overflow-hidden'}>
              <SessionStatusDot branchStem={branchStem} session={session} storedSessionId={session.id} />
            </SidebarRowLead>
          )}
          {handoffSource && handoffLabel ? (
            <Tip label={r.handoffOrigin(handoffLabel)}>
              <PlatformAvatar
                className="size-4 rounded-[4px] text-[0.5rem] [&_svg]:size-2.5"
                platformId={handoffSource}
                platformName={handoffLabel}
              />
            </Tip>
          ) : null}
          <SidebarRowLabel className="flex-1 font-normal group-hover:text-foreground group-data-[working=true]:text-foreground/90">
            {title}
          </SidebarRowLabel>
          {isDelegated && (
            <Tip label={r.delegatedRun}>
              <span aria-label={r.delegatedRun} className="shrink-0 text-muted-foreground/55" role="img">
                <Codicon name="robot" size="0.6875rem" />
              </span>
            </Tip>
          )}
          {showProfile && <ProfileTag profile={session.profile} />}
        </SidebarRowBody>
      </SidebarRowShell>
    </SessionContextMenu>
  )
}

// Element-wise list compare — the owning section rebuilds the selected-id array
// on every render, so reference equality would defeat the memo outright.
function sameIdList(a?: readonly string[], b?: readonly string[]): boolean {
  if (a === b) {
    return true
  }

  if (!a || !b || a.length !== b.length) {
    return false
  }

  return a.every((id, index) => id === b[index])
}

// The sidebar re-renders on every stream tick ($sessions/$workingSessionIds
// churn), and it stays mounted beneath every overlay — so an unmemoized row
// re-rendered the whole list (and its Codicon/label/status-dot subtree) on each
// delta, bleeding churn into Settings, Cron, Profiles, Artifacts, etc.
//
// The callback props (onArchive/onResume/…) are fresh closures every render by
// design (they close over the row's session id), so a default memo never bails.
// They're pure id-forwarders, though — identical behavior for a given row — so
// the comparator deliberately ignores them and compares only the DATA that
// changes what the row paints. A row whose session/selection/working/pin state
// is unchanged now bails out, even while a sibling session streams.
//
// The multi-select props ARE data — they decide whether the row paints a
// checkbox, whether that checkbox is checked, and which ids a bulk context
// action targets — so they are compared too, `bulkSelectedSessionIds`
// element-wise because the section hands down a freshly built array.
function rowPropsEqual(a: SidebarSessionRowProps, b: SidebarSessionRowProps): boolean {
  return (
    a.session === b.session &&
    a.isPinned === b.isPinned &&
    a.isSelected === b.isSelected &&
    a.isWorking === b.isWorking &&
    a.branchStem === b.branchStem &&
    a.reorderable === b.reorderable &&
    a.dragging === b.dragging &&
    a.showProfile === b.showProfile &&
    a.selectable === b.selectable &&
    a.selectionActive === b.selectionActive &&
    a.checked === b.checked &&
    sameIdList(a.bulkSelectedSessionIds, b.bulkSelectedSessionIds) &&
    a.dragHandleProps === b.dragHandleProps &&
    a.className === b.className &&
    a.style === b.style
  )
}

export const SidebarSessionRow = memo(SidebarSessionRowImpl, rowPropsEqual)

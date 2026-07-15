import { useStore } from '@nanostores/react'
import { motion } from 'motion/react'
import type * as React from 'react'

import { type SessionDragPayload, writeSessionDrag } from '@/app/chat/composer/inline-refs'
import { PlatformAvatar } from '@/app/messaging/platform-icon'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { handoffOriginSource, sessionSourceLabel } from '@/lib/session-source'
import { coarseElapsed } from '@/lib/time'
import { cn } from '@/lib/utils'
import { $attentionSessionIds, sessionPinId } from '@/store/session'
import { canOpenSessionWindow, openSessionInNewWindow } from '@/store/windows'

import { SidebarRowBody, SidebarRowGrab, SidebarRowLabel, SidebarRowLead, SidebarRowShell } from './chrome'
import { SessionActionsMenu, type SessionBulkContextActions, SessionContextMenu } from './session-actions-menu'

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
  /** Native session-drag ended (drag-to-pin/unpin/reorder) — clears the owner's
   * live drag state so drop-zone highlights and previews reset. */
  onSessionDragEnd?: () => void
  /** Native session-drag started on the row body — hands the owner the payload
   * so it can drive drop-zone previews while the drag is in flight. */
  onSessionDragStart?: (payload: SessionDragPayload) => void
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

function isNestedDragControl(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && Boolean(target.closest('[data-reorder-handle], [data-session-row-actions]'))
}

function formatAge(seconds: number, r: Translations['sidebar']['row']): string {
  const { unit, value } = coarseElapsed(Date.now() - seconds * 1000)

  // Under a minute reads as "now" — the sidebar never shows a seconds tick.
  return unit === 'second' ? r.ageNow : `${value}${r[AGE_KEY[unit]]}`
}

export function SidebarSessionRow({
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
  onSessionDragEnd,
  onSessionDragStart,
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
  const title = sessionTitle(session)
  const age = formatAge(session.last_active || session.started_at, r)
  const handleLabel = `Reorder ${title}`
  // A handed-off session's live source is local, but it originated on a
  // messaging platform — surface that origin as a small badge so e.g. a
  // Telegram thread continued here still reads as Telegram.
  const handoffSource = handoffOriginSource(session.handoff_state, session.handoff_platform)
  const handoffLabel = handoffSource ? (sessionSourceLabel(handoffSource) ?? handoffSource) : null
  // Subscribe per-row (the leaf) instead of drilling a set through the list —
  // the atom is tiny and rarely non-empty. True when a clarify prompt in this
  // session is waiting on the user.
  const needsInput = useStore($attentionSessionIds).includes(session.id)

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

  const rowDragActivationProps =
    reorderable && dragHandleProps && !selectionActive
      ? {
          onKeyDown: (event: React.KeyboardEvent<HTMLElement>) => {
            if (!isNestedDragControl(event.target)) {
              dragHandleProps.onKeyDown?.(event)
            }
          },
          onMouseDown: (event: React.MouseEvent<HTMLElement>) => {
            if (!isNestedDragControl(event.target)) {
              dragHandleProps.onMouseDown?.(event)
            }
          },
          onPointerDown: (event: React.PointerEvent<HTMLElement>) => {
            if (!isNestedDragControl(event.target)) {
              dragHandleProps.onPointerDown?.(event)
            }
          },
          onTouchStart: (event: React.TouchEvent<HTMLElement>) => {
            if (!isNestedDragControl(event.target)) {
              dragHandleProps.onTouchStart?.(event)
            }
          }
        }
      : undefined

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
      <div
        // The stable (never-animated) drag source + hit-test anchor. Pin/unpin
        // drop zones read every row's rect off `[data-session-id]`; keeping that
        // on the outer node (the animated visual lives one level in) means the
        // rects stay put while the preview shuffle animates, so hover anchoring
        // never chases a moving target. This is also the dnd-kit sortable node
        // (`ref`/`style`) and the virtualizer's measured element (`...rest`).
        className={cn(dragging && 'relative z-10')}
        data-session-id={session.id}
        ref={ref}
        style={style}
        {...rest}
      >
        <motion.div
          data-session-row-chrome
          // The lifted row stays glued to the pointer through dnd-kit's direct
          // transform. Only displaced siblings spring between slots; animating
          // the active chrome too adds a second, lagging offset under the cursor.
          layout={dragging ? false : 'position'}
          transition={{ layout: { bounce: 0, duration: 0.2, type: 'spring' } }}
          {...rowDragActivationProps}
        >
          <SidebarRowShell
            actions={
              <div className="relative z-2 grid w-[1.375rem] place-items-center">
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
                    aria-label={r.actionsFor(title)}
                    className="size-5 rounded-[4px] bg-transparent text-transparent transition-colors duration-100 hover:bg-(--ui-control-active-background) hover:text-foreground focus-visible:bg-(--ui-control-active-background) focus-visible:text-foreground focus-visible:ring-0 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground group-hover:text-(--ui-text-tertiary) [&_svg]:size-3.5!"
                    data-session-row-actions
                    size="icon"
                    title={r.sessionActions}
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
              dragging && 'cursor-grabbing bg-(--ui-sidebar-surface-background)',
              className
            )}
            data-working={isWorking ? 'true' : undefined}
          >
            {isWorking && !needsInput && <span aria-hidden="true" className="arc-border" />}
            <SidebarRowBody
              className={cn('z-0 group-hover:pr-12', branchStem && 'pl-3.5')}
              data-session-row-main
              draggable={!reorderable && !selectionActive}
              onClick={event => {
                const canSelect = Boolean(selectable && onToggleSelect)

                if (canSelect && (event.metaKey || event.ctrlKey || event.altKey)) {
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

                if (event.shiftKey) {
                  event.preventDefault()
                  event.stopPropagation()
                  triggerHaptic('selection')
                  onPin()

                  return
                }

                // ⌘-click (mac) / ⌃-click (win/linux) pops the chat into its own
                // window — the universal "open in a new window" gesture. Archive
                // lives in the row's ⋯ and right-click menus. Falls through to a
                // normal resume when standalone windows aren't available (web embed).
                if ((event.metaKey || event.ctrlKey) && canOpenSessionWindow()) {
                  event.preventDefault()
                  event.stopPropagation()
                  triggerHaptic('selection')
                  void openSessionInNewWindow(session.id)

                  return
                }

                onResume()
              }}
              onDoubleClick={selectionActive ? onResume : undefined}
              onDragEnd={event => {
                // The row button is the concrete native drag source. Keeping
                // this lifecycle off the sortable wrapper avoids Electron's
                // packaged-app failure to promote a nested button drag to its
                // draggable ancestor.
                event.stopPropagation()
                onSessionDragEnd?.()
              }}
              onDragStart={event => {
                const payload: SessionDragPayload = {
                  id: session.id,
                  pinId: sessionPinId(session),
                  pinned: isPinned,
                  profile: session.profile || 'default',
                  title
                }

                writeSessionDrag(event.dataTransfer, payload)
                event.stopPropagation()
                onSessionDragStart?.(payload)
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
                  <SessionRowLeadDot
                    branchStem={branchStem}
                    className="transition-opacity group-hover/handle:opacity-0 group-focus-within/handle:opacity-0"
                    isWorking={isWorking}
                    needsInput={needsInput}
                  />
                </SidebarRowGrab>
              ) : (
                <SidebarRowLead className={needsInput ? 'overflow-visible' : 'overflow-hidden'}>
                  <SessionRowLeadDot branchStem={branchStem} isWorking={isWorking} needsInput={needsInput} />
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
            </SidebarRowBody>
          </SidebarRowShell>
        </motion.div>
      </div>
    </SessionContextMenu>
  )
}

function SessionRowLeadDot({
  branchStem,
  isWorking,
  needsInput = false,
  className
}: {
  branchStem?: string
  isWorking: boolean
  needsInput?: boolean
  className?: string
}) {
  return (
    <span className={cn('flex items-center gap-0.5', className)}>
      {branchStem ? (
        <span aria-hidden className="shrink-0 font-mono text-[0.625rem] leading-none text-(--ui-text-quaternary)">
          {branchStem}
        </span>
      ) : null}
      <SidebarRowDot isWorking={isWorking} needsInput={needsInput} />
    </span>
  )
}

function SidebarRowDot({
  isWorking,
  needsInput = false,
  className
}: {
  isWorking: boolean
  needsInput?: boolean
  className?: string
}) {
  const { t } = useI18n()
  const r = t.sidebar.row

  // "Needs input" wins over "working": a clarify-blocked session is technically
  // still running, but the actionable state is that it's waiting on the user.
  // Amber + steady (no ping) reads as "your turn", distinct from the accent
  // pulse of an active turn.
  if (needsInput) {
    return (
      <span
        aria-label={r.needsInput}
        className={cn('quest-glow relative size-1.5 rounded-full bg-amber-500', className)}
        role="status"
        title={r.waitingForAnswer}
      />
    )
  }

  return (
    <span
      aria-label={isWorking ? r.sessionRunning : undefined}
      className={cn(
        'rounded-full',
        isWorking
          ? "relative size-1.5 bg-(--ui-accent) shadow-[0_0_0.625rem_color-mix(in_srgb,var(--ui-accent)_55%,transparent)] before:absolute before:inset-0 before:animate-ping before:rounded-full before:bg-(--ui-accent) before:opacity-70 before:content-['']"
          : 'size-1 bg-(--ui-text-quaternary) opacity-80',
        className
      )}
      role={isWorking ? 'status' : undefined}
    />
  )
}

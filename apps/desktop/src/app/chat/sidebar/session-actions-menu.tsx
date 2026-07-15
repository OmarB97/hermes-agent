import type * as React from 'react'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { CopyButton } from '@/components/ui/copy-button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { renameSession } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { exportSession } from '@/lib/session-export'
import { activeGateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { $activeSessionId, $selectedStoredSessionId, setSessions } from '@/store/session'
import { clearSidebarSelection } from '@/store/sidebar-selection'
import { canOpenSessionWindow, openSessionInNewWindow } from '@/store/windows'

import type { SessionTitleResponse } from '../../types'

import { BulkRuntimeTextDialog, type BulkRuntimeTextMode } from './bulk-runtime-text-dialog'

// Rename a session, preferring the gateway's session.title RPC over REST.
//
// A freshly *branched* session (and any brand-new chat) lives only in the
// gateway's in-memory _sessions map keyed by its RUNTIME id — no row is
// persisted to state.db until the first turn. REST PATCH /api/sessions/{id}
// resolves against the stored sessions table, so it 404s ("Session not found")
// on these runtime-only sessions. The session.title RPC resolves the live
// runtime session AND persists the row on demand, so it succeeds where REST
// cannot. This mirrors the /title slash command's fix (use-prompt-actions.ts).
//
// We only take the RPC path for the ACTIVE/selected session: its runtime id is
// known ($activeSessionId) and it lives on the active gateway, so there is no
// profile-routing ambiguity. Every other row (already persisted, possibly on a
// background profile) keeps the REST path, which handles profile scoping and a
// non-empty title is required by the RPC (it rejects clears), so clears stay on
// REST too.
export async function renameSessionPreferringRpc(
  storedSessionId: string,
  title: string,
  profile?: string
): Promise<{ title?: string }> {
  const isActiveRow = storedSessionId === $selectedStoredSessionId.get()
  const runtimeId = isActiveRow ? $activeSessionId.get() : null
  const gateway = activeGateway()

  if (title && runtimeId && gateway) {
    try {
      const result = await gateway.request<SessionTitleResponse>('session.title', {
        session_id: runtimeId,
        title
      })

      return { title: result?.title ?? title }
    } catch (err) {
      // Fall through to REST — e.g. the socket is mid-reconnect. REST still
      // works for any session that already has a persisted row. Log so a
      // genuine RPC-side failure (which then surfaces a REST 404 for the
      // runtime id) is at least diagnosable instead of silently swallowed.
      console.warn('session.title RPC rename failed; falling back to REST', err)
    }
  }

  return renameSession(storedSessionId, title, profile)
}

interface SessionActions {
  sessionId: string
  title: string
  pinned?: boolean
  profile?: string
  onPin?: () => void
  onBranch?: () => void
  onArchive?: () => void
  onDelete?: () => void
}

type MenuItem = typeof DropdownMenuItem | typeof ContextMenuItem
type BulkSessionHandler = (sessionIds: string[]) => Promise<unknown> | void

export interface SessionBulkContextActions {
  onArchiveSessions?: BulkSessionHandler
  onDeleteSessions?: BulkSessionHandler
  onHaltSessions?: BulkSessionHandler
  onPromptSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onSteerSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  sessionIds: readonly string[]
}

type PendingBulkAction = 'archive' | 'delete' | 'halt' | 'prompt' | 'steer' | null

interface ItemSpec {
  className?: string
  disabled: boolean
  icon: string
  label: string
  onSelect: (event: Event) => void
  variant?: 'destructive'
}

function useSessionActions({
  sessionId,
  title,
  pinned = false,
  profile,
  onPin,
  onBranch,
  onArchive,
  onDelete
}: SessionActions) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const [renameOpen, setRenameOpen] = useState(false)

  const pinItem: ItemSpec = {
    disabled: !onPin,
    icon: 'pin',
    label: pinned ? r.unpin : r.pin,
    onSelect: () => {
      triggerHaptic('selection')
      onPin?.()
    }
  }

  const items: ItemSpec[] = [
    ...(canOpenSessionWindow()
      ? [
          {
            disabled: !sessionId,
            icon: 'link-external',
            label: r.newWindow,
            onSelect: () => {
              triggerHaptic('selection')
              void openSessionInNewWindow(sessionId)
            }
          }
        ]
      : []),
    {
      disabled: !sessionId,
      icon: 'cloud-download',
      label: r.export,
      onSelect: () => {
        triggerHaptic('selection')
        void exportSession(sessionId, { profile, title })
      }
    },
    {
      disabled: !onBranch,
      icon: 'git-branch',
      label: r.branchFrom,
      onSelect: () => {
        triggerHaptic('selection')
        onBranch?.()
      }
    },
    {
      disabled: !sessionId,
      icon: 'edit',
      label: r.rename,
      onSelect: () => {
        triggerHaptic('selection')
        setRenameOpen(true)
      }
    },
    {
      disabled: !onArchive,
      icon: 'archive',
      label: r.archive,
      onSelect: () => {
        triggerHaptic('selection')
        onArchive?.()
      }
    },
    {
      className: 'text-destructive focus:text-destructive',
      disabled: !onDelete,
      icon: 'trash',
      label: t.common.delete,
      onSelect: () => {
        triggerHaptic('warning')
        onDelete?.()
      },
      variant: 'destructive'
    }
  ]

  const renderMenuItem = (Item: MenuItem, { className, disabled, icon, label, onSelect, variant }: ItemSpec) => (
    <Item className={className} disabled={disabled} key={label} onSelect={onSelect} variant={variant}>
      <Codicon name={icon} size="0.875rem" />
      <span>{label}</span>
    </Item>
  )

  const renderItems = (Item: MenuItem) => (
    <>
      {renderMenuItem(Item, pinItem)}
      <CopyButton
        appearance={Item === DropdownMenuItem ? 'menu-item' : 'context-menu-item'}
        disabled={!sessionId}
        errorMessage={r.copyIdFailed}
        iconClassName="size-3.5 text-current"
        key={r.copyId}
        label={r.copyId}
        onCopyError={err => notifyError(err, r.copyIdFailed)}
        text={sessionId}
      />
      {items.map(spec => renderMenuItem(Item, spec))}
    </>
  )

  const renameDialog = (
    <RenameSessionDialog
      currentTitle={title}
      onOpenChange={setRenameOpen}
      open={renameOpen}
      profile={profile}
      sessionId={sessionId}
    />
  )

  return { renameDialog, renderItems }
}

interface SessionActionsMenuProps
  extends SessionActions, Pick<React.ComponentProps<typeof DropdownMenuContent>, 'align' | 'sideOffset'> {
  children: React.ReactNode
}

export function SessionActionsMenu({ children, align = 'end', sideOffset = 6, ...actions }: SessionActionsMenuProps) {
  const { t } = useI18n()
  const { renameDialog, renderItems } = useSessionActions(actions)

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
        <DropdownMenuContent
          align={align}
          aria-label={t.sidebar.row.actionsFor(actions.title)}
          className="w-40"
          sideOffset={sideOffset}
        >
          {renderItems(DropdownMenuItem)}
        </DropdownMenuContent>
      </DropdownMenu>
      {renameDialog}
    </>
  )
}

interface SessionContextMenuProps extends SessionActions {
  bulkActions?: SessionBulkContextActions
  children: React.ReactNode
}

function useBulkSessionActions({
  onArchiveSessions,
  onDeleteSessions,
  onHaltSessions,
  onPromptSessions,
  onSteerSessions,
  sessionIds
}: SessionBulkContextActions) {
  const { t } = useI18n()
  const s = t.sidebar.bulk
  const [pending, setPending] = useState<PendingBulkAction>(null)
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false)
  const [runtimeTextMode, setRuntimeTextMode] = useState<BulkRuntimeTextMode | null>(null)
  const count = sessionIds.length

  const runBulk = async (action: Exclude<PendingBulkAction, null>, run?: BulkSessionHandler) => {
    if (pending || !run) {
      return
    }

    setPending(action)

    try {
      await run([...sessionIds])
      clearSidebarSelection()
    } finally {
      setPending(null)
    }
  }

  const submitRuntimeText = (mode: BulkRuntimeTextMode, text: string) => {
    setRuntimeTextMode(null)
    triggerHaptic('submit')
    void runBulk(mode, ids => (mode === 'prompt' ? onPromptSessions?.(ids, text) : onSteerSessions?.(ids, text)))
  }

  const items: ItemSpec[] = [
    {
      disabled: pending !== null || !onPromptSessions,
      icon: 'arrow-up',
      label: s.promptCount(count),
      onSelect: () => setRuntimeTextMode('prompt')
    },
    {
      disabled: pending !== null || !onSteerSessions,
      icon: 'comment-discussion',
      label: s.steerCount(count),
      onSelect: () => setRuntimeTextMode('steer')
    },
    {
      className: 'text-destructive focus:text-destructive',
      disabled: pending !== null || !onHaltSessions,
      icon: 'debug-stop',
      label: s.haltCount(count),
      onSelect: () => void runBulk('halt', onHaltSessions),
      variant: 'destructive'
    },
    {
      disabled: pending !== null || !onArchiveSessions,
      icon: 'archive',
      label: s.archiveCount(count),
      onSelect: () => void runBulk('archive', onArchiveSessions)
    },
    {
      className: 'text-destructive focus:text-destructive',
      disabled: pending !== null || !onDeleteSessions,
      icon: 'trash',
      label: s.deleteCount(count),
      onSelect: () => setConfirmDeleteOpen(true),
      variant: 'destructive'
    }
  ]

  const renderItems = (Item: MenuItem) =>
    items.map(spec => (
      <Item
        className={spec.className}
        disabled={spec.disabled}
        key={spec.label}
        onSelect={spec.onSelect}
        variant={spec.variant}
      >
        <Codicon name={spec.icon} size="0.875rem" />
        <span>{spec.label}</span>
      </Item>
    ))

  const dialogs = (
    <>
      <Dialog onOpenChange={setConfirmDeleteOpen} open={confirmDeleteOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{s.deleteDialogTitle(count)}</DialogTitle>
            <DialogDescription>{s.deleteDialogDesc}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button onClick={() => setConfirmDeleteOpen(false)} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button
              onClick={() => {
                setConfirmDeleteOpen(false)
                void runBulk('delete', onDeleteSessions)
              }}
              type="button"
              variant="destructive"
            >
              {s.deleteConfirm}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <BulkRuntimeTextDialog
        count={count}
        mode={runtimeTextMode}
        onOpenChange={open => setRuntimeTextMode(open ? runtimeTextMode : null)}
        onSubmit={submitRuntimeText}
        pending={pending !== null}
      />
    </>
  )

  return { count, dialogs, renderItems }
}

export function SessionContextMenu({ bulkActions, children, ...actions }: SessionContextMenuProps) {
  const { t } = useI18n()
  const { renameDialog, renderItems } = useSessionActions(actions)

  const bulk = useBulkSessionActions({
    ...bulkActions,
    sessionIds: bulkActions?.sessionIds ?? []
  })

  const showBulkMenu = Boolean(bulkActions && bulkActions.sessionIds.length > 1)

  return (
    <>
      <ContextMenu>
        <ContextMenuTrigger asChild>{children}</ContextMenuTrigger>
        <ContextMenuContent
          aria-label={showBulkMenu ? t.sidebar.bulk.selectedCount(bulk.count) : t.sidebar.row.actionsFor(actions.title)}
          className="w-48"
        >
          {showBulkMenu ? bulk.renderItems(ContextMenuItem) : renderItems(ContextMenuItem)}
        </ContextMenuContent>
      </ContextMenu>
      {showBulkMenu ? bulk.dialogs : renameDialog}
    </>
  )
}

interface RenameSessionDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string
  currentTitle: string
  profile?: string
}

function RenameSessionDialog({ open, onOpenChange, sessionId, currentTitle, profile }: RenameSessionDialogProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const [value, setValue] = useState(currentTitle)
  const [submitting, setSubmitting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setValue(currentTitle)
      window.setTimeout(() => inputRef.current?.select(), 0)
    }
  }, [currentTitle, open])

  const submit = async () => {
    const next = value.trim()

    if (!sessionId || submitting) {
      return
    }

    if (next === currentTitle.trim()) {
      onOpenChange(false)

      return
    }

    setSubmitting(true)

    try {
      const result = await renameSessionPreferringRpc(sessionId, next, profile)
      const finalTitle = result.title || next || ''
      setSessions(prev => prev.map(s => (s.id === sessionId ? { ...s, title: finalTitle || null } : s)))
      notify({ durationMs: 2_000, kind: 'success', message: r.renamed })
      onOpenChange(false)
    } catch (err) {
      notifyError(err, r.renameFailed)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{r.renameTitle}</DialogTitle>
          <DialogDescription>{r.renameDesc}</DialogDescription>
        </DialogHeader>
        <Input
          autoFocus
          disabled={submitting}
          onChange={event => setValue(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter') {
              event.preventDefault()
              void submit()
            } else if (event.key === 'Escape') {
              onOpenChange(false)
            }
          }}
          placeholder={r.untitledPlaceholder}
          ref={inputRef}
          value={value}
        />
        <DialogFooter>
          <Button disabled={submitting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={submitting} onClick={() => void submit()} type="button">
            {t.common.save}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

import type * as React from 'react'

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { SessionActionsMenu } from './session-actions-menu'

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, ...props }: React.ComponentProps<'button'>) => <button {...props}>{children}</button>
}))

vi.mock('@/components/ui/context-menu', () => ({
  ContextMenu: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ContextMenuContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ContextMenuItem: ({ children, disabled }: { children: React.ReactNode; disabled?: boolean }) => (
    <button disabled={disabled} role="menuitem" type="button">
      {children}
    </button>
  ),
  ContextMenuTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock('@/components/ui/copy-button', () => ({
  writeClipboardText: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/components/ui/dialog', () => ({
  Dialog: ({ children, open }: { children: React.ReactNode; open?: boolean }) => (open ? <div>{children}</div> : null),
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <p>{children}</p>,
  DialogFooter: ({ children }: { children: React.ReactNode }) => <footer>{children}</footer>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <header>{children}</header>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>
}))

vi.mock('@/components/ui/dropdown-menu', () => ({
  DropdownMenu: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DropdownMenuContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DropdownMenuItem: ({ children, disabled }: { children: React.ReactNode; disabled?: boolean }) => (
    <button disabled={disabled} role="menuitem" type="button">
      {children}
    </button>
  ),
  DropdownMenuTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock('@/components/ui/input', () => ({
  Input: (props: React.ComponentProps<'input'>) => <input {...props} />
}))

vi.mock('@/components/ui/select', () => ({
  Select: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SelectContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SelectItem: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SelectTrigger: ({ children }: { children: React.ReactNode }) => <button type="button">{children}</button>,
  SelectValue: () => <span />
}))

vi.mock('@/components/ui/tooltip', () => ({
  Tip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock('@/hermes', () => ({
  renameSession: vi.fn()
}))

vi.mock('@/lib/cloud-share', () => ({
  copyCloudChannelId: vi.fn(),
  deleteCloudChannel: vi.fn(),
  inviteCloudChannelMember: vi.fn(),
  loadCloudChannelMembers: vi.fn(),
  removeCloudChannelMember: vi.fn(),
  setCloudChannelMemberPermission: vi.fn(),
  shareSessionToCloud: vi.fn()
}))

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: vi.fn()
}))

vi.mock('@/lib/session-export', () => ({
  exportSession: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/session', () => ({
  setSessions: vi.fn()
}))

vi.mock('@/store/sidebar-selection', () => ({
  clearSidebarSelection: vi.fn()
}))

vi.mock('@/store/windows', () => ({
  canOpenSessionWindow: () => true,
  openSessionInNewWindow: vi.fn()
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('SessionActionsMenu ordering', () => {
  it('groups related id-copy and destructive cloud actions', () => {
    render(
      <I18nProvider configClient={null}>
        <SessionActionsMenu
          onArchive={vi.fn()}
          onDelete={vi.fn()}
          onPin={vi.fn()}
          onSelect={vi.fn()}
          sessionId="session-123"
          title="Demo session"
        >
          <button type="button">Actions</button>
        </SessionActionsMenu>
      </I18nProvider>
    )

    const labels = screen
      .getAllByRole('menuitem')
      .map(item => item.textContent?.trim())
      .filter(Boolean)

    expect(labels).toEqual([
      'Select',
      'Pin',
      'Rename',
      'Copy ID',
      'Copy cloud ID',
      'New window',
      'Export',
      'Share to cloud',
      'Invite to cloud',
      'Cloud members',
      'Archive',
      'Delete cloud channel',
      'Delete'
    ])
  })
})

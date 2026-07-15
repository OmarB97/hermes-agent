import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { SidebarSessionRow } from './session-row'

function session(over: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd: null,
    ended_at: null,
    _lineage_root_id: null,
    input_tokens: 0,
    is_active: false,
    last_active: 1000,
    message_count: 3,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: null,
    started_at: 1000,
    title: 'Native session',
    id: 's1',
    tool_call_count: 0,
    ...over
  } as SessionInfo
}

function renderSelectableRow(over: Partial<React.ComponentProps<typeof SidebarSessionRow>> = {}) {
  const handlers = {
    onArchive: vi.fn(),
    onDelete: vi.fn(),
    onPin: vi.fn(),
    onResume: vi.fn(),
    onToggleSelect: vi.fn()
  }

  const utils = render(
    <SidebarSessionRow
      isPinned={false}
      isSelected={false}
      isWorking={false}
      onArchive={handlers.onArchive}
      onDelete={handlers.onDelete}
      onPin={handlers.onPin}
      onResume={handlers.onResume}
      onToggleSelect={handlers.onToggleSelect}
      selectable
      session={session()}
      {...over}
    />
  )

  return {
    ...utils,
    handlers,
    rowButton: utils.container.querySelector('[data-session-row-main]') as HTMLButtonElement
  }
}

afterEach(() => {
  cleanup()
})

describe('SidebarSessionRow multi-select gestures', () => {
  it('uses modifier click for non-contiguous selection and shift-click for a range', () => {
    const { handlers, rowButton } = renderSelectableRow()

    fireEvent.click(rowButton, { metaKey: true })
    fireEvent.click(rowButton, { shiftKey: true })

    expect(handlers.onToggleSelect).toHaveBeenNthCalledWith(1, 'single')
    expect(handlers.onToggleSelect).toHaveBeenNthCalledWith(2, 'range')
    expect(handlers.onPin).not.toHaveBeenCalled()
    expect(handlers.onResume).not.toHaveBeenCalled()
  })

  it('turns plain clicks into toggles and disables native drag while selection is active', () => {
    const { container, handlers, rowButton } = renderSelectableRow({ checked: true, selectionActive: true })

    fireEvent.click(rowButton)

    expect(handlers.onToggleSelect).toHaveBeenCalledWith('single')
    expect(handlers.onResume).not.toHaveBeenCalled()
    expect(rowButton.draggable).toBe(false)
    expect(container.querySelector('[role="checkbox"]')?.getAttribute('aria-checked')).toBe('true')
  })

  it('routes a checked row context menu to the whole selected set', async () => {
    const onArchiveSelectedSessions = vi.fn()

    const { handlers, rowButton } = renderSelectableRow({
      bulkSelectedSessionIds: ['s1', 's2', 's3'],
      checked: true,
      onArchiveSelectedSessions,
      selectionActive: true
    })

    fireEvent.contextMenu(rowButton)
    fireEvent.click(await screen.findByText('Archive 3'))

    await waitFor(() => expect(onArchiveSelectedSessions).toHaveBeenCalledWith(['s1', 's2', 's3']))
    expect(handlers.onArchive).not.toHaveBeenCalled()
  })
})

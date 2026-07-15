import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { readSessionDrag } from '@/app/chat/composer/inline-refs'
import type { SessionInfo } from '@/hermes'
import { $attentionSessionIds } from '@/store/session'

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

function fakeTransfer(data: Record<string, string> = {}) {
  const store = { ...data }

  return {
    dropEffect: 'none',
    effectAllowed: 'uninitialized',
    getData: (type: string) => store[type] ?? '',
    setData: (type: string, value: string) => {
      store[type] = value
    },
    get types() {
      return Object.keys(store)
    }
  } as unknown as DataTransfer
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
  $attentionSessionIds.set([])
})

describe('SidebarSessionRow native drag activation', () => {
  it('starts the session drag from the concrete row button', () => {
    const onSessionDragStart = vi.fn()
    const onSessionDragEnd = vi.fn()

    const { container } = render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        isWorking={false}
        onArchive={vi.fn()}
        onDelete={vi.fn()}
        onPin={vi.fn()}
        onResume={vi.fn()}
        onSessionDragEnd={onSessionDragEnd}
        onSessionDragStart={onSessionDragStart}
        session={session()}
      />
    )

    const rowButton = container.querySelector('[data-session-row-main]') as HTMLButtonElement
    const transfer = fakeTransfer()

    expect(rowButton.draggable).toBe(true)

    fireEvent.dragStart(rowButton, { dataTransfer: transfer })

    expect(readSessionDrag(transfer)).toEqual({
      id: 's1',
      pinId: 's1',
      pinned: false,
      profile: 'default',
      title: 'Native session'
    })
    expect(onSessionDragStart).toHaveBeenCalledWith(expect.objectContaining({ id: 's1', pinId: 's1', pinned: false }))

    fireEvent.dragEnd(rowButton)

    expect(onSessionDragEnd).toHaveBeenCalledTimes(1)
  })

  it('keeps the sortable wrapper out of the native drag path', () => {
    const onMouseDown = vi.fn()

    const { container } = render(
      <SidebarSessionRow
        dragHandleProps={{ onMouseDown }}
        isPinned
        isSelected={false}
        isWorking={false}
        onArchive={vi.fn()}
        onDelete={vi.fn()}
        onPin={vi.fn()}
        onResume={vi.fn()}
        reorderable
        session={session({ _lineage_root_id: 'root-1' })}
      />
    )

    const anchor = container.querySelector('[data-session-id]') as HTMLDivElement
    const chrome = container.querySelector('[data-session-row-chrome]') as HTMLDivElement
    const rowButton = container.querySelector('[data-session-row-main]') as HTMLButtonElement
    const actions = container.querySelector('[data-session-row-actions]') as HTMLButtonElement

    expect(anchor.draggable).toBe(false)
    expect(rowButton.draggable).toBe(false)

    fireEvent.mouseDown(chrome)
    expect(onMouseDown).toHaveBeenCalledTimes(1)

    fireEvent.mouseDown(actions)
    expect(onMouseDown).toHaveBeenCalledTimes(1)
  })
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

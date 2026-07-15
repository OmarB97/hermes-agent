import { cleanup, fireEvent, render } from '@testing-library/react'
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
    expect(onSessionDragStart).toHaveBeenCalledWith(
      expect.objectContaining({ id: 's1', pinId: 's1', pinned: false })
    )

    fireEvent.dragEnd(rowButton)

    expect(onSessionDragEnd).toHaveBeenCalledTimes(1)
  })

  it('keeps the sortable wrapper out of the native drag path', () => {
    const { container } = render(
      <SidebarSessionRow
        dragHandleProps={{ onMouseDown: vi.fn() }}
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
    const rowButton = container.querySelector('[data-session-row-main]') as HTMLButtonElement

    expect(anchor.draggable).toBe(false)
    expect(rowButton.draggable).toBe(true)
  })
})

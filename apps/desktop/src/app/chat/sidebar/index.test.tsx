import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { readSessionDrag } from '@/app/chat/composer/inline-refs'
import type { SessionInfo } from '@/types/hermes'

import { SidebarSessionsSection } from './index'

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
})

describe('SidebarSessionsSection drag wiring', () => {
  it('forwards native drag lifecycle callbacks from non-sortable rows to the sidebar owner', () => {
    const onSessionDragEnd = vi.fn()
    const onSessionDragStart = vi.fn()

    render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={null}
        label="Sessions"
        onArchiveSession={vi.fn()}
        onDeleteSession={vi.fn()}
        onResumeSession={vi.fn()}
        onSessionDragEnd={onSessionDragEnd}
        onSessionDragStart={onSessionDragStart}
        onToggle={() => undefined}
        onTogglePin={vi.fn()}
        open
        pinned={false}
        sectionKey="sessions"
        sessions={[session()]}
        workingSessionIdSet={new Set()}
      />
    )

    const rowButton = screen.getByText('Native session').closest('[data-session-row-main]') as HTMLButtonElement
    const transfer = fakeTransfer()

    fireEvent.dragStart(rowButton, { dataTransfer: transfer })

    expect(readSessionDrag(transfer)).toMatchObject({
      archived: false,
      id: 's1',
      pinned: false,
      profile: 'default',
      title: 'Native session'
    })
    expect(onSessionDragStart).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 's1',
        pinned: false
      })
    )

    fireEvent.dragEnd(rowButton)

    expect(onSessionDragEnd).toHaveBeenCalledTimes(1)
  })
})

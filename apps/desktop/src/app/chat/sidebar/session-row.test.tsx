import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { startSessionDrag } from '@/app/chat/session-drag'
import type { SessionInfo } from '@/hermes'
import type * as ComposerStatusStore from '@/store/composer-status'
import type * as SessionStore from '@/store/session'
import type * as SessionStatesStore from '@/store/session-states'
import type * as WindowsStore from '@/store/windows'

import type * as SessionActionsMenuModule from './session-actions-menu'
import { SidebarSessionRow } from './session-row'

afterEach(cleanup)

// NOTE: `@/i18n` is deliberately NOT mocked. The multi-select block below drives
// the REAL SessionContextMenu, whose bulk verbs read `t.sidebar.bulk.*` (and its
// confirm dialog reads `t.common.*`) — a hand-written stub would have to mirror
// half the translation table and would silently rot. Using the real bundle also
// keeps the kebab's aria-label assertion honest about the actual copy.

vi.mock('@/app/chat/profile-tag', () => ({ ProfileTag: () => null }))
vi.mock('@/app/chat/session-drag', () => ({ startSessionDrag: vi.fn() }))
// PlatformAvatar is intentionally NOT mocked (do not reintroduce this — see
// #67500, Gille's third pass): it's a forwardRef component that spreads its
// props onto the rendered span, and mocking it with a stand-in that spreads
// props itself only proves the MOCK forwards them, not that the real
// component does. This file exercises the actual production component so a
// regression in its ref/prop forwarding fails here again.
vi.mock('@/lib/chat-runtime', () => ({ sessionTitle: (s: SessionInfo) => (s as unknown as { title: string }).title }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/session-source', () => ({
  handoffOriginSource: (state?: string, platform?: string) => (state && platform ? platform : null),
  sessionSourceLabel: (source: string) => source
}))
vi.mock('@/lib/time', () => ({ coarseElapsed: () => ({ unit: 'minute' as const, value: 5 }) }))

// These mocks use importOriginal rather than replacing the module wholesale:
// session-row.tsx (and its transitive imports, e.g. session-color.ts) reads
// several store exports beyond the ones this file cares about, and that set
// keeps growing as the app evolves upstream. A wholesale replacement mock
// silently turns every export it doesn't list into `undefined`, which then
// crashes nanostores' `computed()` the moment a new dependency is added
// upstream (as happened twice already: $stalledSessionIds, then $sessions).
// Overriding only the named atoms we actually control keeps this test
// resilient to that drift.
vi.mock('@/store/composer-status', async importOriginal => {
  const actual = await importOriginal<typeof ComposerStatusStore>()

  return { ...actual, $backgroundRunningSessionIds: atom<string[]>([]) }
})
vi.mock('@/store/session', async importOriginal => {
  const actual = await importOriginal<typeof SessionStore>()

  return { ...actual, $unreadFinishedSessionIds: atom<string[]>([]) }
})
vi.mock('@/store/session-states', async importOriginal => {
  const actual = await importOriginal<typeof SessionStatesStore>()

  return {
    ...actual,
    $attentionSessionIds: atom<string[]>([]),
    $delegatedSessionIds: atom<string[]>([]),
    $stalledSessionIds: atom<string[]>([]),
    openSessionTile: vi.fn()
  }
})
vi.mock('@/store/windows', async importOriginal => {
  const actual = await importOriginal<typeof WindowsStore>()

  return {
    ...actual,
    canOpenSessionWindow: () => false,
    openSessionInNewWindow: vi.fn()
  }
})

// SessionActionsMenu open behavior is covered in session-actions-menu.test.tsx
// against the real component. Stub it here so the row-chrome block stays focused
// on the row itself (handoff avatar tip, etc.). SessionContextMenu stays REAL —
// the multi-select block asserts a checked row's right-click acts on the whole
// selection, which is that component's job.
vi.mock('./session-actions-menu', async importOriginal => {
  const actual = await importOriginal<typeof SessionActionsMenuModule>()

  return {
    ...actual,
    SessionActionsMenu: ({ children }: { children: React.ReactNode }) => <>{children}</>
  }
})

vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), startPrewarm: vi.fn() })
}))

function makeSession(overrides: Partial<SessionInfo> & { title: string }): SessionInfo {
  return {
    handoff_platform: null,
    handoff_state: null,
    id: 's1',
    last_active: 0,
    profile: 'default',
    started_at: 0,
    ...overrides
  } as unknown as SessionInfo
}

const tipTrigger = (el: HTMLElement) => el.closest('[data-slot="tooltip-trigger"]')

const noop = vi.fn()

describe('SidebarSessionRow', () => {
  it('keeps an aria-label on the kebab without wrapping it in a Tip', () => {
    render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        isWorking={false}
        onArchive={noop}
        onDelete={noop}
        onPin={noop}
        onResume={noop}
        session={makeSession({ title: 'Hermes doctor health check results' })}
      />
    )

    const kebab = screen.getByRole('button', { name: 'Session actions' })
    expect(tipTrigger(kebab)).toBeNull()
  })

  it('does not render a handoff avatar for a locally-started session', () => {
    const { container } = render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        isWorking={false}
        onArchive={noop}
        onDelete={noop}
        onPin={noop}
        onResume={noop}
        session={makeSession({ title: 'Local session' })}
      />
    )

    // PlatformAvatar's span is the only aria-hidden SPAN this row ever
    // renders (idle dot / arc-border / branch-stem are all inactive here) —
    // Codicon icons (e.g. the kebab trigger) are also aria-hidden but render
    // as <i>, not <span>, so this selector doesn't accidentally match them.
    expect(container.querySelector('span[aria-hidden="true"]')).toBeNull()
  })

  it('wraps the handoff platform avatar in a Tip for a session started on another platform', () => {
    const { container } = render(
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        isWorking={false}
        onArchive={noop}
        onDelete={noop}
        onPin={noop}
        onResume={noop}
        session={makeSession({
          handoff_platform: 'telegram',
          handoff_state: 'active',
          title: 'Continued from Telegram'
        })}
      />
    )

    // PlatformAvatar is the REAL component here (see the note above the vi.mock
    // block, #67500 third pass) — it renders the Telegram brand SVG rather
    // than the platform name as text, so query the avatar span itself (the
    // row's only aria-hidden span in this state) rather than text content,
    // and confirm its tooltip trigger actually attaches to it — proving the
    // real forwardRef/...rest path works, not a mock that fakes it.
    const avatar = container.querySelector('span[aria-hidden="true"]')
    expect(avatar).toBeTruthy()
    expect(tipTrigger(avatar as HTMLElement)).toBeTruthy()
  })
})

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
      session={makeSession({ id: 's1', title: 'Native session' })}
      {...over}
    />
  )

  return {
    ...utils,
    handlers,
    rowButton: utils.container.querySelector('[data-session-row-main]') as HTMLElement
  }
}

describe('SidebarSessionRow multi-select gestures', () => {
  it('uses modifier click for non-contiguous selection and shift-click for a range', () => {
    const { handlers, rowButton } = renderSelectableRow()

    fireEvent.click(rowButton, { metaKey: true })
    fireEvent.click(rowButton, { shiftKey: true })

    expect(handlers.onToggleSelect).toHaveBeenNthCalledWith(1, 'single')
    expect(handlers.onToggleSelect).toHaveBeenNthCalledWith(2, 'range')
    // A cold shift-click is a RANGE (anchored on the open session), never the
    // ⇧-click pin shortcut, and a ⌘-click never opens a tab.
    expect(handlers.onPin).not.toHaveBeenCalled()
    expect(handlers.onResume).not.toHaveBeenCalled()
  })

  it('toggles on ⌥-click too, and re-clicking a checked row deselects it', () => {
    const { handlers, rowButton } = renderSelectableRow({ checked: true, selectionActive: true })

    fireEvent.click(rowButton, { altKey: true })
    fireEvent.click(rowButton)

    expect(handlers.onToggleSelect).toHaveBeenNthCalledWith(1, 'single')
    expect(handlers.onToggleSelect).toHaveBeenNthCalledWith(2, 'single')
    expect(handlers.onResume).not.toHaveBeenCalled()
  })

  it('turns plain clicks into toggles and suppresses the row drag while a selection is live', () => {
    vi.mocked(startSessionDrag).mockClear()
    const { container, handlers, rowButton } = renderSelectableRow({ checked: true, selectionActive: true })

    fireEvent.click(rowButton)
    fireEvent.pointerDown(rowButton, { button: 0, pointerType: 'mouse' })

    expect(handlers.onToggleSelect).toHaveBeenCalledWith('single')
    expect(handlers.onResume).not.toHaveBeenCalled()
    // While rows are checkboxes a pointer drag would fight range-select, so the
    // shared session drag must not start.
    expect(startSessionDrag).not.toHaveBeenCalled()
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

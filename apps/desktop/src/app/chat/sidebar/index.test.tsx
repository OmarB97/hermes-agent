import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { I18nProvider } from '@/i18n'
import { $cronJobs } from '@/store/cron'
import {
  $pinnedSessionIds,
  $sidebarMessagingOpenIds,
  $sidebarOverlayMounted,
  setSidebarOpen
} from '@/store/layout'
import { $profiles } from '@/store/profile'
import { clearAllPrompts, setApprovalRequest } from '@/store/prompts'
import {
  $cronSessions,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  $sessionsLoading,
  $workingSessionIds
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { ChatSidebar } from './index'

const session = (over: Partial<SessionInfo>): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'session',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 0,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: null,
  tool_call_count: 0,
  ...over
})

function renderSidebar(onResumeSession = vi.fn()) {
  render(
    <I18nProvider configClient={null}>
      <MemoryRouter>
        <SidebarProvider open>
          <ChatSidebar
            currentView="chat"
            onArchiveSession={vi.fn()}
            onDeleteSession={vi.fn()}
            onLoadMoreSessions={vi.fn()}
            onManageCronJob={vi.fn()}
            onNavigate={vi.fn()}
            onNewSessionInWorkspace={vi.fn()}
            onResumeSession={onResumeSession}
            onTriggerCronJob={vi.fn()}
          />
        </SidebarProvider>
      </MemoryRouter>
    </I18nProvider>
  )
}

describe('ChatSidebar prompt attention section', () => {
  beforeEach(() => {
    window.localStorage.clear()
    setSidebarOpen(true)
    $sidebarOverlayMounted.set(false)
    $sessions.set([])
    $sessionsLoading.set(false)
    $messagingSessions.set([])
    $cronSessions.set([])
    $cronJobs.set([])
    $workingSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $pinnedSessionIds.set([])
    $sidebarMessagingOpenIds.set([])
    $profiles.set([])
    clearAllPrompts()
  })

  afterEach(() => {
    cleanup()
    clearAllPrompts()
    vi.restoreAllMocks()
  })

  it('surfaces a messaging-only session with a pending approval prompt', () => {
    const onResumeSession = vi.fn()
    $messagingSessions.set([
      session({
        id: 'external-channel-session',
        last_active: 20,
        source: 'external-channel',
        title: 'External channel review'
      })
    ])
    setApprovalRequest({
      command: 'delete cloud channel',
      context: {
        meshId: 'mesh-ko',
        orgId: 'org-studios',
        requestedBy: { displayName: 'Khristine', platform: 'external-channel', principalId: 'principal-k' },
        requestedVia: 'external-channel',
        targetAudience: { kind: 'owner_admin' }
      },
      description: 'dangerous command',
      sessionId: 'external-channel-session'
    })

    renderSidebar(onResumeSession)

    expect(screen.getByText('Needs input')).toBeTruthy()
    expect(screen.getByText('External channel review')).toBeTruthy()
    expect(screen.getByText('Approval')).toBeTruthy()
    expect(screen.getByText('For owner/admin')).toBeTruthy()
    expect(screen.getByText('From Khristine via external-channel · org-studios / mesh-ko')).toBeTruthy()

    fireEvent.click(screen.getByText('External channel review'))

    expect(onResumeSession).toHaveBeenCalledWith('external-channel-session')
  })
})

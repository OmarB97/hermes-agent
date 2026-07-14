import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $compactingSessions } from '@/store/compaction'
import { $activeSessionId } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'

let handleEvent: ((event: RpcEvent) => void) | null = null

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
      const next = updater(current)
      sessionStateByRuntimeIdRef.current.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

const emit = (event: RpcEvent) => act(() => handleEvent!(event))

const startCompacting = () =>
  emit({
    payload: { kind: 'compacting', text: '🗜️ Compacting context — summarizing earlier conversation...' },
    session_id: SID,
    type: 'status.update'
  })

const isCompacting = () => SID in $compactingSessions.get()

/**
 * Auto-compaction is synchronous: the agent pauses, summarizes, then resumes.
 * The backend announces the START but sends no matching "done" notice, so the
 * only clears were turn-start / turn-complete / error.
 *
 * On a long agentic turn that pinned "Summarizing thread" for the entire REST
 * of the turn — observed live at 16+ minutes, timer counting, while the agent
 * was visibly running tools and writing text underneath it.
 */
describe('compaction indicator lifecycle', () => {
  beforeEach(() => {
    handleEvent = null
    $compactingSessions.set({})
    $activeSessionId.set(SID)
  })

  afterEach(() => {
    cleanup()
    $compactingSessions.set({})
    $activeSessionId.set(null)
    vi.restoreAllMocks()
  })

  it('raises the indicator when the backend announces compaction', async () => {
    await mountStream()
    startCompacting()

    expect(isCompacting()).toBe(true)
  })

  it.each([
    ['message.delta', { text: 'hello' }],
    ['thinking.delta', { text: 'hmm' }],
    ['reasoning.delta', { text: 'because' }],
    ['tool.start', { name: 'read_file' }],
    ['tool.complete', { name: 'read_file' }]
  ])('clears the indicator when the agent resumes with %s', async (type, payload) => {
    await mountStream()
    startCompacting()
    expect(isCompacting()).toBe(true)

    emit({ payload, session_id: SID, type } as RpcEvent)

    expect(isCompacting()).toBe(false)
  })

  it('keeps the indicator up until the agent actually resumes', async () => {
    await mountStream()
    startCompacting()

    // Unrelated traffic must not be mistaken for the agent resuming.
    emit({ payload: { kind: 'process' }, session_id: SID, type: 'status.update' })

    expect(isCompacting()).toBe(true)
  })

  it('does not clear compaction for a different session', async () => {
    await mountStream()
    startCompacting()

    emit({ payload: { text: 'hi' }, session_id: 'other-session', type: 'message.delta' })

    expect(isCompacting()).toBe(true)
  })

  it('still clears on turn completion (existing safety net)', async () => {
    await mountStream()
    startCompacting()

    emit({ payload: { text: 'done' }, session_id: SID, type: 'message.complete' })

    expect(isCompacting()).toBe(false)
  })
})

import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const ACTIVE_ID = 'active-runtime'
const BACKGROUND_ID = 'background-runtime'

let handleEvent: ((event: RpcEvent) => void) | null = null
let states: Map<string, ClientSessionState>

function Harness() {
  const activeSessionIdRef = useRef<string | null>(ACTIVE_ID)
  const sessionStateByRuntimeIdRef = useRef(states)
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater, storedSessionId) => {
      const current =
        sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState(storedSessionId ?? null)

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

describe('desktop live per-session context usage', () => {
  beforeEach(() => {
    handleEvent = null
    states = new Map([
      [ACTIVE_ID, createClientSessionState('stored-active')],
      [BACKGROUND_ID, createClientSessionState('stored-background')]
    ])
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  async function mount() {
    render(<Harness />)
    await waitFor(() => expect(handleEvent).not.toBeNull())
  }

  it('records token.usage before message.complete', async () => {
    await mount()

    act(() =>
      handleEvent!({
        payload: {
          context_length: 131_072,
          context_pct: 49.9,
          context_tokens: 65_432,
          input_tokens: 1_200,
          output_tokens: 34,
          total_tokens: 1_234
        },
        session_id: ACTIVE_ID,
        type: 'token.usage'
      })
    )

    expect(states.get(ACTIVE_ID)?.usage).toMatchObject({
      context_max: 131_072,
      context_percent: 49.9,
      context_used: 65_432,
      total: 1_234
    })
  })

  it('keeps the final live context snapshot after the turn completes', async () => {
    await mount()

    act(() =>
      handleEvent!({
        payload: {
          context_length: 131_072,
          context_pct: 75,
          context_tokens: 98_304,
          total_tokens: 124_000
        },
        session_id: ACTIVE_ID,
        type: 'token.usage'
      })
    )

    act(() =>
      handleEvent!({
        payload: { text: 'done' },
        session_id: ACTIVE_ID,
        type: 'message.complete'
      })
    )

    expect(states.get(ACTIVE_ID)?.busy).toBe(false)
    expect(states.get(ACTIVE_ID)?.usage).toMatchObject({
      context_max: 131_072,
      context_percent: 75,
      context_used: 98_304,
      total: 124_000
    })
  })

  it('updates a background session cache without overwriting the active session cache', async () => {
    states.get(ACTIVE_ID)!.usage = {
      calls: 1,
      context_max: 100_000,
      context_percent: 32,
      context_used: 32_000,
      input: 31_000,
      output: 1_000,
      total: 32_000
    }
    await mount()

    act(() =>
      handleEvent!({
        payload: { context_length: 100_000, context_pct: 88, context_tokens: 88_000 },
        session_id: BACKGROUND_ID,
        type: 'token.usage'
      })
    )

    expect(states.get(ACTIVE_ID)?.usage.context_percent).toBe(32)
    expect(states.get(BACKGROUND_ID)?.usage.context_percent).toBe(88)
  })
})

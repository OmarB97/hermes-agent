import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { chatMessageText, textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent, SessionInflightTurn } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'

let handleEvent: ((event: RpcEvent) => void) | null = null

let hydrateTurnOutcomeState:
  | ((sessionId: string, inflight?: null | SessionInflightTurn, messages?: ClientSessionState['messages']) => void)
  | null = null

let states: Map<string, ClientSessionState>

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

  states = sessionStateByRuntimeIdRef.current

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
    hydrateTurnOutcomeState = stream.hydrateTurnOutcomeState
  }, [stream.handleGatewayEvent, stream.hydrateTurnOutcomeState])

  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

const send = (event: RpcEvent) => act(() => handleEvent!(event))

describe('useMessageStream terminal turn outcomes', () => {
  beforeEach(() => {
    handleEvent = null
    hydrateTurnOutcomeState = null
    states = new Map()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('settles an interrupted partial exactly once and keeps the canonical system row', async () => {
    await mountStream()
    send({ payload: { turn_id: 'turn-1' }, session_id: SID, type: 'message.start' })

    const started = states.get(SID)!
    states.set(SID, {
      ...started,
      interrupted: true,
      streamId: 'assistant-partial',
      messages: [
        {
          id: 'assistant-partial',
          role: 'assistant',
          parts: [textPart('partial answer')],
          pending: true
        }
      ]
    })

    const outcome = {
      payload: {
        completed_at: 123,
        id: 'turn-1',
        status: 'cancelled',
        text: 'turn:cancelled · local-vllm/deepseek-v4-flash · user cancelled the turn'
      },
      session_id: SID,
      type: 'turn.outcome'
    }

    send(outcome)
    send(outcome)

    const settled = states.get(SID)!
    expect(settled.busy).toBe(false)
    expect(settled.awaitingResponse).toBe(false)
    expect(settled.streamId).toBeNull()
    expect(settled.messages.find(message => message.id === 'assistant-partial')?.pending).toBe(false)
    expect(settled.messages.filter(message => message.id === 'turn-outcome:turn-1')).toHaveLength(1)
    expect(chatMessageText(settled.messages.at(-1)!)).toContain('turn:cancelled')
  })

  it('records a delayed prior outcome without settling a newer active turn', async () => {
    await mountStream()
    send({ payload: { turn_id: 'old-turn' }, session_id: SID, type: 'message.start' })
    send({ payload: { turn_id: 'new-turn' }, session_id: SID, type: 'message.start' })

    send({
      payload: {
        id: 'old-turn',
        status: 'failed',
        text: 'turn:failed · provider/model · delayed prior failure'
      },
      session_id: SID,
      type: 'turn.outcome'
    })

    const current = states.get(SID)!
    expect(current.busy).toBe(true)
    expect(current.awaitingResponse).toBe(true)
    expect(current.messages.filter(message => message.id === 'turn-outcome:old-turn')).toHaveLength(1)
  })

  it('seeds the resumed active turn before delayed outcomes arrive', async () => {
    await mountStream()
    states.set(SID, {
      ...createClientSessionState(),
      awaitingResponse: true,
      busy: true
    })

    act(() => hydrateTurnOutcomeState!(SID, { streaming: true, turn_id: 'live-turn' }))

    send({
      payload: {
        id: 'old-turn',
        status: 'failed',
        text: 'turn:failed · provider/model · delayed prior failure'
      },
      session_id: SID,
      type: 'turn.outcome'
    })

    expect(states.get(SID)).toMatchObject({ awaitingResponse: true, busy: true })

    send({
      payload: {
        id: 'live-turn',
        status: 'completed',
        text: 'turn:completed · provider/model · response delivered'
      },
      session_id: SID,
      type: 'turn.outcome'
    })

    expect(states.get(SID)).toMatchObject({ awaitingResponse: false, busy: false })
  })

  it('seeds hydrated outcome ids so a concurrent replay stays deduplicated', async () => {
    await mountStream()

    const hydratedOutcome = {
      id: 'turn-outcome:stored-turn',
      role: 'system' as const,
      parts: [textPart('turn:failed · provider/model · stored failure')]
    }

    states.set(SID, createClientSessionState(null, [hydratedOutcome]))

    act(() => hydrateTurnOutcomeState!(SID, null, [hydratedOutcome]))
    send({
      payload: {
        id: 'stored-turn',
        status: 'failed',
        text: 'turn:failed · provider/model · stored failure'
      },
      session_id: SID,
      type: 'turn.outcome'
    })

    expect(states.get(SID)?.messages).toEqual([hydratedOutcome])
  })
})

import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { DELEGATED_CLARIFY_ANSWER } from '@/lib/delegated-spawn'
import { $clarifyRequests, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $sessionStates, clearAllSessionStates, publishSessionState } from '@/store/session-states'

import type { GatewayRequest } from '../../session/hooks/use-prompt-actions/utils'
import type { ClientSessionState } from '../../types'

import { answerClarifyForNobody, useDelegatedClarify } from './use-delegated-clarify'

// The guarantee: a chat started with `--delegated` never sits forever on a
// question nobody is there to answer. The contract in the prompt is supposed to
// stop the question being asked at all — this is what happens when it isn't.

const copy = {
  autoAnswered: (wait: string) => `no one answered within ${wait}`
} as Parameters<typeof answerClarifyForNobody>[0]['copy']

function delegatedSession(runtimeId: string, timeoutMs: number | null, storedSessionId = `stored-${runtimeId}`) {
  publishSessionState(runtimeId, {
    ...createClientSessionState(storedSessionId),
    delegatedTimeoutMs: timeoutMs
  })
}

// A JSON-RPC double whose calls stay assertable — GatewayRequest is generic in
// its return type, which a bare `vi.fn` cannot satisfy.
function gatewayDouble() {
  const mock = vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({}))

  return { mock, request: mock as unknown as GatewayRequest }
}

function stateDouble() {
  return vi.fn((sessionId: string, updater: (state: ClientSessionState) => ClientSessionState) => {
    const next = updater($sessionStates.get()[sessionId] ?? createClientSessionState())
    publishSessionState(sessionId, next)

    return next
  })
}

beforeEach(() => {
  clearAllSessionStates()
  clearClarifyRequest()
})

afterEach(() => {
  vi.useRealTimers()
  clearAllSessionStates()
  clearClarifyRequest()
  vi.restoreAllMocks()
})

// ------------------------------------------------------------------ the answer

describe('answerClarifyForNobody', () => {
  it('answers with the standing instruction, not an empty skip', async () => {
    const gateway = gatewayDouble()
    const updateSessionState = stateDouble()

    delegatedSession('s1', 120_000)
    setClarifyRequest({ requestId: 'r1', question: 'where should this go?', choices: null, sessionId: 's1' })

    const outcome = await answerClarifyForNobody({
      copy,
      request: $clarifyRequests.get().s1,
      requestGateway: gateway.request,
      timeoutMs: 120_000,
      updateSessionState
    })

    expect(outcome).toBe('answered')
    expect(gateway.mock).toHaveBeenCalledWith('clarify.respond', {
      request_id: 'r1',
      answer: DELEGATED_CLARIFY_ANSWER
    })
  })

  // Whoever reads this chat tomorrow has to be able to see that a decision was
  // made without them. A toast would be gone by then.
  it('writes a visible note into the transcript and drops the card', async () => {
    const updateSessionState = stateDouble()

    delegatedSession('s1', 120_000)
    setClarifyRequest({ requestId: 'r1', question: 'where?', choices: null, sessionId: 's1' })

    await answerClarifyForNobody({
      copy,
      request: $clarifyRequests.get().s1,
      requestGateway: gatewayDouble().request,
      timeoutMs: 120_000,
      updateSessionState
    })

    const state = $sessionStates.get().s1
    const note = state.messages.at(-1)!

    expect(note.role).toBe('system')
    expect(chatMessageText(note)).toBe('no one answered within 2m')
    expect(state.needsInput).toBe(false)
    expect($clarifyRequests.get().s1).toBeUndefined()
  })

  // The gateway's own 300s clarify timeout can get there first. The session is
  // not stuck in that case — it already moved on — so claiming we answered it
  // would put a false line in the transcript.
  it('says nothing when the gateway has already stopped waiting', async () => {
    const updateSessionState = stateDouble()

    delegatedSession('s1', 120_000)
    setClarifyRequest({ requestId: 'r1', question: 'where?', choices: null, sessionId: 's1' })

    const outcome = await answerClarifyForNobody({
      copy,
      request: $clarifyRequests.get().s1,
      requestGateway: (async () => {
        throw new Error('no pending answer request')
      }) as unknown as GatewayRequest,
      timeoutMs: 120_000,
      updateSessionState
    })

    expect(outcome).toBe('stale')
    expect(updateSessionState).not.toHaveBeenCalled()
    // The dead card still goes — it can never be answered now.
    expect($clarifyRequests.get().s1).toBeUndefined()
  })
})

// ------------------------------------------------------------------- the timer

describe('useDelegatedClarify', () => {
  function mount() {
    const gateway = gatewayDouble()
    const updateSessionState = stateDouble()

    renderHook(() => useDelegatedClarify({ requestGateway: gateway.request, updateSessionState }))

    return { requestGateway: gateway.mock, updateSessionState }
  }

  it('answers a delegated session once its wait elapses', async () => {
    vi.useFakeTimers()
    delegatedSession('s1', 120_000)

    const { requestGateway } = mount()

    await act(async () => {
      setClarifyRequest({ requestId: 'r1', question: 'where?', choices: null, sessionId: 's1' })
    })

    expect(requestGateway).not.toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(119_000)
    })

    // Not a millisecond early — a person who IS nearby still owns the question.
    expect(requestGateway).not.toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(1_000)
      await Promise.resolve()
    })

    expect(requestGateway).toHaveBeenCalledTimes(1)
  })

  // The whole feature has to stay invisible to ordinary use: answering a
  // person's question for them would be a bug, not a feature.
  it('never touches a chat that is not delegated', async () => {
    vi.useFakeTimers()
    delegatedSession('s1', null)

    const { requestGateway } = mount()

    await act(async () => {
      setClarifyRequest({ requestId: 'r1', question: 'where?', choices: null, sessionId: 's1' })
    })

    await act(async () => {
      vi.advanceTimersByTime(60 * 60_000)
      await Promise.resolve()
    })

    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('stands down when someone answers first', async () => {
    vi.useFakeTimers()
    delegatedSession('s1', 120_000)

    const { requestGateway } = mount()

    await act(async () => {
      setClarifyRequest({ requestId: 'r1', question: 'where?', choices: null, sessionId: 's1' })
    })

    await act(async () => {
      clearClarifyRequest('r1', 's1')
    })

    await act(async () => {
      vi.advanceTimersByTime(300_000)
      await Promise.resolve()
    })

    expect(requestGateway).not.toHaveBeenCalled()
  })

  it("honours each session's own wait", async () => {
    vi.useFakeTimers()
    delegatedSession('slow', 200_000)
    delegatedSession('quick', 10_000)

    const { requestGateway } = mount()

    await act(async () => {
      setClarifyRequest({ requestId: 'r-slow', question: 'a?', choices: null, sessionId: 'slow' })
      setClarifyRequest({ requestId: 'r-quick', question: 'b?', choices: null, sessionId: 'quick' })
    })

    await act(async () => {
      vi.advanceTimersByTime(10_000)
      await Promise.resolve()
    })

    expect(requestGateway).toHaveBeenCalledTimes(1)
    expect(requestGateway.mock.calls[0][1]).toMatchObject({ request_id: 'r-quick' })
  })

  // One deadline, one answer — a re-render mid-wait must not restart the clock
  // or stack a second timer onto the same question.
  it('answers a question exactly once', async () => {
    vi.useFakeTimers()
    delegatedSession('s1', 120_000)

    const { requestGateway } = mount()

    await act(async () => {
      setClarifyRequest({ requestId: 'r1', question: 'where?', choices: null, sessionId: 's1' })
    })

    // Something unrelated republishes the session — a streamed token would.
    await act(async () => {
      publishSessionState('s1', { ...$sessionStates.get().s1, busy: true })
      vi.advanceTimersByTime(120_000)
      await Promise.resolve()
    })

    expect(requestGateway).toHaveBeenCalledTimes(1)
  })

  it('a closed window does not answer on its way out', async () => {
    vi.useFakeTimers()
    delegatedSession('s1', 120_000)

    const gateway = gatewayDouble()

    const { unmount } = renderHook(() =>
      useDelegatedClarify({ requestGateway: gateway.request, updateSessionState: stateDouble() })
    )

    await act(async () => {
      setClarifyRequest({ requestId: 'r1', question: 'where?', choices: null, sessionId: 's1' })
    })

    unmount()

    await act(async () => {
      vi.advanceTimersByTime(300_000)
      await Promise.resolve()
    })

    expect(gateway.mock).not.toHaveBeenCalled()
  })
})

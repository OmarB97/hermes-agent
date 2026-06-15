import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { HermesGateway } from '@/hermes'
import { $pendingPromptAttention } from '@/store/prompt-attention'
import { $secretRequests, clearAllPrompts } from '@/store/prompts'
import type { RpcEvent } from '@/types/hermes'

import type { ClientSessionState } from '../../types'

import { useMessageStream } from './use-message-stream'

type MessageStreamApi = ReturnType<typeof useMessageStream>
type Listener = (event: unknown) => void

interface HarnessProps {
  activeSessionId: string | null
  onReady: (api: MessageStreamApi) => void
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

function emptySessionState(storedSessionId: string | null = null): ClientSessionState {
  return {
    awaitingResponse: false,
    branch: '',
    busy: false,
    cwd: '',
    fast: false,
    interrupted: false,
    messages: [],
    model: '',
    needsInput: false,
    pendingBranchGroup: null,
    personality: '',
    provider: '',
    reasoningEffort: '',
    sawAssistantPayload: false,
    serviceTier: '',
    storedSessionId,
    streamId: null,
    turnStartedAt: null,
    yolo: false
  }
}

function Harness({ activeSessionId, onReady, updateSessionState }: HarnessProps) {
  const activeSessionIdRef: MutableRefObject<string | null> = { current: activeSessionId }

  const api = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(),
    queryClient: new QueryClient(),
    refreshHermesConfig: vi.fn(),
    refreshSessions: vi.fn(),
    updateSessionState
  })

  onReady(api)

  return null
}

function externalChannelSecretEvent(sessionId: string): RpcEvent {
  return {
    payload: {
      env_var: 'HERMES_CHANNEL_TOKEN',
      mesh_id: 'mesh-smoke',
      org_id: 'org-smoke',
      prompt: 'Paste the channel token',
      request_id: 'secret-external-1',
      requested_by: {
        display_name: 'Ari Reviewer',
        platform: 'external-channel',
        principal_id: 'principal-ari'
      },
      requested_via: 'external-channel',
      target_audience: { kind: 'owner_admin' }
    },
    session_id: sessionId,
    type: 'secret.request'
  }
}

class FakeGatewayWebSocket {
  static CLOSED = 3
  static OPEN = 1
  static instances: FakeGatewayWebSocket[] = []

  readonly sent: string[] = []
  readyState = 0
  private readonly listeners: Record<string, Set<Listener>> = {}

  constructor(readonly url: string) {
    FakeGatewayWebSocket.instances.push(this)
    queueMicrotask(() => {
      this.readyState = FakeGatewayWebSocket.OPEN
      this.emit('open', {})
    })
  }

  addEventListener(type: string, listener: Listener) {
    ;(this.listeners[type] ??= new Set()).add(listener)
  }

  removeEventListener(type: string, listener: Listener) {
    this.listeners[type]?.delete(listener)
  }

  send(data: string) {
    this.sent.push(data)
  }

  close() {
    this.readyState = FakeGatewayWebSocket.CLOSED
    this.emit('close', {})
  }

  emitMessage(frame: unknown) {
    const data = typeof frame === 'string' ? frame : JSON.stringify(frame)
    this.emit('message', { data })
  }

  private emit(type: string, event: unknown) {
    for (const listener of this.listeners[type] ?? []) {
      listener(event)
    }
  }
}

const originalWebSocket = globalThis.WebSocket

describe('useMessageStream channel prompt events', () => {
  beforeEach(() => {
    FakeGatewayWebSocket.instances = []
    ;(globalThis as { WebSocket: unknown }).WebSocket = FakeGatewayWebSocket
    clearAllPrompts()
  })

  afterEach(() => {
    cleanup()
    ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
    clearAllPrompts()
  })

  it('parks a gateway-originated prompt for a non-active external-channel session', () => {
    let api!: MessageStreamApi
    const states = new Map<string, ClientSessionState>()

    const updateSessionState = vi.fn(
      (
        sessionId: string,
        updater: (state: ClientSessionState) => ClientSessionState,
        storedSessionId?: string | null
      ) => {
        const current = states.get(sessionId) ?? emptySessionState(storedSessionId ?? null)
        const next = updater(current)
        states.set(sessionId, next)

        return next
      }
    )

    render(
      <Harness
        activeSessionId="runtime-local-1"
        onReady={nextApi => {
          api = nextApi
        }}
        updateSessionState={updateSessionState}
      />
    )

    act(() => {
      api.handleGatewayEvent(externalChannelSecretEvent('runtime-external-1'))
    })

    const secret = $secretRequests.get()['runtime-external-1']
    expect(secret).toMatchObject({
      envVar: 'HERMES_CHANNEL_TOKEN',
      prompt: 'Paste the channel token',
      requestId: 'secret-external-1',
      sessionId: 'runtime-external-1'
    })
    expect(secret.context).toMatchObject({
      meshId: 'mesh-smoke',
      orgId: 'org-smoke',
      requestedBy: {
        displayName: 'Ari Reviewer',
        platform: 'external-channel',
        principalId: 'principal-ari'
      },
      requestedVia: 'external-channel',
      targetAudience: { kind: 'owner_admin' }
    })

    expect(states.get('runtime-external-1')?.needsInput).toBe(true)
    expect($pendingPromptAttention.get()).toEqual([
      expect.objectContaining({
        detail: 'HERMES_CHANNEL_TOKEN',
        kind: 'secret',
        sessionId: 'runtime-external-1'
      })
    ])
  })

  it('routes a JSON-RPC WebSocket event frame into prompt attention', async () => {
    let api!: MessageStreamApi
    const states = new Map<string, ClientSessionState>()

    const updateSessionState = vi.fn(
      (
        sessionId: string,
        updater: (state: ClientSessionState) => ClientSessionState,
        storedSessionId?: string | null
      ) => {
        const current = states.get(sessionId) ?? emptySessionState(storedSessionId ?? null)
        const next = updater(current)
        states.set(sessionId, next)

        return next
      }
    )

    render(
      <Harness
        activeSessionId="runtime-local-1"
        onReady={nextApi => {
          api = nextApi
        }}
        updateSessionState={updateSessionState}
      />
    )

    const gateway = new HermesGateway()
    const offGatewayEvent = gateway.onEvent(api.handleGatewayEvent)

    await act(async () => {
      await gateway.connect('ws://gateway.test/api/ws')
    })

    const socket = FakeGatewayWebSocket.instances[0]

    if (!socket) {
      throw new Error('expected fake gateway socket to connect')
    }

    expect(socket.url).toBe('ws://gateway.test/api/ws')

    act(() => {
      socket.emitMessage({
        jsonrpc: '2.0',
        method: 'event',
        params: externalChannelSecretEvent('runtime-external-ws')
      })
    })

    expect($secretRequests.get()['runtime-external-ws']).toMatchObject({
      envVar: 'HERMES_CHANNEL_TOKEN',
      requestId: 'secret-external-1',
      sessionId: 'runtime-external-ws'
    })
    expect(states.get('runtime-external-ws')?.needsInput).toBe(true)
    expect($pendingPromptAttention.get()).toEqual([
      expect.objectContaining({
        detail: 'HERMES_CHANNEL_TOKEN',
        kind: 'secret',
        sessionId: 'runtime-external-ws'
      })
    ])

    offGatewayEvent()
    gateway.close()
  })
})

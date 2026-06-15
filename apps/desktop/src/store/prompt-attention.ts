import { computed } from 'nanostores'

import { $clarifyRequests } from './clarify'
import {
  $approvalRequests,
  $secretRequests,
  $sudoRequests,
  type ApprovalRequest,
  type PromptRequestContext,
  type SecretRequest,
  type SudoRequest
} from './prompts'

export type PendingPromptKind = 'approval' | 'clarify' | 'secret' | 'sudo'

export interface PendingPromptAttention {
  context?: PromptRequestContext
  detail: string
  id: string
  kind: PendingPromptKind
  sessionId: null | string
}

const keyFor = (sessionId: null | string | undefined): string => sessionId ?? ''

function promptId(kind: PendingPromptKind, sessionId: null | string, requestId?: string): string {
  return [kind, keyFor(sessionId), requestId || 'session'].join(':')
}

function approvalAttention(request: ApprovalRequest): PendingPromptAttention {
  return {
    context: request.context,
    detail: request.command || request.description,
    id: promptId('approval', request.sessionId, request.command),
    kind: 'approval',
    sessionId: request.sessionId
  }
}

function sudoAttention(request: SudoRequest): PendingPromptAttention {
  return {
    context: request.context,
    detail: request.requestId,
    id: promptId('sudo', request.sessionId, request.requestId),
    kind: 'sudo',
    sessionId: request.sessionId
  }
}

function secretAttention(request: SecretRequest): PendingPromptAttention {
  return {
    context: request.context,
    detail: request.envVar || request.prompt,
    id: promptId('secret', request.sessionId, request.requestId),
    kind: 'secret',
    sessionId: request.sessionId
  }
}

export const $pendingPromptAttention = computed(
  [$clarifyRequests, $approvalRequests, $sudoRequests, $secretRequests],
  (clarifyRequests, approvalRequests, sudoRequests, secretRequests): PendingPromptAttention[] => [
    ...Object.values(clarifyRequests).map(request => ({
      context: request.context,
      detail: request.question,
      id: promptId('clarify', request.sessionId, request.requestId),
      kind: 'clarify' as const,
      sessionId: request.sessionId
    })),
    ...Object.values(approvalRequests).map(approvalAttention),
    ...Object.values(sudoRequests).map(sudoAttention),
    ...Object.values(secretRequests).map(secretAttention)
  ]
)

export const $pendingPromptSessionIds = computed([$pendingPromptAttention], prompts => {
  const seen = new Set<string>()
  const out: string[] = []

  for (const prompt of prompts) {
    if (prompt.sessionId && !seen.has(prompt.sessionId)) {
      seen.add(prompt.sessionId)
      out.push(prompt.sessionId)
    }
  }

  return out
})

import { afterEach, describe, expect, it } from 'vitest'

import { clearClarifyRequest, setClarifyRequest } from './clarify'
import { $pendingPromptAttention, $pendingPromptSessionIds } from './prompt-attention'
import { clearAllPrompts, normalizePromptContext, setApprovalRequest, setSecretRequest, setSudoRequest } from './prompts'

afterEach(() => {
  clearClarifyRequest()
  clearAllPrompts()
})

describe('pending prompt attention', () => {
  it('combines every blocking prompt kind into one attention list', () => {
    setClarifyRequest({
      requestId: 'clarify-1',
      question: 'Which workspace?',
      choices: null,
      sessionId: 's1'
    })
    setApprovalRequest({
      command: 'rm -rf /tmp/demo',
      description: 'dangerous command',
      sessionId: 's2'
    })
    setSudoRequest({ requestId: 'sudo-1', sessionId: 's2' })
    setSecretRequest({ requestId: 'secret-1', envVar: 'HERMES_TOKEN', prompt: 'Token', sessionId: 's3' })

    expect($pendingPromptAttention.get().map(prompt => [prompt.kind, prompt.sessionId])).toEqual([
      ['clarify', 's1'],
      ['approval', 's2'],
      ['sudo', 's2'],
      ['secret', 's3']
    ])
    expect($pendingPromptSessionIds.get()).toEqual(['s1', 's2', 's3'])
  })

  it('preserves actor and audience metadata for multi-human channel workflows', () => {
    setApprovalRequest({
      command: 'delete cloud channel',
      context: normalizePromptContext({
        mesh_id: 'mesh-ko',
        org_id: 'org-studios',
        requested_by: { display_name: 'Khristine', platform: 'external-channel', principal_id: 'principal-k' },
        requested_via: 'external-channel',
        target_audience: { kind: 'owner_admin' }
      }),
      description: 'dangerous command',
      sessionId: 's1'
    })

    expect($pendingPromptAttention.get()[0]).toMatchObject({
      context: {
        meshId: 'mesh-ko',
        orgId: 'org-studios',
        requestedBy: { displayName: 'Khristine', platform: 'external-channel', principalId: 'principal-k' },
        requestedVia: 'external-channel',
        targetAudience: { kind: 'owner_admin' }
      },
      kind: 'approval',
      sessionId: 's1'
    })
  })
})

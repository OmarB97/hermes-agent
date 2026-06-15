import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $approvalRequest,
  $secretRequest,
  $sudoRequest,
  clearAllPrompts,
  clearApprovalRequest,
  clearSecretRequest,
  clearSudoRequest,
  normalizePromptContext,
  setApprovalRequest,
  setSecretRequest,
  setSudoRequest
} from './prompts'
import { $activeSessionId } from './session'

// Prompts are parked per-session; the exported $*Request views are scoped to the
// active session, so each test focuses the session it's asserting on.
beforeEach(() => {
  $activeSessionId.set('s1')
})

afterEach(() => {
  clearAllPrompts()
  $activeSessionId.set(null)
})

describe('approval prompt store', () => {
  it('holds the active session-keyed approval request', () => {
    setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'recursive delete', sessionId: 's1' })

    expect($approvalRequest.get()).toEqual({
      command: 'rm -rf /tmp/x',
      description: 'recursive delete',
      sessionId: 's1'
    })
  })

  it('parks a background session prompt out of the active view', () => {
    setApprovalRequest({ command: 'x', description: 'd', sessionId: 's2' })

    // Not visible while s1 is focused …
    expect($approvalRequest.get()).toBeNull()

    // … but surfaces once the user switches to the session that raised it.
    $activeSessionId.set('s2')
    expect($approvalRequest.get()?.sessionId).toBe('s2')
  })

  it('clears the active session prompt', () => {
    setApprovalRequest({ command: 'x', description: 'd', sessionId: 's1' })
    clearApprovalRequest('s1')

    expect($approvalRequest.get()).toBeNull()
  })

  it('carries allowPermanent so the bar can hide "Always allow"', () => {
    setApprovalRequest({ allowPermanent: false, command: 'curl x | bash', description: 'content-security', sessionId: 's1' })

    expect($approvalRequest.get()?.allowPermanent).toBe(false)
  })

  it('carries multi-entity context for shared channel workflows', () => {
    const context = normalizePromptContext(
      {
        mesh_id: 'mesh-ko',
        org_id: 'org-studios',
        requested_by: { display_name: 'Khristine', platform: 'external-channel', principal_id: 'principal-k' },
        requested_via: 'external-channel',
        target_audience: { kind: 'owner_admin' }
      },
      { targetAudience: { kind: 'owner_admin' } }
    )

    setApprovalRequest({
      command: 'delete channel',
      context,
      description: 'dangerous command',
      sessionId: 's1'
    })

    expect($approvalRequest.get()?.context).toMatchObject({
      meshId: 'mesh-ko',
      orgId: 'org-studios',
      requestedBy: { displayName: 'Khristine', platform: 'external-channel', principalId: 'principal-k' },
      requestedVia: 'external-channel',
      targetAudience: { kind: 'owner_admin' }
    })
  })
})

describe('sudo prompt store', () => {
  it('clears only when the request id matches the in-flight prompt', () => {
    setSudoRequest({ requestId: 'abc', sessionId: 's1' })

    // A stale clear for a different request must NOT drop the live prompt —
    // otherwise a late response to a prior sudo ask would dismiss the current
    // one and leave the agent blocked.
    clearSudoRequest('s1', 'stale')
    expect($sudoRequest.get()).toEqual({ requestId: 'abc', sessionId: 's1' })

    clearSudoRequest('s1', 'abc')
    expect($sudoRequest.get()).toBeNull()
  })

  it('clears unconditionally when no request id is given', () => {
    setSudoRequest({ requestId: 'abc', sessionId: 's1' })
    clearSudoRequest('s1')

    expect($sudoRequest.get()).toBeNull()
  })
})

describe('secret prompt store', () => {
  it('carries env var and prompt, and clears on id match', () => {
    setSecretRequest({ requestId: 'r1', envVar: 'OPENAI_API_KEY', prompt: 'Paste your key', sessionId: 's1' })

    expect($secretRequest.get()).toEqual({
      requestId: 'r1',
      envVar: 'OPENAI_API_KEY',
      prompt: 'Paste your key',
      sessionId: 's1'
    })

    clearSecretRequest('s1', 'mismatch')
    expect($secretRequest.get()).not.toBeNull()

    clearSecretRequest('s1', 'r1')
    expect($secretRequest.get()).toBeNull()
  })
})

describe('clearAllPrompts', () => {
  it('drops every kind for one session at once (turn end / interrupt)', () => {
    setApprovalRequest({ command: 'x', description: 'd', sessionId: 's1' })
    setSudoRequest({ requestId: 'abc', sessionId: 's1' })
    setSecretRequest({ requestId: 'r1', envVar: 'E', prompt: 'p', sessionId: 's1' })

    clearAllPrompts('s1')

    expect($approvalRequest.get()).toBeNull()
    expect($sudoRequest.get()).toBeNull()
    expect($secretRequest.get()).toBeNull()
  })

  it('leaves other sessions parked prompts intact', () => {
    setApprovalRequest({ command: 'x', description: 'd', sessionId: 's1' })
    setApprovalRequest({ command: 'y', description: 'e', sessionId: 's2' })

    clearAllPrompts('s1')

    $activeSessionId.set('s2')
    expect($approvalRequest.get()?.command).toBe('y')
  })
})

describe('normalizePromptContext', () => {
  it('normalizes platform, principal, and audience aliases', () => {
    expect(
      normalizePromptContext({
        actor: { name: 'Omar', device_id: 'mac', user_id: '42' },
        audience: 'owner_or_admin',
        via: 'external-channel'
      })
    ).toMatchObject({
      requestedBy: { displayName: 'Omar', deviceId: 'mac', platformUserId: '42' },
      requestedVia: 'external-channel',
      targetAudience: { kind: 'owner_admin' }
    })
  })

  it('falls back to a caller-provided audience when the payload is sparse', () => {
    expect(normalizePromptContext({}, { targetAudience: { kind: 'originator' } })).toEqual({
      targetAudience: { kind: 'originator' }
    })
  })
})

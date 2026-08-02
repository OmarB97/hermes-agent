import { describe, expect, it } from 'vitest'

import {
  applyDelegationContract,
  DELEGATED_CLARIFY_ANSWER,
  DELEGATED_CLARIFY_TIMEOUT_MAX_MS,
  DELEGATED_CLARIFY_TIMEOUT_MIN_MS,
  DELEGATED_CLARIFY_TIMEOUT_MS,
  DELEGATION_CONTRACT,
  formatDelegatedWait,
  normalizeDelegatedTimeoutMs
} from './delegated-spawn'

// These are the rules that make an unattended spawn finish by itself. The
// contract is what we hope works; the timeout and the answer are what has to
// hold when it doesn't.

describe('applyDelegationContract', () => {
  it('puts the contract ahead of the brief', () => {
    const result = applyDelegationContract('audit the auth flow')

    expect(result.startsWith(DELEGATION_CONTRACT)).toBe(true)
    expect(result.endsWith('audit the auth flow')).toBe(true)
    // Order is the whole point: rules first, then the thing they govern.
    expect(result.indexOf(DELEGATION_CONTRACT)).toBeLessThan(result.indexOf('audit the auth flow'))
  })

  it('trims the brief but keeps it intact', () => {
    expect(applyDelegationContract('  go  ').endsWith('\n\ngo')).toBe(true)
  })

  it('leaves an empty brief empty rather than sending a contract with no work', () => {
    expect(applyDelegationContract('   ')).toBe('')
  })

  // The three things the DS4 session on 2026-08-01 needed to be told, and was
  // not: don't ask, pick a safe default, and deliver here.
  it('covers not asking, the safe default, and delivering in the chat', () => {
    expect(DELEGATION_CONTRACT).toMatch(/do not ask/i)
    expect(DELEGATION_CONTRACT).toMatch(/safest option you can undo/i)
    expect(DELEGATION_CONTRACT).toMatch(/put the result in this chat/i)
  })

  it('asks for assumptions and a DONE/BLOCKER ending', () => {
    expect(DELEGATION_CONTRACT).toMatch(/assum/i)
    expect(DELEGATION_CONTRACT).toContain('DONE')
    expect(DELEGATION_CONTRACT).toContain('BLOCKER')
  })
})

describe('normalizeDelegatedTimeoutMs', () => {
  it('keeps a sane request', () => {
    expect(normalizeDelegatedTimeoutMs(30_000)).toBe(30_000)
  })

  it.each([undefined, null, 'soon', Number.NaN, Number.POSITIVE_INFINITY, 0, -1])(
    'falls back to the default for %p',
    value => {
      expect(normalizeDelegatedTimeoutMs(value)).toBe(DELEGATED_CLARIFY_TIMEOUT_MS)
    }
  )

  // Bad input must never leave the session unguarded — the answer is always a
  // usable deadline, never "no deadline".
  it('clamps rather than rejects', () => {
    expect(normalizeDelegatedTimeoutMs(1)).toBe(DELEGATED_CLARIFY_TIMEOUT_MIN_MS)
    expect(normalizeDelegatedTimeoutMs(999 * 60_000)).toBe(DELEGATED_CLARIFY_TIMEOUT_MAX_MS)
  })

  it('rounds a fractional millisecond', () => {
    expect(normalizeDelegatedTimeoutMs(30_000.6)).toBe(30_001)
  })
})

describe('formatDelegatedWait', () => {
  it.each([
    [120_000, '2m'],
    [45_000, '45s'],
    [90_000, '1m 30s'],
    [5_000, '5s'],
    [100, '1s']
  ])('renders %dms as %s', (ms, expected) => {
    expect(formatDelegatedWait(ms)).toBe(expected)
  })
})

describe('DELEGATED_CLARIFY_ANSWER', () => {
  // Empty is what Skip sends, and the agent cannot tell it from a person who
  // declined to say — it learns nothing and is free to ask again.
  it('is never empty', () => {
    expect(DELEGATED_CLARIFY_ANSWER.trim().length).toBeGreaterThan(0)
  })

  it('says nobody is there, names the rule to apply, and closes the loop', () => {
    expect(DELEGATED_CLARIFY_ANSWER).toMatch(/no one is available/i)
    expect(DELEGATED_CLARIFY_ANSWER).toMatch(/safest option you can undo/i)
    expect(DELEGATED_CLARIFY_ANSWER).toMatch(/do not ask again/i)
  })
})

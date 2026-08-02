import { expect, it } from 'vitest'

import { hasSessionOverrides } from './session-overrides'

// This predicate is the gate that decides whether a programmatic chat's
// overrides reach `session.create` at all. It exists so callers do not
// hand-roll `a || b || c` field lists — one of those, in use-spawn-bridge,
// is exactly why `--toolsets` silently did nothing for its whole first life
// (#298, removed in #315, rewired here). A field added to
// SessionCreateOverrides and forgotten here is that bug again.

it('is false for nothing to override', () => {
  expect(hasSessionOverrides(undefined)).toBe(false)
  expect(hasSessionOverrides(null)).toBe(false)
  expect(hasSessionOverrides({})).toBe(false)
})

it('is true for any single override, including toolsets alone', () => {
  expect(hasSessionOverrides({ model: 'm' })).toBe(true)
  expect(hasSessionOverrides({ provider: 'p' })).toBe(true)
  expect(hasSessionOverrides({ profile: 'work' })).toBe(true)
  expect(hasSessionOverrides({ toolsets: ['file'] })).toBe(true)
})

// An empty array is not a pin. Passing it through would send `toolsets: []`,
// which the gateway refuses outright — turning "no opinion" into a failed spawn.
it('does not count an empty toolsets array as an override', () => {
  expect(hasSessionOverrides({ toolsets: [] })).toBe(false)
})

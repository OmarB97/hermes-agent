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

// A goal has nothing else it depends on to matter — a spawn naming only a
// goal must still reach session.create, or the backend never starts the loop.
it('is true when only goal is set', () => {
  expect(hasSessionOverrides({ goal: 'ship the release' })).toBe(true)
})

// goalMaxTurns is a budget FOR a goal, not a standing objective of its own —
// alone it has no loop to attach to, so it must not count as an override.
it('does not count goalMaxTurns alone as an override', () => {
  expect(hasSessionOverrides({ goalMaxTurns: 40 })).toBe(false)
})

// A declared command allowlist that never reaches session.create is the whole
// bug this predicate exists to prevent, with a worse consequence than a wrong
// toolset: the session was spawned unattended precisely so it would not stall
// on an approval, and it would stall anyway.
it('is true when a declared command allowlist is set', () => {
  expect(hasSessionOverrides({ allowedCommands: ['godot'], allowedCommandRoot: '/w' })).toBe(true)
})

// An empty list declares nothing, and a root scopes nothing on its own — the
// backend refuses both, so neither is worth a create call.
it('does not count an empty allowlist or a bare root as an override', () => {
  expect(hasSessionOverrides({ allowedCommands: [] })).toBe(false)
  expect(hasSessionOverrides({ allowedCommandRoot: '/w' })).toBe(false)
})

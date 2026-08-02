/**
 * Per-session overrides for a programmatically started chat.
 *
 * A new chat normally takes its model / provider / profile from the composer's
 * live selection (see `desktopSessionCreateParams`). A spawn asked for by a
 * local CLI has to be able to name a different model WITHOUT moving the user's
 * selection: `setCurrentModel` persists to storage, so mutating it to steer one
 * spawned session would silently change the default for every chat the user
 * starts afterwards. Passing the choice down instead keeps the override scoped
 * to the one session it was meant for.
 *
 * Every field is optional and an omitted field means "use the current
 * selection", so a spawn with no overrides is byte-identical to a typed chat.
 */
export interface SessionCreateOverrides {
  model?: string
  provider?: string
  profile?: string
}

/** True when at least one override is actually set (empty objects are noise). */
export function hasSessionOverrides(overrides: null | SessionCreateOverrides | undefined): boolean {
  return Boolean(overrides && (overrides.model || overrides.provider || overrides.profile))
}

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
  /** Toolsets to pin for this session only. The gateway validates the names and
   *  fails the create if any does not resolve, so a pinned session either has
   *  the tools that were asked for or does not exist. */
  toolsets?: string[]
  /** Standing objective for this session. The gateway starts its goal loop
   *  against this at create time, so the first turn is already judged — rather
   *  than the loop only existing once someone has typed `/goal`. */
  goal?: string
  /** Turn budget for `goal`. Omitted means the backend's configured default. */
  goalMaxTurns?: number
  /** Commands this session may run without a human when the approval
   *  classifier cannot be reached. Without this the classifier's failure means
   *  "ask a human", which in an unattended run is a stall rather than a
   *  decision. The gateway refuses a create it cannot store, so a session that
   *  asked for an allowlist either has one or does not exist. */
  allowedCommands?: string[]
  /** Absolute directory `allowedCommands` are scoped to — required with it,
   *  and meaningless without it. */
  allowedCommandRoot?: string
}

/** True when at least one override is actually set (empty objects are noise). */
export function hasSessionOverrides(overrides: null | SessionCreateOverrides | undefined): boolean {
  return Boolean(
    overrides &&
      (overrides.model ||
        overrides.provider ||
        overrides.profile ||
        overrides.toolsets?.length ||
        overrides.goal ||
        overrides.allowedCommands?.length)
  )
}

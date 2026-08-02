import type { ComposerAttachment } from '@/store/composer'

/** Probe a local path for existence/readability. Injected so the preflight is
 *  testable without the Electron bridge. */
export type PathProbe = (path: string) => Promise<boolean>

/**
 * A queued entry snapshots its attachments by LOCAL path at enqueue time — the
 * bytes are only staged into the session at send time. Between those two
 * moments the file can be deleted, renamed, or (for a screenshot parked under
 * `~/Library/Application Support/…`) belong to a profile home that no longer
 * exists. Only unstaged file/image chips carry a local path worth probing;
 * terminal/url refs and already-staged chips have nothing on this disk to
 * check.
 */
export const localAttachmentPaths = (attachments: readonly ComposerAttachment[]): string[] =>
  attachments
    .filter(a => Boolean(a) && (a.kind === 'image' || a.kind === 'file') && Boolean(a.path) && !a.attachedSessionId)
    .map(a => a.path!)

/**
 * Return the first referenced attachment path that is no longer on disk, or
 * null when every attachment is still resolvable.
 *
 * Run this BEFORE a drain submits, not after: the submit pipeline paints its
 * optimistic user bubble as soon as it has a session id, so discovering a dead
 * attachment inside the send is what leaves duplicate bubbles and error rows
 * behind on every retry. A probe that itself fails (bridge missing) reports
 * "present" — a preflight may never be the thing that dead-letters a good
 * entry.
 */
export async function firstMissingAttachmentPath(
  attachments: readonly ComposerAttachment[],
  probe: PathProbe | undefined
): Promise<null | string> {
  if (!probe) {
    return null
  }

  for (const path of localAttachmentPaths(attachments)) {
    let exists = true

    try {
      exists = await probe(path)
    } catch {
      exists = true
    }

    if (!exists) {
      return path
    }
  }

  return null
}

/** The live probe: the desktop bridge when it exists, undefined in a browser
 *  test/preview context (where the preflight then no-ops). */
export const desktopPathProbe = (): PathProbe | undefined => window.hermesDesktop?.pathExists

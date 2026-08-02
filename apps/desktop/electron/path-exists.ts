import { resolveReadableFileForIpc } from './hardening'

interface PathExistsDeps {
  resolveReadableFile?: typeof resolveReadableFileForIpc
}

/**
 * Answer "is this local file still there and readable?" — nothing more.
 *
 * The renderer needs this to preflight a QUEUED prompt's attachments before
 * sending it: a turn queued weeks ago references files by local path, and a
 * path that has since gone away turns the entry into a poison pill that fails
 * every send forever. Existence is the one fact the renderer cannot learn on
 * its own without reading the whole file through `readFileDataUrl` (16 MB cap,
 * full base64 load) — this is the cheap question.
 *
 * Goes through the same hardened resolver as every other file IPC, so unsafe
 * path syntax and Windows device paths are rejected before touching the disk.
 * Sensitivity blocking is off on purpose: we never read a byte, and reporting a
 * present-but-sensitive file as "missing" would dead-letter a queued turn with
 * a wrong reason. Any failure at all answers `false` — the caller treats that
 * as "not usable", which is the only actionable meaning.
 */
export async function pathExistsForIpc(filePath: unknown, deps: PathExistsDeps = {}): Promise<boolean> {
  const resolveReadableFile = deps.resolveReadableFile ?? resolveReadableFileForIpc
  const raw = typeof filePath === 'string' ? filePath.trim() : ''

  if (!raw) {
    return false
  }

  try {
    await resolveReadableFile(raw, { purpose: 'Attachment check', blockSensitive: false })

    return true
  } catch {
    return false
  }
}

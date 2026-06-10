import { type DragEvent as ReactDragEvent, useCallback, useRef, useState } from 'react'

import {
  dragHasSession,
  dragSessionIsArchived,
  dragSessionIsPinned,
  readSessionDrag,
  type SessionDragPayload
} from '@/app/chat/composer/inline-refs'

/** Drag-state flags readable during dragover (they ride as marker MIME types;
 * the payload itself is sealed until drop). */
export interface SessionDragFlags {
  pinned: boolean
  archived: boolean
}

interface SessionDropZoneOptions {
  /** Which drags this zone acts on — e.g. Pinned takes unpinned+unarchived
   * rows, Sessions takes pinned (unpin) or archived (restore) rows, Archived
   * takes anything not already archived. */
  accepts: (flags: SessionDragFlags) => boolean
  /** The drop event rides along so handlers can resolve the drop position
   * (see {@link sessionDropAnchor}). */
  onDropSession: (session: SessionDragPayload, event: ReactDragEvent) => void
}

export interface SessionDropAnchor {
  /** Live session id of the row under the pointer. */
  sessionId: string
  /** True when the pointer sat in the row's top half → insert before it. */
  before: boolean
}

/**
 * Resolve the session row under a drop point (rows carry `data-session-id`)
 * so drop handlers can insert at the pointer position instead of appending.
 * Null for drops on the section header or empty space.
 */
export function sessionDropAnchor(event: ReactDragEvent): null | SessionDropAnchor {
  const target = event.target as HTMLElement | null
  const row = target?.closest?.('[data-session-id]') as HTMLElement | null
  const sessionId = row?.dataset.sessionId

  if (!row || !sessionId) {
    return null
  }

  const rect = row.getBoundingClientRect()

  return { before: event.clientY < rect.top + rect.height / 2, sessionId }
}

/**
 * Native drop target for sidebar session rows — the row body's drag already
 * carries `application/x-hermes-session`, so dropping it on the Pinned /
 * Sessions section headers-or-bodies pins and unpins without the context menu.
 *
 * A zone only engages for drags it would act on (Pinned accepts unpinned rows,
 * Sessions accepts pinned rows); other drags never preventDefault, so the
 * cursor honestly reports "no drop here". The enter/leave depth counter keeps
 * nested children from flickering the highlight, mirroring use-file-drop-zone.
 *
 * Spread `dropHandlers` onto the section container; style off `active`.
 */
export function useSessionDropZone({ accepts: acceptsFlags, onDropSession }: SessionDropZoneOptions) {
  const [active, setActive] = useState(false)
  const depth = useRef(0)

  const accepts = useCallback(
    (event: ReactDragEvent) =>
      dragHasSession(event.dataTransfer) &&
      acceptsFlags({
        archived: dragSessionIsArchived(event.dataTransfer),
        pinned: dragSessionIsPinned(event.dataTransfer)
      }),
    [acceptsFlags]
  )

  const reset = useCallback(() => {
    depth.current = 0
    setActive(false)
  }, [])

  const onDragEnter = useCallback(
    (event: ReactDragEvent) => {
      if (!accepts(event)) {
        return
      }

      event.preventDefault()
      depth.current += 1
      setActive(true)
    },
    [accepts]
  )

  const onDragOver = useCallback(
    (event: ReactDragEvent) => {
      if (!accepts(event)) {
        return
      }

      event.preventDefault()
      // The row drag advertises effectAllowed='copy' (for composer drops);
      // anything else here would cancel the drop.
      event.dataTransfer.dropEffect = 'copy'
    },
    [accepts]
  )

  // Unaccepted drags never increment, but their leave events still arrive —
  // guard the decrement so they can't drive depth negative and wedge the
  // highlight on a later accepted drag.
  const onDragLeave = useCallback(() => {
    if (depth.current > 0 && --depth.current <= 0) {
      reset()
    }
  }, [reset])

  const onDrop = useCallback(
    (event: ReactDragEvent) => {
      if (!accepts(event)) {
        return
      }

      event.preventDefault()
      reset()

      const session = readSessionDrag(event.dataTransfer)

      if (session) {
        onDropSession(session, event)
      }
    },
    [accepts, onDropSession, reset]
  )

  return {
    active,
    dropHandlers: { onDragEnter, onDragLeave, onDragOver, onDrop }
  }
}

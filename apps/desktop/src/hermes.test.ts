import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { bulkArchiveSessions, getSessionMessages, listAllProfileSessions, listSessions } from './hermes'

describe('bulkArchiveSessions', () => {
  const originalHermesDesktop = window.hermesDesktop

  afterEach(() => {
    vi.restoreAllMocks()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: originalHermesDesktop,
      writable: true
    })
  })

  it('posts deduped preserve ids to the manual bulk archive endpoint', async () => {
    const api = vi.fn().mockResolvedValue({ ok: true, archived: 12 })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api },
      writable: true
    })

    await expect(bulkArchiveSessions(['pin', '', 'current', 'pin'])).resolves.toEqual({ ok: true, archived: 12 })

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/bulk-archive',
      method: 'POST',
      body: { preserve_ids: ['pin', 'current'] }
    })
  })
})

const emptySessionsResponse = {
  limit: 0,
  offset: 0,
  sessions: [],
  total: 0
}

describe('Hermes REST session helpers', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn().mockResolvedValue(emptySessionsResponse)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('uses a longer timeout for the single-profile session list', async () => {
    await listSessions(50, 1)

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/sessions?limit=50&offset=0&min_messages=1&archived=exclude&order=recent',
        timeoutMs: 60_000
      })
    )
  })

  it('uses a longer timeout for the all-profile session list', async () => {
    await listAllProfileSessions(50, 1)

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/profiles/sessions?limit=50&offset=0&min_messages=1&archived=exclude&order=recent&profile=all',
        timeoutMs: 60_000
      })
    )
  })

  it('tags cross-profile message reads for Electron routing and backend lookup', async () => {
    api.mockResolvedValue({ messages: [], session_id: 'session-1' })

    await getSessionMessages('session-1', 'xiaoxuxu')

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/session-1/messages?profile=xiaoxuxu',
      profile: 'xiaoxuxu'
    })
  })
})

import { describe, expect, it, vi } from 'vitest'

import type { ComposerAttachment } from '@/store/composer'

import { firstMissingAttachmentPath, localAttachmentPaths } from './queued-attachment-preflight'

const attachment = (overrides: Partial<ComposerAttachment> & { id: string }): ComposerAttachment => ({
  kind: 'image',
  label: overrides.id,
  ...overrides
})

describe('localAttachmentPaths', () => {
  it('collects only the unstaged file/image chips that reference local disk', () => {
    const paths = localAttachmentPaths([
      attachment({ id: 'img', path: '/tmp/Screen Shot.png' }),
      attachment({ id: 'doc', kind: 'file', path: '/tmp/notes.md' }),
      // Already staged into a session — its `path` is a gateway-side path now.
      attachment({ id: 'staged', attachedSessionId: 'rt-1', path: '/gateway/side.png' }),
      // Pathless refs have nothing on this disk to check.
      attachment({ id: 'term', kind: 'terminal', refText: '@terminal:1' }),
      attachment({ id: 'url', kind: 'url', refText: 'https://example.com' }),
      attachment({ id: 'folder', kind: 'folder', path: '/tmp/dir' })
    ])

    expect(paths).toEqual(['/tmp/Screen Shot.png', '/tmp/notes.md'])
  })

  it('survives holes a session switch can leave in the array', () => {
    const withHoles = [
      undefined,
      attachment({ id: 'img', path: '/tmp/a.png' }),
      null
    ] as unknown as ComposerAttachment[]

    expect(localAttachmentPaths(withHoles)).toEqual(['/tmp/a.png'])
  })
})

describe('firstMissingAttachmentPath', () => {
  it('returns the full path of the first attachment that is gone, spaces intact', async () => {
    const missing = '/Users/me/Library/Application Support/Hermes/images/clip_20260614.png'
    const probe = vi.fn(async (path: string) => path !== missing)

    await expect(
      firstMissingAttachmentPath(
        [attachment({ id: 'a', path: '/tmp/here.png' }), attachment({ id: 'b', path: missing })],
        probe
      )
    ).resolves.toBe(missing)
  })

  it('returns null when every referenced attachment is still on disk', async () => {
    await expect(
      firstMissingAttachmentPath([attachment({ id: 'a', path: '/tmp/here.png' })], async () => true)
    ).resolves.toBeNull()
  })

  it('stops probing once it finds a miss', async () => {
    const probe = vi.fn(async () => false)

    await firstMissingAttachmentPath(
      [attachment({ id: 'a', path: '/a.png' }), attachment({ id: 'b', path: '/b.png' })],
      probe
    )

    expect(probe).toHaveBeenCalledTimes(1)
  })

  it('treats a broken or absent probe as "present" — a preflight may never dead-letter a good entry', async () => {
    const attachments = [attachment({ id: 'a', path: '/tmp/here.png' })]

    await expect(firstMissingAttachmentPath(attachments, undefined)).resolves.toBeNull()
    await expect(
      firstMissingAttachmentPath(attachments, async () => {
        throw new Error('bridge exploded')
      })
    ).resolves.toBeNull()
  })

  it('never probes an entry with no local attachments', async () => {
    const probe = vi.fn(async () => false)

    await expect(firstMissingAttachmentPath([], probe)).resolves.toBeNull()
    expect(probe).not.toHaveBeenCalled()
  })
})

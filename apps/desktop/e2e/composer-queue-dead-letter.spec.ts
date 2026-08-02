/**
 * E2E: a poison queued prompt must not replay on every launch.
 *
 * Reproduces the reported bug against a real app boot with an isolated
 * HERMES_HOME: a prompt queued months ago in another session, referencing a
 * screenshot that no longer exists at a path containing spaces, used to replay
 * four duplicate user bubbles plus a stack of "image not found" errors into
 * whichever chat happened to be open — on every single restart, forever.
 *
 * Unit tests cover the store transitions and the drain logic. What only a real
 * boot can prove is the part the user actually reported: that the entry
 * survives in localStorage across a relaunch and stays inert.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, test } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

const QUEUE_STORAGE_KEY = 'hermes.desktop.composerQueue.v1'

/** The operator's fossil, to the letter: a June turn whose screenshot lived
 *  under a path with spaces and is long gone. */
const FOSSIL_TEXT = "there's no title anymore. also what about the season posters?"

const FOSSIL_ATTACHMENT_PATH = '/Users/obaradei/Library/Application Support/Hermes/images/clip_20260614_120000_1.png'

const JUNE = Date.parse('2026-06-14T12:00:00Z')

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture!, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

/** Count the user turns currently rendered in the transcript. */
async function userBubbleCount(page: MockBackendFixture['page']): Promise<number> {
  return page.locator('[data-role="user"], [data-message-role="user"]').count()
}

/** Any visible "image not found" error row. */
function imageErrorRows(page: MockBackendFixture['page']) {
  return page.getByText(/image not found/i)
}

test.describe('queued poison prompt', () => {
  test('a months-old queued turn with a missing attachment stays inert across a relaunch', async () => {
    const page = fixture!.page

    // Plant the fossil under a stored session id the app has never seen — the
    // shape of the bug: the origin conversation is not hydrated, so the drain
    // has no runtime binding for it and used to fall through to the foreground.
    await page.evaluate(
      ([key, text, attachmentPath, queuedAt]) => {
        window.localStorage.setItem(
          key as string,
          JSON.stringify({
            'stored-session-june-fossil': [
              {
                id: 'queued-fossil-1',
                text,
                queuedAt,
                attachments: [
                  {
                    id: 'att-fossil-1',
                    kind: 'image',
                    label: 'clip_20260614_120000_1.png',
                    path: attachmentPath
                  }
                ]
              }
            ]
          })
        )
      },
      [QUEUE_STORAGE_KEY, FOSSIL_TEXT, FOSSIL_ATTACHMENT_PATH, JUNE] as const
    )

    // Relaunch the renderer — this is the "every restart" in the bug report.
    await page.reload()
    await waitForAppReady(fixture!, 120_000)

    const bubblesAfterBoot = await userBubbleCount(page)

    // Give the background drain every chance to misbehave: its retry tick is
    // 750ms and the old code burned four attempts.
    await page.waitForTimeout(6_000)

    // 1. No transcript spam: not one bubble, let alone four.
    expect(await userBubbleCount(page)).toBe(bubblesAfterBoot)
    expect(await imageErrorRows(page).count()).toBe(0)
    await expect(page.getByText(FOSSIL_TEXT, { exact: false })).toHaveCount(0)

    // 2. The entry is dead-lettered IN STORAGE, not merely skipped in memory.
    const persisted = await page.evaluate(key => {
      const raw = window.localStorage.getItem(key as string)

      return raw ? (JSON.parse(raw) as Record<string, { state?: string; stuckReason?: string }[]>) : null
    }, QUEUE_STORAGE_KEY)

    const fossil = persisted?.['stored-session-june-fossil']?.[0]
    expect(fossil?.state).toBe('stuck')
    // 30-days-old wins at load time, before a single send is attempted.
    expect(fossil?.stuckReason).toBe('expired')

    // 3. It is still THERE — migration never deletes the user's words.
    expect(persisted?.['stored-session-june-fossil']).toHaveLength(1)

    // 4. And it stays inert across another relaunch.
    await page.reload()
    await waitForAppReady(fixture!, 120_000)
    await page.waitForTimeout(3_000)

    expect(await imageErrorRows(page).count()).toBe(0)
    await expect(page.getByText(FOSSIL_TEXT, { exact: false })).toHaveCount(0)
  })
})

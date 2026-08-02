/**
 * The statusbar context meter has to track a session started by
 * `hermes desktop spawn`, not just one the user typed.
 *
 * Usage used to reach the renderer only at turn boundaries — `session.info`
 * just before `message.start`, `message.complete` at the end. A chat someone
 * types turns over often enough that the gauge looks live; a spawned
 * `--delegated` run is ONE long agentic turn, so nothing arrived between its
 * two ends and the meter sat frozen for the whole run.
 *
 * The gateway now pushes `token.usage` once per usage-bearing API response, as
 * the provider's real prompt count lands, plus a cheap sample as each tool
 * completes. That is why the mock answers `stream_options: {include_usage}`
 * with a real, GROWING prompt count: a mock that reports nothing (or a
 * constant) exercises none of the recorder path, and cannot tell a working
 * meter from a frozen one.
 *
 * The mock backend is also told to take its time over each tool round trip so a
 * single turn lasts long enough to sample; without that the turn finishes
 * inside one poll and the test could pass without the fix.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import fs from 'node:fs'
import path from 'node:path'

import { expect, test } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

/**
 * Long enough that one turn spans several polls below, and longer than the
 * gateway's minimum interval between usage frames — a real agentic round trip
 * takes seconds to minutes, so a round that outlives that floor is the shape
 * being reproduced. Below the floor the loop would still be reported, just at
 * the floor's rate rather than once per round.
 */
const TOOL_ROUND_TRIP_MS = 1_400

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  process.env.MOCK_TOOL_CALL_DELAY_MS = String(TOOL_ROUND_TRIP_MS)
  fixture = await setupMockBackend()
  await waitForAppReady(fixture!, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
  delete process.env.MOCK_TOOL_CALL_DELAY_MS
})

/**
 * The context readout, e.g. "20k/256k [█░░░░░░░░░] 8%" — '' when absent.
 *
 * Read off the item's own button, not the statusbar's concatenated text: the
 * items render flush against each other, so the neighbouring subagent counter
 * turns "16.3k/256k" into an unparseable "0316.3k/256k".
 */
async function meterText(): Promise<string> {
  return (
    (await fixture!.page.evaluate(() => {
      const bar = document.querySelector('[data-slot="statusbar"]')

      for (const button of Array.from(bar?.querySelectorAll('button') ?? [])) {
        const spans = Array.from(button.querySelectorAll(':scope > span')).map(span => span.textContent ?? '')

        if (spans.some(span => /^[\d.]+[kmb]?\/[\d.]+[kmb]?$/i.test(span.trim()))) {
          return spans.join(' ').replace(/\s+/g, ' ').trim()
        }
      }

      return ''
    })) ?? ''
  )
}

/** Poll the meter, returning each DISTINCT value it took, in order. */
async function meterSeries(label: string, seconds: number): Promise<string[]> {
  const series: string[] = []

  for (let i = 0; i < seconds * 5; i += 1) {
    const value = await meterText()

    if (series[series.length - 1] !== value) {
      series.push(value)
      console.log(`  [${label}] t=${(i / 5).toFixed(1)}s  meter="${value}"`)
    }

    await fixture!.page.waitForTimeout(200)
  }

  return series
}

/**
 * Read the spawn control channel the app published under HERMES_HOME.
 *
 * Polled: the app writes this once its control server is listening, which can
 * land after the window is otherwise ready.
 */
async function readControlFile(): Promise<{ port: number; token: string }> {
  const controlPath = path.join(fixture!.sandbox.hermesHome, 'desktop', 'control.json')

  for (let i = 0; i < 60; i += 1) {
    try {
      const parsed = JSON.parse(fs.readFileSync(controlPath, 'utf8'))

      if (parsed?.port && parsed?.token) {
        return { port: Number(parsed.port), token: String(parsed.token) }
      }
    } catch {
      // not written yet, or written half — retry
    }

    await fixture!.page.waitForTimeout(500)
  }

  throw new Error(`no usable control.json at ${controlPath} after 30s`)
}

async function postSpawn(prompt: string, extra: Record<string, unknown> = {}): Promise<number> {
  const { port, token } = await readControlFile()

  const res = await fetch(`http://127.0.0.1:${port}/spawn`, {
    body: JSON.stringify({ prompt, ...extra }),
    headers: { 'Content-Type': 'application/json', 'X-Hermes-Desktop-Token': token },
    method: 'POST'
  })

  return res.status
}

/** Values the gauge actually showed — drops the empty reading a new chat starts at. */
function gaugeReadings(series: string[]): string[] {
  return series.filter(Boolean)
}

/** "16.3k/256k [█░░░] 6%" -> 16300. Only the occupancy side is compared. */
function occupancyOf(reading: string): number {
  const [, digits, suffix = ''] = /^([\d.]+)([kmb]?)\//i.exec(reading) ?? []

  if (!digits) {
    throw new Error(`unparseable gauge reading: "${reading}"`)
  }

  return Number(digits) * { '': 1, b: 1e9, k: 1e3, m: 1e6 }[suffix.toLowerCase()]!
}

/**
 * A turn's context only grows: each round appends the last tool's result to the
 * prompt. A reading that goes BACKWARDS means something other than this
 * session's own main-agent calls reached the gauge — an auxiliary call such as
 * the goal judge, or a delegate subagent reporting its own much smaller window.
 * Compaction is the one legitimate drop and cannot happen at these sizes.
 */
function expectNeverDips(readings: string[]): void {
  const occupancies = readings.map(occupancyOf)

  for (let i = 1; i < occupancies.length; i += 1) {
    expect(occupancies[i], `dipped: ${readings.join(' -> ')}`).toBeGreaterThanOrEqual(occupancies[i - 1])
  }
}

test.describe('statusbar context meter', () => {
  test('tracks a spawned session while its single turn is still running', async () => {
    expect(await postSpawn('Use your tools and then report back.', { delegated: true })).toBe(202)

    const series = await meterSeries('spawn', 20)
    const readings = gaugeReadings(series)

    console.log('SPAWN readings:', readings)

    // Before the mid-turn `token.usage` frames existed this was at most one
    // reading, published by message.complete once the whole turn had finished.
    expect(readings.length).toBeGreaterThanOrEqual(2)
    // A real gauge, not a bare token count: "16.3k/256k [█░░░░░░░░░] 6%".
    expect(readings[0]).toMatch(/^[\d.]+[kmb]?\/[\d.]+[kmb]?\s+\[[█░]+\]\s+\d+%$/i)
    expectNeverDips(readings)
  })

  test('still tracks a session the user typed', async () => {
    const page = fixture!.page
    const composer = page.locator('[contenteditable="true"]').first()

    await composer.waitFor({ state: 'visible', timeout: 15_000 })
    await composer.click()
    await page.keyboard.insertText('Use your tools and then report back.')
    await page.keyboard.press('Enter')

    const readings = gaugeReadings(await meterSeries('typed', 20))

    console.log('TYPED readings:', readings)

    expect(readings.length).toBeGreaterThanOrEqual(2)
    expectNeverDips(readings)
  })
})

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { QueuedPromptEntry } from '@/store/composer-queue'

import { QueuePanel } from './queue-panel'

// A dead-lettered queued turn has stopped retrying on its own. The panel is the
// only place the user can see WHY and do something about it, so it must show
// the reason with its untruncated detail and offer both ways out — Retry and
// Delete — instead of the Send action that no longer applies.

function entry(overrides: Partial<QueuedPromptEntry> = {}): QueuedPromptEntry {
  return {
    id: 'q1',
    text: 'there is no title anymore. also what about the season posters?',
    attachments: [],
    queuedAt: Date.now(),
    attempts: 0,
    state: 'pending',
    ...overrides
  }
}

const MISSING_PATH = '/Users/obaradei/Library/Application Support/Hermes/images/clip_20260614.png'

function renderPanel(entries: QueuedPromptEntry[], handlers: Partial<Parameters<typeof QueuePanel>[0]> = {}) {
  const props = {
    busy: false,
    editingId: null,
    entries,
    onDelete: vi.fn(),
    onEdit: vi.fn(),
    onResume: vi.fn(),
    onRetry: vi.fn(),
    onSendNow: vi.fn(),
    parked: false,
    ...handlers
  }

  render(<QueuePanel {...props} />)

  return props
}

/** The section collapses itself for a healthy queue; open it to read the rows. */
function expandSection() {
  fireEvent.click(screen.getByRole('button', { name: /Queued/ }))
}

describe('QueuePanel dead-letter row', () => {
  afterEach(cleanup)

  it('offers Send (not Retry) for a pending entry', () => {
    renderPanel([entry()])
    expandSection()

    expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  })

  it('opens itself when an entry is dead-lettered, and stays shut when all are healthy', () => {
    const { unmount } = render(
      <QueuePanel
        busy={false}
        editingId={null}
        entries={[entry()]}
        onDelete={vi.fn()}
        onEdit={vi.fn()}
        onResume={vi.fn()}
        onRetry={vi.fn()}
        onSendNow={vi.fn()}
        parked={false}
      />
    )

    // Healthy: tucked away, nothing demanding attention.
    expect(screen.queryByRole('button', { name: 'Send' })).toBeNull()
    unmount()

    // Stuck: a failure the user must act on cannot hide behind a disclosure.
    renderPanel([entry({ state: 'stuck', attempts: 4, stuckReason: 'drain-failed' })])
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  })

  it('swaps Send for Retry once the entry is dead-lettered', () => {
    const props = renderPanel([
      entry({ state: 'stuck', attempts: 4, stuckReason: 'attachment-missing', lastError: MISSING_PATH })
    ])

    expect(screen.queryByRole('button', { name: 'Send' })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(props.onRetry).toHaveBeenCalledWith('q1')
  })

  it('shows the reason with the FULL failure detail, never truncated', () => {
    renderPanel([entry({ state: 'stuck', attempts: 4, stuckReason: 'attachment-missing', lastError: MISSING_PATH })])

    // A path cut at the first space ("…/Library/Application") is what made this
    // class of failure impossible to diagnose. The whole path must be present.
    const row = screen.getByText(new RegExp(MISSING_PATH.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')))
    expect(row.textContent).toContain('Application Support')
    expect(row.textContent).toContain('Attachment file no longer exists')
  })

  it('names the reason even when there is no detail to show', () => {
    renderPanel([entry({ state: 'stuck', attempts: 4, stuckReason: 'expired' })])

    expect(screen.getByText(/Queued too long ago/)).toBeTruthy()
  })

  it('always offers Delete so a poison entry can be removed', () => {
    const props = renderPanel([entry({ state: 'stuck', attempts: 4, stuckReason: 'drain-failed' })])

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
    expect(props.onDelete).toHaveBeenCalledWith('q1')
  })

  it('keeps a healthy entry beside a dead-lettered one, each with its own actions', () => {
    renderPanel([
      entry({ id: 'poison', state: 'stuck', attempts: 4, stuckReason: 'attachment-missing', lastError: MISSING_PATH }),
      entry({ id: 'good', text: 'still fine' })
    ])

    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy()
    expect(screen.getAllByRole('button', { name: 'Delete' })).toHaveLength(2)
  })
})

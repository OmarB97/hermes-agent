import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { StatusRule } from '../components/appChrome.js'
import { DEFAULT_THEME } from '../theme.js'

type ReactNodeLike = React.ReactNode

const textContent = (node: ReactNodeLike): string => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return ''
  }

  if (typeof node === 'string' || typeof node === 'number') {
    return String(node)
  }

  if (Array.isArray(node)) {
    return node.map(textContent).join('')
  }

  if (React.isValidElement(node)) {
    return textContent(node.props.children)
  }

  return ''
}

const findClickableWithText = (node: ReactNodeLike, needle: string): React.ReactElement | null => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return null
  }

  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findClickableWithText(child, needle)

      if (found) {
        return found
      }
    }

    return null
  }

  if (!React.isValidElement(node)) {
    return null
  }

  if (typeof node.props.onClick === 'function' && textContent(node).includes(needle)) {
    return node
  }

  return findClickableWithText(node.props.children, needle)
}

const findTextWithColor = (node: ReactNodeLike, needle: string, color: string): React.ReactElement | null => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return null
  }

  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findTextWithColor(child, needle, color)

      if (found) {
        return found
      }
    }

    return null
  }

  if (!React.isValidElement(node)) {
    return null
  }

  if (node.props.color === color && textContent(node).includes(needle)) {
    return node
  }

  return findTextWithColor(node.props.children, needle, color)
}

const hasComponentNamed = (node: ReactNodeLike, name: string): boolean => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return false
  }

  if (Array.isArray(node)) {
    return node.some(child => hasComponentNamed(child, name))
  }

  if (!React.isValidElement(node)) {
    return false
  }

  const type = node.type

  if (typeof type === 'function' && type.name === name) {
    return true
  }

  return hasComponentNamed(node.props.children, name)
}

describe('StatusRule session count click target', () => {
  it('makes the live session count itself clickable', () => {
    const openSwitcher = vi.fn()

    const element = StatusRule({
      bgCount: 0,
      busy: false,
      cols: 100,
      cwdLabel: '~/repo',
      liveSessionCount: 1,
      model: 'kimi-k2.6',
      onSessionCountClick: openSwitcher,
      sessionStartedAt: null,
      showCost: false,
      status: 'ready',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: { total: 0 },
      voiceLabel: ''
    })

    const clickableSessionCount = findClickableWithText(element, '1 session')

    expect(clickableSessionCount).not.toBeNull()
    clickableSessionCount!.props.onClick({ stopImmediatePropagation: vi.fn() })
    expect(openSwitcher).toHaveBeenCalledOnce()
  })

  it('keeps status + model and drops the low-value tail on a narrow terminal', () => {
    const element = StatusRule({
      bgCount: 0,
      busy: false,
      cols: 44,
      cwdLabel: '~/src/hermes-agent/apps/desktop (bb/tui-statusbar-responsive)',
      liveSessionCount: 3,
      model: 'opus-4.8',
      onSessionCountClick: vi.fn(),
      sessionStartedAt: Date.now() - 60_000,
      showCost: true,
      status: 'ready',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: { context_max: 200_000, context_percent: 25, context_used: 50_000, cost_usd: 0.5, total: 50_000 },
      voiceLabel: 'voice off'
    })

    const rendered = textContent(element)

    // Must-keep essentials survive intact …
    expect(rendered).toContain('ready')
    expect(rendered).toContain('opus 4.8')
    // … while the low-value tail (session count, cost) is dropped, not truncated.
    expect(rendered).not.toContain('3 sessions')
    expect(rendered).not.toContain('$0.5000')
  })

  it('uses stable narrow rows with personality indicator, model, and colored context capacity', () => {
    const element = StatusRule({
      bgCount: 2,
      busy: true,
      cols: 44,
      cwdLabel: '~/src/hermes-agent/apps/desktop (bb/tui-statusbar-responsive)',
      indicatorStyle: 'emoji',
      liveSessionCount: 3,
      model: 'qwen3.6-27b',
      modelReasoningEffort: 'none',
      onSessionCountClick: vi.fn(),
      sessionStartedAt: Date.now() - 60_000,
      showCost: true,
      status: 'working',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: Date.now() - 14_000,
      usage: {
        context_max: 131_072,
        context_percent: 13.5,
        context_used: 17_700,
        cost_usd: 0.5,
        total: 17_700
      },
      voiceLabel: 'voice off'
    })

    const rendered = textContent(element)

    expect(hasComponentNamed(element, 'FaceTicker')).toBe(true)
    expect(rendered).toContain('qwen3.6 27b none')
    expect(rendered).toContain('ctx 17.7k/131.1k 14%')
    expect(findTextWithColor(element, '17.7k/131.1k', DEFAULT_THEME.color.statusGood)).not.toBeNull()
    expect(rendered).not.toContain('3 sessions')
    expect(rendered).not.toContain('$0.5000')
    expect(rendered).not.toContain('~/src/hermes-agent')
  })
})

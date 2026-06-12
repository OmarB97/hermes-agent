import { beforeEach, describe, expect, it } from 'vitest'
import { $bootstrap, $progress, applyBootstrapEvent } from './store'

const stages = [
  {
    name: 'update',
    title: 'Updating Hermes',
    category: 'update',
    needs_user_input: false
  },
  {
    name: 'rebuild',
    title: 'Rebuilding the desktop app',
    category: 'update',
    needs_user_input: false
  },
  {
    name: 'install',
    title: 'Installing the updated app',
    category: 'update',
    needs_user_input: false
  }
]

describe('bootstrap progress', () => {
  beforeEach(() => {
    applyBootstrapEvent({
      type: 'manifest',
      stages,
      protocolVersion: null
    })
  })

  it('reports the active stage as the current step', () => {
    applyBootstrapEvent({
      type: 'stage',
      name: 'update',
      state: 'running'
    })

    expect($progress.get()).toEqual({
      done: 0,
      current: 1,
      total: 3,
      fraction: 1 / 3
    })

    applyBootstrapEvent({
      type: 'stage',
      name: 'update',
      state: 'succeeded'
    })
    applyBootstrapEvent({
      type: 'stage',
      name: 'rebuild',
      state: 'running'
    })

    expect($bootstrap.get().currentStage).toBe('rebuild')
    expect($progress.get()).toMatchObject({
      done: 1,
      current: 2,
      total: 3
    })
  })

  it('clears stale active stages when a stage finishes', () => {
    applyBootstrapEvent({
      type: 'stage',
      name: 'update',
      state: 'running'
    })
    applyBootstrapEvent({
      type: 'stage',
      name: 'update',
      state: 'succeeded'
    })

    expect($bootstrap.get().currentStage).toBeNull()
    expect($progress.get()).toMatchObject({
      done: 1,
      current: 1,
      total: 3
    })
  })
})

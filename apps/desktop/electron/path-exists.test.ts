import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { pathExistsForIpc } from './path-exists'

function mkTmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-path-exists-'))
}

test('reports a real file as present, including through a path containing spaces', async () => {
  const root = mkTmpDir()

  try {
    // The shape that started this: a screenshot parked under a directory whose
    // name has a space, the way macOS lays out Application Support.
    const dir = path.join(root, 'Application Support', 'Hermes Desktop')
    fs.mkdirSync(dir, { recursive: true })
    const file = path.join(dir, 'season poster.png')
    fs.writeFileSync(file, Buffer.from('\x89PNG\r\n\x1a\n'))

    assert.equal(await pathExistsForIpc(file), true)
  } finally {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

test('reports a deleted file as absent so a queued turn can be dead-lettered', async () => {
  const root = mkTmpDir()

  try {
    const missing = path.join(root, 'Application Support', 'Hermes Desktop', 'gone forever.png')

    assert.equal(await pathExistsForIpc(missing), false)
  } finally {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

test('a directory is not a usable attachment', async () => {
  const root = mkTmpDir()

  try {
    assert.equal(await pathExistsForIpc(root), false)
  } finally {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

test('blank and non-string input answer false instead of throwing', async () => {
  assert.equal(await pathExistsForIpc(''), false)
  assert.equal(await pathExistsForIpc('   '), false)
  assert.equal(await pathExistsForIpc(null), false)
  assert.equal(await pathExistsForIpc(undefined), false)
  assert.equal(await pathExistsForIpc(42), false)
})

test('any resolver failure answers false rather than propagating', async () => {
  const boom = () => {
    throw new Error('unsafe path syntax')
  }

  assert.equal(await pathExistsForIpc('/whatever', { resolveReadableFile: boom as never }), false)
})

test('a present-but-sensitive file still reports present (we never read a byte)', async () => {
  const root = mkTmpDir()

  try {
    // Sensitivity blocking guards file CONTENT reads. Reporting such a file as
    // "missing" would dead-letter a queued turn with a wrong reason, so the
    // probe must answer the existence question honestly.
    const dir = path.join(root, '.ssh')
    fs.mkdirSync(dir, { recursive: true })
    const file = path.join(dir, 'id_rsa')
    fs.writeFileSync(file, 'not really a key')

    assert.equal(await pathExistsForIpc(file), true)
  } finally {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

import type { TestProjectConfiguration } from 'vitest/config'
import { defineConfig } from 'vitest/config'

const reactUi: TestProjectConfiguration = {
  extends: './vite.config.ts',
  test: {
    name: 'ui',
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    globals: true,
    // The first test in each file pays jsdom env init + full module transform,
    // which can exceed vitest's 5000ms default under CI/load. 15s gives the
    // cold start headroom without masking genuinely hung tests.
    testTimeout: 15_000
  }
}

const electronNative: TestProjectConfiguration = {
  test: {
    name: 'electron',
    environment: 'node',
    // Hermetic git env — see the file for why the host's ~/.gitconfig is not
    // something these suites can be allowed to inherit.
    setupFiles: ['./vitest.setup.electron.ts'],
    include: ['electron/**/*.test.ts', 'scripts/**.test.{ts,mjs}'],
    // The git suites shell out to the real binary — a dozen or so process
    // spawns each. That is ~250ms on an idle machine, but process creation is
    // the first thing to degrade under load, and measured here it reached 3.8s
    // at load average 190 while the whole suite ran. vitest's 5000ms default
    // left no room for that and these tests timed out intermittently. 15s
    // matches the ui project and still catches a genuine hang, which is
    // unbounded rather than merely slow.
    testTimeout: 15_000
  }
}

export default defineConfig({
  test: {
    projects: [reactUi, electronNative]
  }
})

// Give every `git` the electron suites spawn a hermetic environment.
//
// These tests shell out to the real git binary on purpose — worktree layout,
// upstream tracking and porcelain status are things only git can tell us
// truthfully. What they must NOT do is inherit whoever's ~/.gitconfig happens
// to be on the machine, because that is both a correctness and a speed problem.
//
// The one that actually bit us is `core.fsmonitor=true`. With it set, git
// starts a per-repo `fsmonitor--daemon` for each of the ~16 throwaway repos
// these files create. The daemon is detached and multi-threaded, and the test's
// `rmSync` deletes the repo out from under it — so every run leaks a handful of
// live processes that are still there for the next one. They accumulate (a
// developer machine here had 99), and they compete for the cores the next run
// needs. That is the whole story behind these suites timing out at vitest's 5s
// default while passing comfortably when run alone: the flake was self-inflicted
// and cumulative, not random.
//
// The rest are hangs waiting to happen rather than slowdowns. `commit.gpgsign`
// blocks on a passphrase nobody will type; `core.hooksPath` runs somebody's
// hooks against our fixtures; a credential helper can sit waiting on a prompt.
// A test that hangs looks exactly like a test that is merely slow, which is how
// this kind of thing hides.
//
// `/dev/null` is git's documented way to say "there is no config file here".
process.env.GIT_CONFIG_GLOBAL = '/dev/null'
process.env.GIT_CONFIG_SYSTEM = '/dev/null'

// Belt and braces: if git still finds a reason to prompt, fail instead of
// blocking. A prompt in a headless run is never going to be answered.
process.env.GIT_TERMINAL_PROMPT = '0'

// With no config files left there is no identity to commit with and no
// `init.defaultBranch`, so supply both through git's env-config channel. This
// also spares each fixture the two `git config` subprocesses it would otherwise
// spawn, and stops `git init` printing its default-branch hint into the test
// output. Command-line `-c` still overrides these, so a test that wants a
// different identity can say so inline.
const ENV_CONFIG: ReadonlyArray<readonly [string, string]> = [
  ['init.defaultBranch', 'main'],
  ['user.email', 'hermes-test@example.com'],
  ['user.name', 'Hermes Test'],
  // Redundant while the two files above are /dev/null, but this is the setting
  // that caused the damage — say it out loud so it survives a future edit.
  ['core.fsmonitor', 'false']
]

process.env.GIT_CONFIG_COUNT = String(ENV_CONFIG.length)

ENV_CONFIG.forEach(([key, value], index) => {
  process.env[`GIT_CONFIG_KEY_${index}`] = key
  process.env[`GIT_CONFIG_VALUE_${index}`] = value
})

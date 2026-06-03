const { spawn } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: 'inherit' })
    child.on('error', reject)
    child.on('close', code => (code === 0 ? resolve() : reject(new Error(`${command} ${args.join(' ')} exited ${code}`))))
  })
}

function resolveArtifactFromContext(context) {
  const appPath = context?.appOutDir && context?.packager?.appInfo?.productFilename
    ? path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`)
    : null
  if (appPath && fs.existsSync(appPath)) return appPath

  const onlyPaths = process.argv.filter((arg, idx) => idx > 0 && !arg.startsWith('-') && !arg.includes('='))
  if (onlyPaths.length > 0) return onlyPaths[onlyPaths.length - 1]

  const explicit = process.argv.find(arg => arg.startsWith('--artifact='))
  if (explicit) return explicit.split('=').slice(1).join('=')

  return process.env.NOTARIZE_ARTIFACT || process.env.ARTIFACT_PATH || ''
}

async function maybeNotarize(artifact) {
  if (!artifact || !fs.existsSync(artifact)) {
    console.log('[notarize] no artifact to notarize; skipping')
    return
  }

  const profile = String(process.env.APPLE_NOTARY_PROFILE || '').trim()
  const keyId = String(process.env.APPLE_API_KEY_ID || '').trim()
  const issuer = String(process.env.APPLE_API_ISSUER || '').trim()
  const rawApiKey = process.env.APPLE_API_KEY

  if (!profile && !(rawApiKey && keyId && issuer)) {
    console.log('[notarize] no Apple notarization credentials configured; skipping')
    return
  }

  const tool = path.join(__dirname, 'notarize-artifact.cjs')
  await run('node', [tool, artifact])
}

exports.default = async function notarize(context) {
  const artifact = resolveArtifactFromContext(context)
  await maybeNotarize(artifact)
}

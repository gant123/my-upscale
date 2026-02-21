import { BrowserWindow, app, dialog, ipcMain, shell } from 'electron'
import { electronApp, is, optimizer } from '@electron-toolkit/utils'

import fs from 'fs'
import icon from '../../resources/icon.png?asset'
import { join } from 'path'
import path from 'path'
import { spawn } from 'child_process'

// ─── Constants ───
const ALLOWED_IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'])
const ALLOWED_ENHANCE_MODES = new Set(['pro', 'color', 'smooth', 'light', 'portrait'])
const ALLOWED_XRAY_MODES = new Set([
  'none', 'structure', 'depth', 'frequency', 'thermal',
  'bones', 'reveal', 'bright', 'occlusion'
])
const MAX_IMAGE_SIZE_BYTES = 500 * 1024 * 1024
const ENGINE_TIMEOUT_MS = 120_000 // 2 min for AI operations
const ENGINE_TIMEOUT_QUICK_MS = 30_000

// ─── Helpers ───
function getMime(p) {
  const ext = p.split('.').pop().toLowerCase()
  return { jpg: 'jpeg', jpeg: 'jpeg', png: 'png', webp: 'webp', bmp: 'bmp', tiff: 'tiff' }[ext] || 'png'
}

function toBase64(p) {
  return `data:image/${getMime(p)};base64,${fs.readFileSync(p).toString('base64')}`
}

function isValidImagePath(filePath) {
  if (typeof filePath !== 'string' || !filePath.trim()) return false
  if (!fs.existsSync(filePath)) return false
  const ext = path.extname(filePath).toLowerCase()
  if (!ALLOWED_IMAGE_EXTENSIONS.has(ext)) return false
  const { size } = fs.statSync(filePath)
  return size > 0 && size <= MAX_IMAGE_SIZE_BYTES
}

function sanitizeAdjustParams(params = {}) {
  const bn = (value, min, max, fallback = 0) => {
    const n = Number(value)
    if (!Number.isFinite(n)) return fallback
    return Math.max(min, Math.min(max, n))
  }
  return {
    exposure: bn(params.exposure, -3, 3),
    contrast: bn(params.contrast, -100, 100),
    highlights: bn(params.highlights, -100, 100),
    shadows: bn(params.shadows, -100, 100),
    whites: bn(params.whites, -100, 100),
    blacks: bn(params.blacks, -100, 100),
    temperature: bn(params.temperature, -100, 100),
    tint: bn(params.tint, -100, 100),
    vibrance: bn(params.vibrance, -100, 100),
    saturation: bn(params.saturation, -100, 100),
    clarity: bn(params.clarity, -100, 100),
    dehaze: bn(params.dehaze, 0, 100),
    sharpness: bn(params.sharpness, 0, 100),
    grain: bn(params.grain, 0, 100),
    vignette: bn(params.vignette, -100, 100),
    xray: ALLOWED_XRAY_MODES.has(params.xray) ? params.xray : 'none',
    xray_blend: bn(params.xray_blend, 0, 100, 100)
  }
}

// ─── Engine Call ───
// Two separate resolvers: one for the Python binary, one for engine.py

function venvPython() {
  const candidates = [
    join(app.getAppPath(), 'myenv', 'bin', 'python'),
    join(process.cwd(), 'myenv', 'bin', 'python'),
    join(app.getAppPath(), '.venv', 'bin', 'python'),
    join(process.cwd(), '.venv', 'bin', 'python'),
  ]
  return candidates.find((p) => {
    try { return fs.existsSync(p) } catch { return false }
  }) || null  // null = fall back to system python3/python
}

function enginePath() {
  const candidates = [
    join(process.resourcesPath || '', 'python', 'engine.py'),
    join(app.getAppPath(), 'python', 'engine.py'),
    join(app.getAppPath(), 'engine.py'),
    join(process.cwd(), 'engine.py'),
    'engine.py'
  ]
  return candidates.find((p) => fs.existsSync(p)) || 'engine.py'
}

function callEngine(command, timeoutMs = ENGINE_TIMEOUT_MS) {
  const scriptPath = enginePath()
  const venvBin = venvPython()

  // Try venv python first, then system python3, then python
  const binaries = [venvBin, 'python3', 'python'].filter(Boolean)

  const run = (binary) =>
    new Promise((resolve) => {
      // binary = python executable, scriptPath = engine.py
      const child = spawn(binary, [scriptPath])
      let stdout = ''
      let stderr = ''
      let settled = false

      const finish = (value) => {
        if (!settled) { settled = true; resolve(value) }
      }

      const timer = setTimeout(() => {
        child.kill('SIGKILL')
        finish({ error: 'Engine timed out' })
      }, timeoutMs)

      child.stdout.on('data', (d) => { stdout += d.toString() })
      child.stderr.on('data', (d) => { stderr += d.toString() })
      child.on('error', (error) => {
        clearTimeout(timer)
        finish({ spawnError: error })
      })
      child.on('close', (code) => {
        clearTimeout(timer)
        if (code !== 0 && !stdout.includes('{')) {
          return finish({ error: stderr || `Engine exited with code ${code}` })
        }
        try {
          const s = stdout.indexOf('{')
          const e = stdout.lastIndexOf('}')
          if (s === -1 || e === -1) throw new Error('No JSON')
          finish(JSON.parse(stdout.substring(s, e + 1)))
        } catch {
          console.error('Engine raw output:', stdout, 'stderr:', stderr)
          finish({ error: 'Failed to parse engine output' })
        }
      })
      child.stdin.write(JSON.stringify(command))
      child.stdin.end()
    })

  // Chain through binaries: try each until one works
  let chain = run(binaries[0])
  for (let i = 1; i < binaries.length; i++) {
    const nextBin = binaries[i]
    chain = chain.then((result) => {
      if (!result?.spawnError) return result
      if (result.spawnError.code !== 'ENOENT') {
        return { error: `Failed to start engine: ${result.spawnError.message}` }
      }
      return run(nextBin)
    })
  }
  return chain.then((result) => {
    if (result?.spawnError) {
      const mainWindow = BrowserWindow.getFocusedWindow() || BrowserWindow.getAllWindows()[0]
      if (mainWindow) {
        dialog.showErrorBox(
          'Virtual Environment Not Found',
          'The "myenv" virtual environment is missing.\n\n' +
          'Please run: python3 -m venv myenv\n' +
          'Then run: source myenv/bin/activate && pip install opencv-python numpy\n\n' +
          'The app will restart after setup.'
        )
      }
      return { error: 'Python not found. Create venv: python3 -m venv myenv && source myenv/bin/activate && pip install opencv-python numpy' }
    }
    return result
  })
}
// ─── Preload Resolution ───
function preloadPath() {
  const candidates = [
    join(__dirname, '../preload/index.js'),
    join(__dirname, '../preload/index.mjs'),
    join(__dirname, '../preload/index.cjs'),
    join(__dirname, '../preload/index')
  ]
  return candidates.find((p) => fs.existsSync(p)) || candidates[0]
}

// ─── Window ───
function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1500,
    height: 950,
    minWidth: 1080,
    minHeight: 700,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#09090b',
    titleBarStyle: 'hiddenInset',
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: preloadPath(),
      contextIsolation: true,
      sandbox: false,
      nodeIntegration: false,
      webSecurity: true
    }
  })
  mainWindow.on('ready-to-show', () => mainWindow.show())
  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// ─── App Lifecycle ───
app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.aurora.ops')
  app.on('browser-window-created', (_, w) => optimizer.watchWindowShortcuts(w))

  // ── Capabilities ──
  ipcMain.handle('get-capabilities', async () => {
    return callEngine({ command: 'capabilities' }, ENGINE_TIMEOUT_QUICK_MS)
  })

  // ── File Picker ──
  ipcMain.handle('dialog:openFile', async (event) => {
    try {
      const win = BrowserWindow.fromWebContents(event.sender) || BrowserWindow.getFocusedWindow()
      if (win && !win.isDestroyed()) win.focus()

      const { canceled, filePaths } = await dialog.showOpenDialog(
        win && !win.isDestroyed() ? win : undefined,
        {
          properties: ['openFile'],
          filters: [{ name: 'Images', extensions: ['jpg', 'png', 'jpeg', 'webp', 'bmp', 'tiff'] }]
        }
      )
      if (canceled) return null
      if (!isValidImagePath(filePaths[0])) {
        return { error: 'Unsupported, missing, or too-large image file' }
      }
      return { path: filePaths[0], preview: toBase64(filePaths[0]) }
    } catch (error) {
      return { error: `Backend Error: ${error.message}` }
    }
  })

  // ── Multi-file Picker ──
  ipcMain.handle('dialog:openFiles', async (event) => {
    try {
      const win = BrowserWindow.fromWebContents(event.sender) || BrowserWindow.getFocusedWindow()
      if (win && !win.isDestroyed()) win.focus()

      const { canceled, filePaths } = await dialog.showOpenDialog(
        win && !win.isDestroyed() ? win : undefined,
        {
          properties: ['openFile', 'multiSelections'],
          filters: [{ name: 'Images', extensions: ['jpg', 'png', 'jpeg', 'webp', 'bmp', 'tiff'] }]
        }
      )
      if (canceled) return null
      return filePaths
        .filter(isValidImagePath)
        .map((p) => ({ path: p, preview: toBase64(p) }))
    } catch (error) {
      return { error: `Backend Error: ${error.message}` }
    }
  })

  // ── Analyze ──
  ipcMain.handle('run-autopilot', async (_e, imagePath) => {
    if (!isValidImagePath(imagePath)) return { error: 'Invalid image path or file type' }
    return callEngine({ command: 'analyze', image: imagePath }, ENGINE_TIMEOUT_QUICK_MS)
  })

  // ── Adjust ──
  ipcMain.handle('adjust-image', async (_e, imagePath, params) => {
    if (!isValidImagePath(imagePath)) return { error: 'Invalid image path or file type' }
    const result = await callEngine({
      command: 'adjust', image: imagePath,
      params: sanitizeAdjustParams(params)
    })
    if (result.temp_path) result.preview = toBase64(result.temp_path)
    return result
  })

  // ── Enhance ──
  ipcMain.handle('enhance-image', async (_e, imagePath, modes) => {
    if (!isValidImagePath(imagePath)) return { error: 'Invalid image path or file type' }
    const safeModes = Array.isArray(modes)
      ? modes.filter((m) => ALLOWED_ENHANCE_MODES.has(m))
      : []
    const result = await callEngine({ command: 'enhance', image: imagePath, modes: safeModes })
    if (result.temp_path) result.preview = toBase64(result.temp_path)
    return result
  })

  // ── AI Upscale ──
  ipcMain.handle('upscale-image', async (_e, imagePath, scale = 4) => {
    if (!isValidImagePath(imagePath)) return { error: 'Invalid image path or file type' }
    const s = [2, 4].includes(Number(scale)) ? Number(scale) : 4
    const result = await callEngine({ command: 'upscale', image: imagePath, scale: s })
    if (result.temp_path) result.preview = toBase64(result.temp_path)
    return result
  })

  // ── AI Face Restore ──
  ipcMain.handle('face-restore', async (_e, imagePath, model = 'gfpgan', fidelity = 0.7) => {
    if (!isValidImagePath(imagePath)) return { error: 'Invalid image path or file type' }
    const result = await callEngine({
      command: 'face_restore', image: imagePath,
      model: ['gfpgan', 'codeformer'].includes(model) ? model : 'gfpgan',
      fidelity: Math.max(0, Math.min(1, Number(fidelity) || 0.7))
    })
    if (result.temp_path) result.preview = toBase64(result.temp_path)
    return result
  })

  // ── AI Background Remove ──
  ipcMain.handle('bg-remove', async (_e, imagePath, bgColor = null) => {
    if (!isValidImagePath(imagePath)) return { error: 'Invalid image path or file type' }
    const result = await callEngine({
      command: 'bg_remove', image: imagePath,
      bg_color: Array.isArray(bgColor) ? bgColor.slice(0, 3).map(Number) : null
    })
    if (result.temp_path) result.preview = toBase64(result.temp_path)
    return result
  })

  // ── AI Inpaint ──
  ipcMain.handle('inpaint', async (_e, imagePath, maskPath, method = 'telea') => {
    if (!isValidImagePath(imagePath)) return { error: 'Invalid image path' }
    if (typeof maskPath !== 'string' || !fs.existsSync(maskPath)) {
      return { error: 'Invalid mask path' }
    }
    const result = await callEngine({
      command: 'inpaint', image: imagePath, mask: maskPath,
      method: ['telea', 'ns', 'lama'].includes(method) ? method : 'telea'
    })
    if (result.temp_path) result.preview = toBase64(result.temp_path)
    return result
  })

  // ── AI Auto-Enhance ──
  ipcMain.handle('auto-enhance', async (_e, imagePath) => {
    if (!isValidImagePath(imagePath)) return { error: 'Invalid image path or file type' }
    const result = await callEngine({ command: 'auto_enhance', image: imagePath })
    if (result.temp_path) result.preview = toBase64(result.temp_path)
    return result
  })

  // ── Batch ──
  ipcMain.handle('batch-process', async (_e, jobs) => {
    if (!Array.isArray(jobs)) return { error: 'Jobs must be an array' }
    const result = await callEngine({ command: 'batch', jobs })
    // Attach previews to any results that have temp_path
    if (result.results) {
      for (const r of result.results) {
        if (r.temp_path && fs.existsSync(r.temp_path)) {
          r.preview = toBase64(r.temp_path)
        }
      }
    }
    return result
  })

  // ── Save As ──
  ipcMain.handle('save-image', async (_e, tempPath) => {
    if (typeof tempPath !== 'string' || !fs.existsSync(tempPath))
      return { error: 'Invalid temp file' }
    const ext = tempPath.split('.').pop().toLowerCase()
    const { canceled, filePath } = await dialog.showSaveDialog({
      defaultPath: `aurora_output.${ext}`,
      filters: [
        { name: 'JPEG', extensions: ['jpg', 'jpeg'] },
        { name: 'PNG', extensions: ['png'] },
        { name: 'WebP', extensions: ['webp'] },
        { name: 'All', extensions: ['*'] }
      ]
    })
    if (canceled || !filePath) return { error: 'Cancelled' }
    return callEngine({ command: 'save', temp_path: tempPath, save_path: filePath })
  })

  // ── Install AI Package ──
  ipcMain.handle('install-package', async (_e, packageName) => {
    // Whitelist of allowed packages (never let frontend run arbitrary pip commands)
    const ALLOWED_PACKAGES = {
      'realesrgan': ['realesrgan', 'basicsr'],
      'gfpgan': ['gfpgan'],
      'rembg': ['rembg', 'onnxruntime'],
    }

    const packages = ALLOWED_PACKAGES[packageName]
    if (!packages) return { error: `Unknown package: ${packageName}` }

    // Find pip in venv, or fall back to system
    const venv = venvPython()
    const pipBin = venv ? venv.replace(/python[23]?$/, 'pip') : 'pip3'

    // Build pip args: upgrade pip/tools first, then install packages.
    const upgradeArgs = ['install', '--upgrade', 'pip', 'setuptools', 'wheel']

    // For packages that historically fail to build from source (basicsr), prefer binary wheels
    const forceBinary = new Set(['basicsr'])
    const installArgs = ['install']
    for (const p of packages) {
      installArgs.push(p)
    }
    if (packages.some((p) => forceBinary.has(p))) {
      installArgs.push('--only-binary', ':all:')
    }

    return new Promise((resolve) => {
      // Helper to spawn pip and collect output
      const run = (args, cb) => {
        const child = spawn(pipBin, args, { timeout: 300_000 })
        let stdout = ''
        let stderr = ''
        child.stdout.on('data', (d) => { stdout += d.toString() })
        child.stderr.on('data', (d) => { stderr += d.toString() })
        child.on('error', (err) => cb(err, null))
        child.on('close', (code) => cb(null, { code, stdout, stderr }))
      }

      // Step 1: ensure build tools are present
      run(upgradeArgs, (err, res) => {
        if (err) return resolve({ error: `Failed to run pip: ${err.message}` })
        if (res.code !== 0) {
          // Non-fatal: continue to install but surface a hint
          // fallthrough to install
        }

        // Step 2: attempt install with preferred binary flag when appropriate
        run(installArgs, async (err2, res2) => {
          if (err2) return resolve({ error: `Failed to run pip: ${err2.message}` })
          if (res2.code === 0) {
            // Refresh capabilities from engine if possible
            try {
              const caps = await callEngine({ command: 'capabilities' }, ENGINE_TIMEOUT_QUICK_MS)
              return resolve({ status: 'installed', package: packageName, output: res2.stdout.slice(-500), capabilities: caps })
            } catch (e) {
              return resolve({ status: 'installed', package: packageName, output: res2.stdout.slice(-500) })
            }
          }

          // Install failed — analyze stderr for common causes and attempt fallbacks
          const stderr = res2.stderr || ''
          const hint = {}

          // If pip reports dependency resolution conflicts, try sequential installs and legacy resolver
          const isResolutionError = /ResolutionImpossible/i.test(stderr)
          if (isResolutionError) {
            // Try installing packages one-by-one (preferring binary wheels for known bad-builds)
            const runP = (args) => new Promise((rescb) => run(args, (err, r) => rescb({ err, r })))
            let seqFailed = false
            let seqOutput = ''
            for (const p of packages) {
              const argsP = ['install', p]
              if (forceBinary.has(p)) argsP.push('--only-binary', ':all:')
              /* eslint-disable no-await-in-loop */
              const { err: e1, r: r1 } = await runP(argsP)
              seqOutput += `\n=== ${p} (code=${r1?.code}) ===\n` + (r1?.stdout || '') + (r1?.stderr || '')
              if (e1 || r1.code !== 0) seqFailed = true
            }
            if (!seqFailed) {
              try {
                const caps = await callEngine({ command: 'capabilities' }, ENGINE_TIMEOUT_QUICK_MS)
                return resolve({ status: 'installed', package: packageName, output: seqOutput.slice(-500), capabilities: caps })
              } catch (e) {
                return resolve({ status: 'installed', package: packageName, output: seqOutput.slice(-500) })
              }
            }

            // As a last resort try legacy resolver (less strict dependency solving)
            const legacyArgs = ['install', ...packages, '--use-deprecated=legacy-resolver']
            run(legacyArgs, (err3, res3) => {
              if (err3) return resolve({ error: `Failed to run pip: ${err3.message}` })
              if (res3.code === 0) {
                try {
                  callEngine({ command: 'capabilities' }, ENGINE_TIMEOUT_QUICK_MS).then((caps) => resolve({ status: 'installed', package: packageName, output: res3.stdout.slice(-500), capabilities: caps })).catch(() => resolve({ status: 'installed', package: packageName, output: res3.stdout.slice(-500) }))
                } catch {
                  return resolve({ status: 'installed', package: packageName, output: res3.stdout.slice(-500) })
                }
              } else {
                hint.message = 'Dependency conflict detected. Try creating a fresh virtual environment and installing packages individually, or use a Python version compatible with these packages.'
                return resolve({ error: `pip install failed (code ${res3.code}): ${(res3.stderr || '').slice(-1000)}`, hint })
              }
            })
            return
          }

          if (/Failed to build/i.test(stderr) || /error: command 'x86_64-linux-gnu-gcc'|clang: error/i.test(stderr)) {
            hint.message = 'Build tools missing. Install a C compiler and python dev headers (e.g. build-essential, Xcode command line tools)'
          }
          if (/KeyError: '__version__'/i.test(stderr) || /Failed to build 'basicsr'/i.test(stderr)) {
            hint.message = 'basicsr failed to build. Try installing with binary wheels: pip install basicsr --only-binary :all:'
          }
          if (!hint.message) hint.message = 'pip install failed. See pip output for details.'

          return resolve({ error: `pip install failed (code ${res2.code}): ${stderr.slice(-1000)}`, hint })
        })
      })
    })
  })

  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
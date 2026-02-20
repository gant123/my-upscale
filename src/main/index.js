import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { spawn } from 'child_process'
import fs from 'fs'
import path from 'path'

const ALLOWED_IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'])
const ALLOWED_ENHANCE_MODES = new Set(['pro', 'color', 'smooth', 'light', 'portrait'])
const ALLOWED_XRAY_MODES = new Set([
  'none',
  'structure',
  'depth',
  'frequency',
  'thermal',
  'bones',
  'reveal',
  'bright',
  'occlusion'
])
const MAX_IMAGE_SIZE_BYTES = 50 * 1024 * 1024
const ENGINE_TIMEOUT_MS = 30_000

function getMime(p) {
  const ext = p.split('.').pop().toLowerCase()
  return (
    { jpg: 'jpeg', jpeg: 'jpeg', png: 'png', webp: 'webp', bmp: 'bmp', tiff: 'tiff' }[ext] || 'png'
  )
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
  const boundedNumber = (value, min, max, fallback = 0) => {
    const n = Number(value)
    if (!Number.isFinite(n)) return fallback
    return Math.max(min, Math.min(max, n))
  }

  return {
    exposure: boundedNumber(params.exposure, -3, 3),
    contrast: boundedNumber(params.contrast, -100, 100),
    highlights: boundedNumber(params.highlights, -100, 100),
    shadows: boundedNumber(params.shadows, -100, 100),
    whites: boundedNumber(params.whites, -100, 100),
    blacks: boundedNumber(params.blacks, -100, 100),
    temperature: boundedNumber(params.temperature, -100, 100),
    tint: boundedNumber(params.tint, -100, 100),
    vibrance: boundedNumber(params.vibrance, -100, 100),
    saturation: boundedNumber(params.saturation, -100, 100),
    clarity: boundedNumber(params.clarity, -100, 100),
    dehaze: boundedNumber(params.dehaze, 0, 100),
    sharpness: boundedNumber(params.sharpness, 0, 100),
    grain: boundedNumber(params.grain, 0, 100),
    vignette: boundedNumber(params.vignette, -100, 100),
    xray: ALLOWED_XRAY_MODES.has(params.xray) ? params.xray : 'none',
    xray_blend: boundedNumber(params.xray_blend, 0, 100, 100)
  }
}

/**
 * Every Python call goes through this one function.
 * Spawns engine.py, writes JSON to stdin, reads JSON from stdout.
 * No shell escaping issues ever.
 */
function callEngine(command) {
  const run = (binary) =>
    new Promise((resolve) => {
      const child = spawn(binary, ['engine.py'])
      let stdout = ''
      let stderr = ''
      let settled = false

      const finish = (value) => {
        if (!settled) {
          settled = true
          resolve(value)
        }
      }

      const timer = setTimeout(() => {
        child.kill('SIGKILL')
        finish({ error: 'Engine timed out' })
      }, ENGINE_TIMEOUT_MS)

      child.stdout.on('data', (d) => {
        stdout += d.toString()
      })
      child.stderr.on('data', (d) => {
        stderr += d.toString()
      })
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

  return run('python3').then((result) => {
    if (!result?.spawnError) return result
    if (result.spawnError.code !== 'ENOENT') {
      return { error: `Failed to start engine: ${result.spawnError.message}` }
    }
    return run('python').then((fallbackResult) => {
      if (!fallbackResult?.spawnError) return fallbackResult
      return { error: `Failed to start engine: ${fallbackResult.spawnError.message}` }
    })
  })
}

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 650,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#09090b',
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      sandbox: true,
      webSecurity: true,
      nodeIntegration: false
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

app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.electron')
  app.on('browser-window-created', (_, w) => optimizer.watchWindowShortcuts(w))

  // ─── File Picker ───
  ipcMain.handle('dialog:openFile', async () => {
    try {
      const { canceled, filePaths } = await dialog.showOpenDialog({
        properties: ['openFile'],
        filters: [{ name: 'Images', extensions: ['jpg', 'png', 'jpeg', 'webp', 'bmp', 'tiff'] }]
      })
      if (canceled) return null
      if (!isValidImagePath(filePaths[0])) return { error: 'Unsupported, missing, or too large image file' }
      return { path: filePaths[0], preview: toBase64(filePaths[0]) }
    } catch (error) {
      return { error: `Failed to open image: ${error.message}` }
    }
  })

  // ─── Analyze ───
  ipcMain.handle('run-autopilot', async (_e, imagePath) => {
    if (!isValidImagePath(imagePath)) return { error: 'Invalid image path or file type' }
    return callEngine({ command: 'analyze', image: imagePath })
  })

  // ─── Adjust (sliders + xray) ───
  ipcMain.handle('adjust-image', async (_e, imagePath, params) => {
    if (!isValidImagePath(imagePath)) return { error: 'Invalid image path or file type' }
    const result = await callEngine({
      command: 'adjust',
      image: imagePath,
      params: sanitizeAdjustParams(params)
    })
    if (result.temp_path) result.preview = toBase64(result.temp_path)
    return result
  })

  // ─── Enhance (layer stack) ───
  ipcMain.handle('enhance-image', async (_e, imagePath, modes) => {
    if (!isValidImagePath(imagePath)) return { error: 'Invalid image path or file type' }
    const safeModes = Array.isArray(modes)
      ? modes.filter((mode) => ALLOWED_ENHANCE_MODES.has(mode))
      : []
    const result = await callEngine({ command: 'enhance', image: imagePath, modes: safeModes })
    if (result.temp_path) result.preview = toBase64(result.temp_path)
    return result
  })

  // ─── Save As ───
  ipcMain.handle('save-image', async (_e, tempPath) => {
    if (typeof tempPath !== 'string' || !fs.existsSync(tempPath))
      return { error: 'Invalid temp file' }
    const ext = tempPath.split('.').pop().toLowerCase()
    const { canceled, filePath } = await dialog.showSaveDialog({
      defaultPath: `enhanced.${ext}`,
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

  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

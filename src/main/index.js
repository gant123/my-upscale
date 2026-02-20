import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { spawn } from 'child_process'
import fs from 'fs'

function getMime(p) {
  const ext = p.split('.').pop().toLowerCase()
  return { jpg: 'jpeg', jpeg: 'jpeg', png: 'png', webp: 'webp', bmp: 'bmp', tiff: 'tiff' }[ext] || 'png'
}

function toBase64(p) {
  return `data:image/${getMime(p)};base64,${fs.readFileSync(p).toString('base64')}`
}

/**
 * Every Python call goes through this one function.
 * Spawns engine.py, writes JSON to stdin, reads JSON from stdout.
 * No shell escaping issues ever.
 */
function callEngine(command) {
  return new Promise((resolve) => {
    const child = spawn('python', ['engine.py'])
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', (d) => { stdout += d.toString() })
    child.stderr.on('data', (d) => { stderr += d.toString() })
    child.on('close', (code) => {
      if (code !== 0 && !stdout.includes('{')) {
        return resolve({ error: stderr || `Engine exited with code ${code}` })
      }
      try {
        const s = stdout.indexOf('{')
        const e = stdout.lastIndexOf('}')
        if (s === -1 || e === -1) throw new Error('No JSON')
        resolve(JSON.parse(stdout.substring(s, e + 1)))
      } catch {
        console.error('Engine raw output:', stdout, 'stderr:', stderr)
        resolve({ error: 'Failed to parse engine output' })
      }
    })
    child.stdin.write(JSON.stringify(command))
    child.stdin.end()
  })
}

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1400, height: 900, minWidth: 1000, minHeight: 650,
    show: false, autoHideMenuBar: true, backgroundColor: '#09090b',
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false, webSecurity: false,
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
    const { canceled, filePaths } = await dialog.showOpenDialog({
      properties: ['openFile'],
      filters: [{ name: 'Images', extensions: ['jpg', 'png', 'jpeg', 'webp', 'bmp', 'tiff'] }]
    })
    if (canceled) return null
    return { path: filePaths[0], preview: toBase64(filePaths[0]) }
  })

  // ─── Analyze ───
  ipcMain.handle('run-autopilot', async (_e, imagePath) => {
    return callEngine({ command: 'analyze', image: imagePath })
  })

  // ─── Adjust (sliders + xray) ───
  ipcMain.handle('adjust-image', async (_e, imagePath, params) => {
    const result = await callEngine({ command: 'adjust', image: imagePath, params })
    if (result.temp_path) result.preview = toBase64(result.temp_path)
    return result
  })

  // ─── Enhance (layer stack) ───
  ipcMain.handle('enhance-image', async (_e, imagePath, modes) => {
    const result = await callEngine({ command: 'enhance', image: imagePath, modes })
    if (result.temp_path) result.preview = toBase64(result.temp_path)
    return result
  })

  // ─── Save As ───
  ipcMain.handle('save-image', async (_e, tempPath) => {
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
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow() })
})

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })
import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

console.log('[preload] Aurora Ops v2 booted ✅')

const api = {
  // System
  getCapabilities: () => ipcRenderer.invoke('get-capabilities'),

  // File I/O
  selectImage: () => ipcRenderer.invoke('dialog:openFile'),
  selectImages: () => ipcRenderer.invoke('dialog:openFiles'),
  saveImage: (tempPath) => ipcRenderer.invoke('save-image', tempPath),
  installPackage: (packageName) => ipcRenderer.invoke('install-package', packageName),
  // Classical
  runAutopilot: (imagePath) => ipcRenderer.invoke('run-autopilot', imagePath),
  adjustImage: (imagePath, params) => ipcRenderer.invoke('adjust-image', imagePath, params),
  enhanceImage: (imagePath, modes) => ipcRenderer.invoke('enhance-image', imagePath, modes),

  // AI Features
  upscaleImage: (imagePath, scale) => ipcRenderer.invoke('upscale-image', imagePath, scale),
  faceRestore: (imagePath, model, fidelity) => ipcRenderer.invoke('face-restore', imagePath, model, fidelity),
  bgRemove: (imagePath, bgColor) => ipcRenderer.invoke('bg-remove', imagePath, bgColor),
  inpaint: (imagePath, maskPath, method) => ipcRenderer.invoke('inpaint', imagePath, maskPath, method),
  autoEnhance: (imagePath) => ipcRenderer.invoke('auto-enhance', imagePath),

  // Batch
  batchProcess: (jobs) => ipcRenderer.invoke('batch-process', jobs),
}

if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
    contextBridge.exposeInMainWorld('__preload_ok', true)
  } catch (error) {
    console.error(error)
  }
} else {
  window.electron = electronAPI
  window.api = api
}
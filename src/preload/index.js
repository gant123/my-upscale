import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

const api = {
  selectImage: () => ipcRenderer.invoke('dialog:openFile'),
  runAutopilot: (imagePath) => ipcRenderer.invoke('run-autopilot', imagePath),
  adjustImage: (imagePath, params) => ipcRenderer.invoke('adjust-image', imagePath, params),
  enhanceImage: (imagePath, modes) => ipcRenderer.invoke('enhance-image', imagePath, modes),
  saveImage: (tempPath) => ipcRenderer.invoke('save-image', tempPath),
}

if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
  } catch (error) {
    console.error(error)
  }
} else {
  window.electron = electronAPI
  window.api = api
}
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'

const ENHANCE_MODES = [
  { id: 'pro', label: 'Pro Detail', desc: 'Sharpness + micro-contrast', icon: '◈' },
  { id: 'color', label: 'Color Restore', desc: 'Recover faded colors', icon: '◉' },
  { id: 'smooth', label: 'Denoise', desc: 'Reduce noise artifacts', icon: '○' },
  { id: 'light', label: 'Lighting Fix', desc: 'Shadow/highlight recovery', icon: '☀' },
  { id: 'portrait', label: 'Portrait', desc: 'Skin-safe face tuning', icon: '◎' }
]
const XRAY_MODES = [
  { id: 'none', label: 'Off' }, { id: 'structure', label: 'Structure' },
  { id: 'depth', label: 'Depth' }, { id: 'frequency', label: 'Frequency' },
  { id: 'thermal', label: 'Thermal' }, { id: 'bones', label: 'Bones' },
  { id: 'reveal', label: 'Reveal' }, { id: 'bright', label: 'Bright' },
  { id: 'occlusion', label: 'Occlusion' }
]
const DEFAULTS = {
  exposure: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0,
  temperature: 0, tint: 0, vibrance: 0, saturation: 0,
  clarity: 0, dehaze: 0, sharpness: 0, grain: 0, vignette: 0,
  xray: 'none', xray_blend: 100
}
const TABS = [
  { id: 'adjust', label: 'Adjust', icon: '⊞' }, { id: 'enhance', label: 'Enhance', icon: '◈' },
  { id: 'ai', label: 'AI Tools', icon: '✦' }, { id: 'analyze', label: 'Analyze', icon: '◎' },
  { id: 'export', label: 'Export', icon: '↗' }
]

export default function App() {
  const [originalPath, setOriginalPath] = useState(null)
  const [originalPreview, setOriginalPreview] = useState(null)
  const [resultPreview, setResultPreview] = useState(null)
  const [tempPath, setTempPath] = useState(null)
  const [resultLabel, setResultLabel] = useState('')
  const [adj, setAdj] = useState({ ...DEFAULTS })
  const [activeLayers, setActiveLayers] = useState(['pro'])
  const [liveAdjust, setLiveAdjust] = useState(true)
  const [diagnosis, setDiagnosis] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [rec, setRec] = useState(null)
  const [caps, setCaps] = useState(null)
  const [installHints, setInstallHints] = useState({})
  const [capsError, setCapsError] = useState(null)
  const [tab, setTab] = useState('adjust')
  const [busy, setBusy] = useState(false)
  const [busyMsg, setBusyMsg] = useState('')
  const [saving, setSaving] = useState(false)
  const [uiError, setUiError] = useState('')
  const [toast, setToast] = useState(null)
  const [holdOriginal, setHoldOriginal] = useState(false)
  const [splitPos, setSplitPos] = useState(55)
  const [zoom, setZoom] = useState(1)
  const [fitMode, setFitMode] = useState('contain')
  const [installing, setInstalling] = useState(null)

  const fileName = useMemo(() => originalPath?.split(/[/\\]/).pop(), [originalPath])
  const dirty = useMemo(() => Object.keys(DEFAULTS).some((k) => adj[k] !== DEFAULTS[k]), [adj])
  const hasResult = !!resultPreview && !!tempPath

  const toastTimer = useRef(0)
  const pushToast = useCallback((type, title, message) => {
    setToast({ type, title, message })
    clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(null), 4000)
  }, [])

  /* ── FIX #1: Capabilities with error handling + retry ── */
  const loadCaps = useCallback(async () => {
    if (!window.api) { setCapsError('Backend bridge unavailable'); return }
    if (!window.api.getCapabilities) { setCapsError('getCapabilities missing from preload'); return }
    try {
      const r = await window.api.getCapabilities()
      if (r?.error) { setCapsError(r.error); setCaps({}); pushToast('error', 'Engine', r.error) }
      else if (r?.capabilities) { setCaps(r.capabilities); setInstallHints(r.install_hints || {}); setCapsError(null) }
      else { setCapsError('Unexpected response'); setCaps({}) }
    } catch (err) {
      setCapsError(err?.message || 'Failed to reach engine')
      setCaps({})
      pushToast('error', 'Engine', err?.message || 'Could not load capabilities')
    }
  }, [pushToast])

  useEffect(() => { loadCaps() }, [loadCaps])

  useEffect(() => {
    const d = (e) => {
      if (e.code === 'Space' && !e.target?.matches?.('input,textarea,button')) { e.preventDefault(); setHoldOriginal(true) }
      if ((e.ctrlKey || e.metaKey) && e.key === 'o') { e.preventDefault(); handleOpen() }
      if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); if (tempPath) handleSave() }
      if (e.key === 'Escape') { setUiError(''); setToast(null) }
    }
    const u = (e) => { if (e.code === 'Space') { e.preventDefault(); setHoldOriginal(false) } }
    window.addEventListener('keydown', d); window.addEventListener('keyup', u)
    return () => { window.removeEventListener('keydown', d); window.removeEventListener('keyup', u) }
  }, [tempPath])

  /* ── FIX #2: runCommand surfaces ALL errors to UI ── */
  const runCommand = useCallback(async (label, apiFn) => {
    setBusy(true); setBusyMsg(label); setUiError('')
    try {
      const r = await apiFn()
      if (r?.error) { setUiError(r.error); pushToast('error', label, r.error); return null }
      return r
    } catch (err) {
      const msg = err?.message || 'Unknown error'
      setUiError(msg); pushToast('error', label, msg); return null
    } finally { setBusy(false); setBusyMsg('') }
  }, [pushToast])

  const handleOpen = useCallback(async () => {
    setUiError('')
    try {
      const f = await window.api?.selectImage()
      if (!f) return
      if (f.error) { setUiError(f.error); return }
      setOriginalPath(f.path); setOriginalPreview(f.preview)
      setResultPreview(null); setTempPath(null); setResultLabel('')
      setDiagnosis(null); setMetrics(null); setRec(null)
      setAdj({ ...DEFAULTS }); setActiveLayers(['pro']); setSplitPos(55); setZoom(1); setTab('adjust')
      pushToast('ok', 'Loaded', f.path.split(/[/\\]/).pop())
    } catch (err) { setUiError(err?.message || 'Open failed') }
  }, [pushToast])

  const handleAnalyze = useCallback(async () => {
    if (!originalPath) return
    const r = await runCommand('Analyzing', () => window.api.runAutopilot(originalPath))
    if (!r) return
    if (r.recommendations) {
      setDiagnosis(r.analysis); setMetrics(r.metrics); setRec(r.recommendations)
      if (r.recommendations.best_mode) setActiveLayers([r.recommendations.best_mode])
      setAdj((p) => ({ ...p, exposure: r.recommendations.exposure || 0 }))
      if (r.capabilities) setCaps(r.capabilities)
      setTab('analyze')
      pushToast('ok', 'Analyzed', (r.recommendations.best_mode || '').toUpperCase() + ' recommended')
    }
  }, [originalPath, runCommand, pushToast])

  const handleApply = useCallback(async () => {
    if (!originalPath || !dirty) return
    const r = await runCommand('Applying', () => window.api.adjustImage(originalPath, adj))
    if (r?.temp_path) { setTempPath(r.temp_path); setResultPreview(r.preview); setResultLabel('Adjusted'); pushToast('ok', 'Applied', '') }
  }, [adj, dirty, originalPath, runCommand, pushToast])

  const handleReset = useCallback(() => { setAdj({ ...DEFAULTS }); pushToast('ok', 'Reset', '') }, [pushToast])
  const setA = useCallback((k, v) => setAdj((p) => ({ ...p, [k]: v })), [])
  const toggleLayer = useCallback((id) => {
    setActiveLayers((p) => { if (p.includes(id)) { const n = p.filter((m) => m !== id); return n.length ? n : p }; return [...p, id] })
  }, [])

  const handleEnhance = useCallback(async () => {
    if (!originalPath) return
    const r = await runCommand('Enhancing', () => window.api.enhanceImage(originalPath, activeLayers))
    if (r?.temp_path) { setTempPath(r.temp_path); setResultPreview(r.preview); setResultLabel((r.applied_modes || activeLayers).join(' + ')); pushToast('ok', 'Enhanced', 'Done') }
  }, [activeLayers, originalPath, runCommand, pushToast])

  const handleAutoEnhance = useCallback(async () => {
    if (!originalPath) return
    const r = await runCommand('Auto-Enhancing', () => window.api.autoEnhance(originalPath))
    if (r?.temp_path) {
      setTempPath(r.temp_path); setResultPreview(r.preview)
      const steps = (r.steps || []).map((s) => s.step).join(' → ')
      setResultLabel('Auto: ' + (steps || 'enhanced'))
      if (r.metrics) setMetrics(r.metrics)
      pushToast('ok', 'Auto-enhanced', steps || 'Done')
    }
  }, [originalPath, runCommand, pushToast])

  const handleUpscale = useCallback(async (scale = 4) => {
    const src = tempPath || originalPath; if (!src) return
    const r = await runCommand('Upscaling ' + scale + '×', () => window.api.upscaleImage(src, scale))
    if (r?.temp_path) { setTempPath(r.temp_path); setResultPreview(r.preview); setResultLabel('Upscaled ' + scale + '×'); pushToast('ok', 'Upscaled', (r.output_size?.width || '') + '×' + (r.output_size?.height || '')) }
  }, [tempPath, originalPath, runCommand, pushToast])

  const handleFaceRestore = useCallback(async () => {
    const src = tempPath || originalPath; if (!src) return
    const r = await runCommand('Restoring faces', () => window.api.faceRestore(src, 'gfpgan', 0.7))
    if (r?.temp_path) { setTempPath(r.temp_path); setResultPreview(r.preview); setResultLabel('Face Restored'); pushToast('ok', 'Faces restored', '') }
  }, [tempPath, originalPath, runCommand, pushToast])

  const handleBgRemove = useCallback(async () => {
    const src = tempPath || originalPath; if (!src) return
    const r = await runCommand('Removing background', () => window.api.bgRemove(src))
    if (r?.temp_path) { setTempPath(r.temp_path); setResultPreview(r.preview); setResultLabel('BG Removed'); pushToast('ok', 'BG removed', '') }
  }, [tempPath, originalPath, runCommand, pushToast])

  const handleSave = useCallback(async () => {
    if (!tempPath) return; setSaving(true)
    try {
      const r = await window.api.saveImage(tempPath)
      if (r?.saved_path) { pushToast('ok', 'Saved', r.saved_path.split(/[/\\]/).pop()); setSaving('done'); setTimeout(() => setSaving(false), 1600) }
      else { if (r?.error !== 'Cancelled') setUiError(r?.error || 'Save failed'); setSaving(false) }
    } catch (e) { setUiError(e?.message || 'Save error'); setSaving(false) }
  }, [tempPath, pushToast])

  /* ── FIX #3: Install refreshes caps properly ── */
  const handleInstall = useCallback(async (packageKey) => {
    if (!window.api?.installPackage) { pushToast('error', 'Missing', 'installPackage not in preload'); return }
    setInstalling(packageKey)
    pushToast('ok', 'Installing', 'Installing ' + packageKey + '…')
    try {
      const r = await window.api.installPackage(packageKey)
      if (r?.error) { pushToast('error', 'Install failed', r.error.substring(0, 120)) }
      else { pushToast('ok', 'Installed!', packageKey + ' is ready'); await loadCaps() }
    } catch (e) { pushToast('error', 'Install error', e?.message || 'Unknown') }
    finally { setInstalling(null) }
  }, [pushToast, loadCaps])

  /* ── Live adjust with proper error surfacing ── */
  const adjTimer = useRef(null), adjSeq = useRef(0), adjBusy = useRef(false)
  useEffect(() => {
    if (!liveAdjust || !originalPath || !dirty) return
    if (adjTimer.current) clearTimeout(adjTimer.current)
    adjTimer.current = setTimeout(async () => {
      if (adjBusy.current) return
      const seq = ++adjSeq.current; adjBusy.current = true; setBusy(true); setBusyMsg('Adjusting')
      try {
        const r = await window.api.adjustImage(originalPath, adj)
        if (seq !== adjSeq.current) return
        if (r?.error) setUiError(r.error)
        else if (r?.temp_path) { setTempPath(r.temp_path); setResultPreview(r.preview); setResultLabel('Live') }
      } catch (err) { if (seq === adjSeq.current) setUiError(err?.message || 'Adjust failed') }
      finally { if (seq === adjSeq.current) { setBusy(false); setBusyMsg('') }; adjBusy.current = false }
    }, 200)
    return () => clearTimeout(adjTimer.current)
  }, [adj, dirty, originalPath, liveAdjust])

  /* ── FIX #4: Computed caps with null-safe checks ── */
  const capsLoaded = caps !== null
  const aiCaps = useMemo(() => ({
    upscale: !!caps?.upscale_realesrgan,
    face: !!caps?.face_gfpgan || !!caps?.face_codeformer,
    bg: !!caps?.bg_remove
  }), [caps])
  const anyAi = aiCaps.upscale || aiCaps.face || aiCaps.bg
  const pipeline = useMemo(() => activeLayers.join(' → '), [activeLayers])
  const status = useMemo(() => {
    if (!originalPath) return { label: 'Ready', tone: 'dim' }
    if (busy) return { label: busyMsg || 'Working', tone: 'amber' }
    if (hasResult) return { label: 'Result', tone: 'green' }
    if (dirty) return { label: 'Editing', tone: 'cyan' }
    return { label: 'Loaded', tone: 'cyan' }
  }, [busy, busyMsg, dirty, hasResult, originalPath])

  /* ═══════════════════ RENDER ═══════════════════ */
  return (
    <div style={S.app}>
      <header style={S.topBar}>
        <div style={S.topLeft}>
          <div style={S.logo}>A</div>
          <div><div style={S.title}>Aurora Ops</div><div style={S.sub}>{fileName || 'AI Image Engine'}</div></div>
        </div>
        <div style={S.topRight}>
          <StatusPill label={status.label} tone={status.tone} />
          <TopBtn onClick={handleOpen}>Open</TopBtn>
          <TopBtn onClick={handleAnalyze} disabled={!originalPath || busy}>Analyze</TopBtn>
          <TopBtn onClick={handleAutoEnhance} disabled={!originalPath || busy} accent>✦ Auto Fix</TopBtn>
          <TopBtn onClick={handleSave} disabled={!tempPath || saving === true}>{saving === 'done' ? '✓' : saving ? '…' : 'Save'}</TopBtn>
          <div style={S.zoomGroup}>
            <ZoomBtn onClick={() => setFitMode((m) => m === 'contain' ? 'cover' : 'contain')}>{fitMode === 'contain' ? 'Fit' : 'Fill'}</ZoomBtn>
            <ZoomBtn onClick={() => setZoom((z) => Math.min(2.5, +(z + 0.15).toFixed(2)))}>+</ZoomBtn>
            <ZoomBtn onClick={() => setZoom((z) => Math.max(1, +(z - 0.15).toFixed(2)))}>−</ZoomBtn>
          </div>
        </div>
      </header>

      {uiError && <div style={S.errBar}><span style={{ fontWeight: 800 }}>ERROR</span><span style={{ flex: 1, opacity: 0.9, marginLeft: 8 }}>{uiError}</span><button onClick={() => setUiError('')} style={S.errX}>✕</button></div>}

      <div style={S.body}>
        <aside style={S.sidebar}>
          <div style={S.tabRow}>{TABS.map((t) => <button key={t.id} onClick={() => setTab(t.id)} style={{ ...S.tab, ...(tab === t.id ? S.tabOn : {}) }}><span style={S.tabIcon}>{t.icon}</span>{t.label}</button>)}</div>
          <div style={S.panelArea}>
            {/* ─── ADJUST ─── */}
            {tab === 'adjust' && <div style={S.pi}>
              <PH title="Adjust" sub="Lightroom-style controls" />
              <Toggle label="Live Preview" on={liveAdjust} flip={() => setLiveAdjust(!liveAdjust)} />
              {!liveAdjust && dirty && <ABtn onClick={handleApply} disabled={busy}>Apply</ABtn>}
              <GBtn onClick={handleReset} disabled={!dirty}>Reset</GBtn>
              <Sep /><SH>Tone</SH>
              <Sl l="Exposure" v={adj.exposure} mn={-3} mx={3} st={0.05} f={(v) => setA('exposure', v)} />
              <Sl l="Contrast" v={adj.contrast} mn={-100} mx={100} f={(v) => setA('contrast', v)} />
              <Sl l="Highlights" v={adj.highlights} mn={-100} mx={100} f={(v) => setA('highlights', v)} />
              <Sl l="Shadows" v={adj.shadows} mn={-100} mx={100} f={(v) => setA('shadows', v)} />
              <Sl l="Whites" v={adj.whites} mn={-100} mx={100} f={(v) => setA('whites', v)} />
              <Sl l="Blacks" v={adj.blacks} mn={-100} mx={100} f={(v) => setA('blacks', v)} />
              <Sep /><SH>Color</SH>
              <Sl l="Temperature" v={adj.temperature} mn={-100} mx={100} f={(v) => setA('temperature', v)} />
              <Sl l="Tint" v={adj.tint} mn={-100} mx={100} f={(v) => setA('tint', v)} />
              <Sl l="Vibrance" v={adj.vibrance} mn={-100} mx={100} f={(v) => setA('vibrance', v)} />
              <Sl l="Saturation" v={adj.saturation} mn={-100} mx={100} f={(v) => setA('saturation', v)} />
              <Sep /><SH>Detail</SH>
              <Sl l="Clarity" v={adj.clarity} mn={-100} mx={100} f={(v) => setA('clarity', v)} />
              <Sl l="Dehaze" v={adj.dehaze} mn={0} mx={100} f={(v) => setA('dehaze', v)} />
              <Sl l="Sharpness" v={adj.sharpness} mn={0} mx={100} f={(v) => setA('sharpness', v)} />
              <Sl l="Grain" v={adj.grain} mn={0} mx={100} f={(v) => setA('grain', v)} />
              <Sl l="Vignette" v={adj.vignette} mn={-100} mx={100} f={(v) => setA('vignette', v)} />
              <Sep /><SH>X-Ray</SH>
              <Pills items={XRAY_MODES} value={adj.xray} onChange={(id) => setA('xray', id)} />
              {adj.xray !== 'none' && <Sl l="Blend" v={adj.xray_blend} mn={0} mx={100} f={(v) => setA('xray_blend', v)} />}
            </div>}

            {/* ─── ENHANCE ─── */}
            {tab === 'enhance' && <div style={S.pi}>
              <PH title="Enhance" sub="Stackable layers" />
              <FR l="Pipeline" v={pipeline || '—'} /><Sep />
              {ENHANCE_MODES.map((m) => { const on = activeLayers.includes(m.id); return (
                <button key={m.id} onClick={() => toggleLayer(m.id)} style={{ ...S.lCard, ...(on ? S.lOn : {}) }}>
                  <span style={S.lIcon}>{m.icon}</span>
                  <div style={{ flex: 1 }}><div style={S.lName}>{m.label}</div><div style={S.lDesc}>{m.desc}</div></div>
                  <span style={{ ...S.badge, ...(on ? S.bOn : S.bOff) }}>{on ? 'ON' : 'OFF'}</span>
                </button>
              )})}
              <Sep />
              <ABtn onClick={handleEnhance} disabled={!originalPath || busy}>{busy && busyMsg === 'Enhancing' ? 'Processing…' : 'Run Enhancement'}</ABtn>
            </div>}

            {/* ─── AI TOOLS (FIX #5: proper state rendering) ─── */}
            {tab === 'ai' && <div style={S.pi}>
              <PH title="✦ AI Tools" sub="Neural network processing" />

              {!capsLoaded && <div style={S.sBox}><div style={S.spinner} /><span>Loading engine…</span></div>}
              {capsError && <div style={{ ...S.sBox, borderColor: 'rgba(239,68,68,0.3)' }}><span style={{ color: '#f87171', fontWeight: 800 }}>Engine: {capsError}</span><GBtn onClick={loadCaps}>Retry</GBtn></div>}

              <AIBox title="✦ Auto Fix" desc="Analyze → correct → enhance → face → upscale"
                status={anyAi ? 'ready' : 'basic'}
                hint={!anyAi && capsLoaded ? 'No AI packages — runs classical processing. Install below for neural enhancement.' : null}
                onClick={handleAutoEnhance} disabled={!originalPath || busy} accent />

              <AIBox title="AI Upscale" desc="Real-ESRGAN 2×/4× neural detail synthesis"
                status={aiCaps.upscale ? 'ready' : capsLoaded ? 'missing' : 'loading'}
                hint={installHints.upscale_realesrgan} packageKey={!aiCaps.upscale ? 'realesrgan' : null}
                onInstall={handleInstall} installing={installing}>
                {aiCaps.upscale && <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                  <GBtn onClick={() => handleUpscale(2)} disabled={!originalPath || busy}>2×</GBtn>
                  <ABtn onClick={() => handleUpscale(4)} disabled={!originalPath || busy}>4×</ABtn>
                </div>}
              </AIBox>

              <AIBox title="Face Restore" desc="GFPGAN: fix blurry/damaged faces"
                status={aiCaps.face ? 'ready' : capsLoaded ? 'missing' : 'loading'}
                hint={installHints.face_gfpgan} packageKey={!aiCaps.face ? 'gfpgan' : null}
                onInstall={handleInstall} installing={installing}
                onClick={aiCaps.face ? handleFaceRestore : null} disabled={!originalPath || busy} />

              <AIBox title="BG Remove" desc="U2-Net: transparent PNG output"
                status={aiCaps.bg ? 'ready' : capsLoaded ? 'missing' : 'loading'}
                hint={installHints.bg_remove} packageKey={!aiCaps.bg ? 'rembg' : null}
                onInstall={handleInstall} installing={installing}
                onClick={aiCaps.bg ? handleBgRemove : null} disabled={!originalPath || busy} />

              <AIBox title="Inpaint" desc="Paint mask to erase objects" status="soon" hint="Coming in a future update" />

              <Sep />
              <div style={S.hint}>AI tools process current result (or original). Chain: Adjust → Enhance → Upscale → Face.</div>
            </div>}

            {/* ─── ANALYZE ─── */}
            {tab === 'analyze' && <div style={S.pi}>
              <PH title="Analyze" sub="Diagnostics + suggestions" />
              <ABtn onClick={handleAnalyze} disabled={!originalPath || busy}>{busy && busyMsg === 'Analyzing' ? 'Analyzing…' : 'Run Analysis'}</ABtn>
              <Sep /><FR l="Diagnosis" v={diagnosis || '—'} /><Sep />
              <div style={S.mGrid}>
                <MC l="Noise" v={metrics?.noise ?? '—'} t={metrics?.noise > 15 ? 'red' : metrics?.noise > 8 ? 'amber' : 'green'} />
                <MC l="Sharp" v={metrics?.sharpness == null ? '—' : metrics.sharpness > 500 ? 'Hi' : metrics.sharpness > 100 ? 'Med' : 'Lo'} t={metrics?.sharpness > 500 ? 'green' : 'amber'} />
                <MC l="Range" v={metrics?.dynamic_range ?? '—'} t={metrics?.dynamic_range > 200 ? 'green' : 'amber'} />
                <MC l="Faces" v={metrics?.faces?.length ?? '—'} t={metrics?.faces?.length > 0 ? 'cyan' : 'dim'} />
                <MC l="Size" v={metrics ? metrics.width + '×' + metrics.height : '—'} t="dim" />
                <MC l="Skin" v={metrics?.skin_pct != null ? metrics.skin_pct + '%' : '—'} t="dim" />
              </div>
              {rec?.ai_suggestions?.length > 0 && <><Sep /><SH>AI Suggestions</SH>{rec.ai_suggestions.map((s, i) => <div key={i} style={S.sugCard}><div style={S.sugT}>{s.action.replace('_', ' ')}</div><div style={S.sugX}>{s.reason}</div></div>)}</>}
              {rec?.best_mode && <><Sep /><div style={S.recCard}><FR l="Best Mode" v={rec.best_mode.toUpperCase()} /><div style={S.recW}>{rec.mode_reason}</div><GBtn onClick={() => { setActiveLayers([rec.best_mode]); setTab('enhance') }}>Apply → Enhance</GBtn></div></>}
            </div>}

            {/* ─── EXPORT ─── */}
            {tab === 'export' && <div style={S.pi}>
              <PH title="Export" sub="Save processed image" />
              <FR l="Result" v={resultLabel || (hasResult ? 'Ready' : '—')} /><Sep />
              <ABtn onClick={handleSave} disabled={!tempPath || saving === true}>{saving === 'done' ? '✓ Saved' : saving ? 'Saving…' : 'Save As…'}</ABtn>
              <GBtn onClick={() => setTab('adjust')}>Back to Editing</GBtn>
            </div>}
          </div>
        </aside>

        <main style={S.canvas}>
          {busy && <div style={S.busyOvl}><div style={S.spinner} /><span style={{ fontWeight: 800 }}>{busyMsg}…</span></div>}
          {!originalPath ? <Empty onOpen={handleOpen} /> : <Compare orig={originalPreview} result={resultPreview} hold={holdOriginal} sp={splitPos} setSp={setSplitPos} has={hasResult} lbl={resultLabel} z={zoom} fit={fitMode} />}
        </main>
      </div>
      {toast && <Toast t={toast} />}
    </div>
  )
}

/* ═══════════════════ SUB-COMPONENTS ═══════════════════ */

const PH = ({ title, sub }) => <div style={S.ph}><div style={S.phT}>{title}</div><div style={S.phS}>{sub}</div></div>
const Sep = () => <div style={S.sep} />
const SH = ({ children }) => <div style={S.sh}>{children}</div>
const FR = ({ l, v }) => <div style={S.fr}><span style={S.frL}>{l}</span><span style={S.frV}>{v}</span></div>
const Toggle = ({ label, on, flip }) => <div style={S.tog}><span style={S.togL}>{label}</span><button onClick={flip} style={{ ...S.togBtn, ...(on ? S.togOn : {}) }}>{on ? 'ON' : 'OFF'}</button></div>
const ABtn = ({ children, onClick, disabled, style: sx }) => <button onClick={onClick} disabled={disabled} style={{ ...S.aBtn, ...(disabled ? S.dis : {}), ...sx }}>{children}</button>
const GBtn = ({ children, onClick, disabled }) => <button onClick={onClick} disabled={disabled} style={{ ...S.gBtn, ...(disabled ? S.dis : {}) }}>{children}</button>
const TopBtn = ({ children, onClick, disabled, accent }) => <button onClick={onClick} disabled={disabled} style={{ ...S.tBtn, ...(accent ? S.tBtnA : {}), ...(disabled ? S.dis : {}) }}>{children}</button>
const ZoomBtn = ({ children, onClick }) => <button onClick={onClick} style={S.zBtn}>{children}</button>
const StatusPill = ({ label, tone }) => { const c = { dim: '#94a3b8', cyan: '#67e8f9', amber: '#fcd34d', green: '#86efac' }[tone] || '#94a3b8'; return <div style={{ ...S.sp, color: c, borderColor: c + '44', background: c + '18' }}>{label}</div> }

const Pills = ({ items, value, onChange }) => <div style={S.pills}>{items.map((i) => <button key={i.id} onClick={() => onChange(i.id)} style={{ ...S.pBtn, ...(value === i.id ? S.pOn : {}) }}>{i.label}</button>)}</div>

const Sl = ({ l, v, mn, mx, st = 1, f }) => {
  const ctr = mn < 0, pct = ((v - mn) / (mx - mn)) * 100, cp = ctr ? ((0 - mn) / (mx - mn)) * 100 : 0
  const left = ctr ? Math.min(pct, cp) : 0, width = ctr ? Math.abs(pct - cp) : pct
  return <div style={S.slRow}><div style={S.slHead}><span style={S.slL}>{l}</span><span style={S.slV}>{st < 1 ? (+v).toFixed(2) : Math.round(v)}</span></div><div style={S.slTrack}><div style={S.slBg} /><div style={{ ...S.slFill, left: left + '%', width: width + '%' }} />{ctr && <div style={{ ...S.slMid, left: cp + '%' }} />}<input type="range" min={mn} max={mx} step={st} value={v} onChange={(e) => f(parseFloat(e.target.value))} onDoubleClick={() => f(ctr ? 0 : mn)} style={S.slIn} /></div></div>
}

/* ── FIX #6: AIBox properly shows state: ready/basic/missing/loading/soon ── */
const AIBox = ({ title, desc, status, hint, onClick, disabled, children, accent, packageKey, onInstall, installing }) => {
  const bMap = {
    ready:   { text: 'READY',   s: S.bOn },
    basic:   { text: 'BASIC',   s: S.bWarn },
    missing: { text: 'INSTALL', s: S.bOff },
    loading: { text: '…',       s: S.bOff },
    soon:    { text: 'SOON',    s: S.bOff },
  }
  const b = bMap[status] || bMap.missing
  const usable = status === 'ready' || status === 'basic'
  const showInstall = status === 'missing' && packageKey && onInstall
  return (
    <div style={{ ...S.aiBox, opacity: status === 'soon' ? 0.5 : 1 }}>
      <div style={S.aiTop}>
        <div><div style={S.aiT}>{title}</div><div style={S.aiD}>{desc}</div></div>
        <span style={{ ...S.badge, ...b.s }}>{b.text}</span>
      </div>
      {hint && <div style={S.aiH}>{hint}</div>}
      {showInstall && <ABtn onClick={() => onInstall(packageKey)} disabled={installing === packageKey} style={{ marginTop: 8 }}>{installing === packageKey ? 'Installing… please wait' : 'Install ' + packageKey}</ABtn>}
      {usable && children}
      {usable && !children && onClick && <ABtn onClick={onClick} disabled={disabled} style={{ marginTop: 8 }}>{accent ? 'Run Auto Fix' : 'Run ' + title}</ABtn>}
    </div>
  )
}

const MC = ({ l, v, t }) => { const c = { dim: '#64748b', cyan: '#67e8f9', amber: '#fbbf24', green: '#4ade80', red: '#f87171' }[t] || '#64748b'; return <div style={{ ...S.mc, borderColor: c + '33' }}><div style={S.mcL}>{l}</div><div style={{ ...S.mcV, color: c }}>{v}</div></div> }

const Empty = ({ onOpen }) => <div style={S.empty}><div style={S.emI}>✦</div><div style={S.emT}>Aurora Ops</div><div style={S.emS}>Load an image to begin</div><div style={{ height: 16 }} /><ABtn onClick={onOpen}>Open Image</ABtn></div>

const Compare = ({ orig, result, hold, sp, setSp, has, lbl, z, fit }) => {
  const show = has && !hold
  return (
    <div style={S.cInner}>
      <img src={orig} alt="" style={{ ...S.cImg, objectFit: fit, transform: 'scale(' + z + ')' }} />
      {show && <img src={result} alt="" style={{ ...S.cImg, objectFit: fit, transform: 'scale(' + z + ')', clipPath: 'polygon(0 0,' + sp + '% 0,' + sp + '% 100%,0 100%)' }} />}
      {show && <>
        <div style={{ ...S.spLine, left: sp + '%' }}><div style={S.spKnob}>⇆</div></div>
        <input type="range" min="0" max="100" value={sp} onChange={(e) => setSp(+e.target.value)} style={S.spIn} />
        <div style={S.tagL}>RESULT</div><div style={S.tagR}>ORIGINAL</div>
      </>}
      {!show && <div style={S.tagL}>{has ? 'ORIGINAL (Space)' : 'ORIGINAL'}</div>}
      {lbl && <div style={S.tagB}>{lbl.toUpperCase()}</div>}
      <div style={S.helpTag}>Hold <span style={S.hKey}>Space</span> compare · Drag split</div>
    </div>
  )
}

const Toast = ({ t }) => { const c = t.type === 'ok' ? '#4ade80' : t.type === 'error' ? '#f87171' : '#60a5fa'; return <div style={{ ...S.toast, borderColor: c + '44' }}><span style={{ fontWeight: 900, color: c }}>{t.title}</span><span style={{ opacity: 0.9 }}>{t.message}</span></div> }

/* ═══════════════════ STYLES ═══════════════════ */

const L = 'rgba(148,163,184,0.14)'
const B = '#60a5fa'

const S = {
  app: { height: '100vh', background: '#070b14', color: '#e2e8f0', fontFamily: "system-ui,-apple-system,'Segoe UI',Roboto,sans-serif", display: 'flex', flexDirection: 'column', overflow: 'hidden' },
  topBar: { height: 52, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 14px', borderBottom: '1px solid ' + L, background: 'rgba(15,23,42,0.92)', backdropFilter: 'blur(8px)', WebkitAppRegion: 'drag' },
  topLeft: { display: 'flex', alignItems: 'center', gap: 10 },
  logo: { width: 30, height: 30, borderRadius: 10, background: 'linear-gradient(135deg,' + B + ',#2563eb)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 950, fontSize: 15, color: '#fff' },
  title: { fontSize: 13, fontWeight: 900, letterSpacing: 0.3 },
  sub: { fontSize: 11, color: '#64748b', fontWeight: 700, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  topRight: { display: 'flex', alignItems: 'center', gap: 8, WebkitAppRegion: 'no-drag' },
  sp: { padding: '5px 10px', borderRadius: 999, border: '1px solid', fontSize: 11, fontWeight: 900 },
  tBtn: { height: 32, padding: '0 12px', borderRadius: 10, border: '1px solid ' + L, background: 'rgba(2,6,23,0.35)', color: '#e2e8f0', fontSize: 12, fontWeight: 800, cursor: 'pointer' },
  tBtnA: { background: 'rgba(59,130,246,0.18)', borderColor: 'rgba(59,130,246,0.30)', color: '#93c5fd' },
  zoomGroup: { display: 'flex', gap: 4 },
  zBtn: { width: 32, height: 32, borderRadius: 8, border: '1px solid ' + L, background: 'transparent', color: '#94a3b8', fontSize: 12, fontWeight: 900, cursor: 'pointer' },
  errBar: { display: 'flex', gap: 10, alignItems: 'center', padding: '8px 14px', borderBottom: '1px solid rgba(239,68,68,0.25)', background: 'rgba(239,68,68,0.08)', color: '#fecaca', fontSize: 12 },
  errX: { marginLeft: 'auto', background: 'none', border: 'none', color: '#fecaca', cursor: 'pointer', fontSize: 14 },
  body: { display: 'flex', flex: 1, minHeight: 0 },
  sidebar: { width: 380, borderRight: '1px solid ' + L, background: 'rgba(15,23,42,0.6)', display: 'flex', flexDirection: 'column' },
  tabRow: { display: 'flex', padding: '10px 8px 6px', gap: 4, borderBottom: '1px solid ' + L },
  tab: { flex: 1, height: 32, borderRadius: 8, border: '1px solid transparent', background: 'transparent', color: '#64748b', fontSize: 11, fontWeight: 800, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 },
  tabOn: { background: 'rgba(59,130,246,0.14)', borderColor: 'rgba(59,130,246,0.25)', color: '#93c5fd' },
  tabIcon: { fontSize: 13 },
  panelArea: { flex: 1, overflow: 'hidden' },
  pi: { height: '100%', overflowY: 'auto', padding: '12px 12px 20px' },
  ph: { marginBottom: 12 },
  phT: { fontSize: 14, fontWeight: 950 },
  phS: { fontSize: 11, color: '#64748b', marginTop: 2 },
  sep: { height: 1, background: L, margin: '12px 0' },
  sh: { fontSize: 11, fontWeight: 900, color: '#94a3b8', marginBottom: 6, marginTop: 4, textTransform: 'uppercase', letterSpacing: 0.5 },
  fr: { display: 'flex', justifyContent: 'space-between', padding: '5px 0', fontSize: 12 },
  frL: { color: '#64748b', fontWeight: 800 },
  frV: { fontWeight: 800, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  hint: { fontSize: 11, color: '#64748b', lineHeight: 1.5 },
  tog: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0' },
  togL: { fontSize: 12, fontWeight: 800 },
  togBtn: { height: 30, padding: '0 12px', borderRadius: 999, border: '1px solid ' + L, background: 'rgba(2,6,23,0.2)', color: '#64748b', fontWeight: 900, fontSize: 11, cursor: 'pointer' },
  togOn: { background: 'rgba(59,130,246,0.14)', borderColor: 'rgba(59,130,246,0.25)', color: '#93c5fd' },
  aBtn: { width: '100%', height: 36, borderRadius: 10, border: '1px solid rgba(59,130,246,0.30)', background: 'rgba(59,130,246,0.16)', color: '#93c5fd', fontSize: 12, fontWeight: 900, cursor: 'pointer', marginTop: 6 },
  gBtn: { width: '100%', height: 36, borderRadius: 10, border: '1px solid ' + L, background: 'rgba(2,6,23,0.2)', color: '#94a3b8', fontSize: 12, fontWeight: 900, cursor: 'pointer', marginTop: 6 },
  dis: { opacity: 0.45, cursor: 'default' },
  pills: { display: 'flex', flexWrap: 'wrap', gap: 6 },
  pBtn: { height: 30, padding: '0 10px', borderRadius: 999, border: '1px solid ' + L, background: 'rgba(2,6,23,0.2)', color: '#94a3b8', fontSize: 11, fontWeight: 800, cursor: 'pointer' },
  pOn: { background: 'rgba(59,130,246,0.14)', borderColor: 'rgba(59,130,246,0.25)', color: '#93c5fd' },
  slRow: { marginBottom: 10 },
  slHead: { display: 'flex', justifyContent: 'space-between' },
  slL: { fontSize: 11, fontWeight: 800, color: '#cbd5e1' },
  slV: { fontSize: 11, fontWeight: 800, color: '#64748b', fontVariantNumeric: 'tabular-nums' },
  slTrack: { position: 'relative', height: 16, marginTop: 4 },
  slBg: { position: 'absolute', left: 0, right: 0, top: '50%', height: 3, transform: 'translateY(-50%)', background: 'rgba(148,163,184,0.15)', borderRadius: 10 },
  slFill: { position: 'absolute', top: '50%', height: 3, transform: 'translateY(-50%)', background: B, borderRadius: 10, opacity: 0.85 },
  slMid: { position: 'absolute', top: '50%', width: 1.5, height: 8, transform: 'translateY(-50%)', background: 'rgba(148,163,184,0.4)', borderRadius: 10 },
  slIn: { position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', opacity: 0, cursor: 'pointer' },
  lCard: { width: '100%', display: 'flex', alignItems: 'center', gap: 10, padding: '10px 12px', borderRadius: 12, border: '1px solid ' + L, background: 'rgba(2,6,23,0.2)', cursor: 'pointer', textAlign: 'left', marginBottom: 6, color: '#e2e8f0' },
  lOn: { borderColor: 'rgba(59,130,246,0.30)', background: 'rgba(59,130,246,0.08)' },
  lIcon: { fontSize: 18, width: 28, textAlign: 'center' },
  lName: { fontSize: 12, fontWeight: 900 },
  lDesc: { fontSize: 11, color: '#64748b', fontWeight: 700 },
  badge: { padding: '3px 8px', borderRadius: 999, border: '1px solid ' + L, fontSize: 10, fontWeight: 900 },
  bOn: { borderColor: 'rgba(34,197,94,0.30)', background: 'rgba(34,197,94,0.12)', color: '#86efac' },
  bOff: { color: '#64748b' },
  bWarn: { borderColor: 'rgba(245,158,11,0.30)', background: 'rgba(245,158,11,0.12)', color: '#fbbf24' },
  aiBox: { borderRadius: 14, border: '1px solid ' + L, padding: 12, marginBottom: 8 },
  aiTop: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 },
  aiT: { fontSize: 12, fontWeight: 950 },
  aiD: { fontSize: 11, color: '#64748b', marginTop: 2, lineHeight: 1.4 },
  aiH: { fontSize: 11, color: '#fbbf24', marginTop: 8, padding: '6px 8px', borderRadius: 8, background: 'rgba(251,191,36,0.08)', lineHeight: 1.4, wordBreak: 'break-word' },
  sBox: { display: 'flex', flexDirection: 'column', gap: 6, padding: 10, borderRadius: 10, border: '1px solid ' + L, background: 'rgba(2,6,23,0.3)', marginBottom: 10 },
  mGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 },
  mc: { padding: 10, borderRadius: 10, border: '1px solid ' + L, background: 'rgba(2,6,23,0.2)' },
  mcL: { fontSize: 10, color: '#64748b', fontWeight: 800, textTransform: 'uppercase' },
  mcV: { fontSize: 14, fontWeight: 950, marginTop: 4 },
  sugCard: { padding: '8px 10px', borderRadius: 10, border: '1px solid rgba(59,130,246,0.20)', background: 'rgba(59,130,246,0.06)', marginBottom: 6 },
  sugT: { fontSize: 11, fontWeight: 900, color: '#93c5fd', textTransform: 'uppercase' },
  sugX: { fontSize: 11, color: '#94a3b8', marginTop: 2 },
  recCard: { borderRadius: 12, border: '1px solid ' + L, padding: 12 },
  recW: { fontSize: 11, color: '#64748b', margin: '6px 0' },
  canvas: { flex: 1, minWidth: 0, background: 'radial-gradient(ellipse at 30% 20%, rgba(59,130,246,0.12) 0%, #070b14 60%)', position: 'relative', overflow: 'hidden' },
  busyOvl: { position: 'absolute', inset: 0, background: 'rgba(7,11,20,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, zIndex: 10, color: '#e2e8f0', fontSize: 13 },
  spinner: { width: 14, height: 14, borderRadius: 999, border: '2px solid rgba(255,255,255,0.15)', borderTopColor: 'rgba(255,255,255,0.7)', animation: 'spin .8s linear infinite' },
  empty: { position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 6 },
  emI: { fontSize: 42, color: B, marginBottom: 8 },
  emT: { fontSize: 22, fontWeight: 950 },
  emS: { fontSize: 13, color: '#64748b' },
  cInner: { position: 'absolute', inset: 0 },
  cImg: { position: 'absolute', inset: 0, width: '100%', height: '100%' },
  spLine: { position: 'absolute', top: 0, bottom: 0, width: 2, background: 'rgba(255,255,255,0.4)', transform: 'translateX(-50%)', pointerEvents: 'none' },
  spKnob: { position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', width: 30, height: 30, borderRadius: 999, background: 'rgba(255,255,255,0.9)', color: '#0f172a', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 950, boxShadow: '0 8px 20px rgba(0,0,0,0.35)' },
  spIn: { position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0, cursor: 'ew-resize' },
  tagL: { position: 'absolute', top: 12, left: 12, padding: '4px 10px', borderRadius: 999, border: '1px solid ' + L, background: 'rgba(7,11,20,0.5)', fontSize: 11, fontWeight: 900, color: '#cbd5e1' },
  tagR: { position: 'absolute', top: 12, right: 12, padding: '4px 10px', borderRadius: 999, border: '1px solid ' + L, background: 'rgba(7,11,20,0.5)', fontSize: 11, fontWeight: 900, color: '#cbd5e1' },
  tagB: { position: 'absolute', bottom: 14, left: '50%', transform: 'translateX(-50%)', padding: '5px 12px', borderRadius: 999, border: '1px solid rgba(59,130,246,0.30)', background: 'rgba(59,130,246,0.14)', color: '#93c5fd', fontWeight: 900, fontSize: 11 },
  helpTag: { position: 'absolute', bottom: 14, right: 14, padding: '4px 10px', borderRadius: 999, border: '1px solid ' + L, background: 'rgba(7,11,20,0.5)', color: '#64748b', fontWeight: 800, fontSize: 11 },
  hKey: { padding: '1px 6px', borderRadius: 6, border: '1px solid ' + L, fontWeight: 950, color: '#cbd5e1' },
  toast: { position: 'fixed', bottom: 14, left: 14, padding: '10px 14px', borderRadius: 12, border: '1px solid ' + L, background: 'rgba(15,23,42,0.9)', backdropFilter: 'blur(8px)', display: 'flex', gap: 8, alignItems: 'center', fontSize: 12, zIndex: 999, boxShadow: '0 12px 30px rgba(0,0,0,0.4)' },
}

if (typeof document !== 'undefined' && !document.getElementById('__aurora_css')) {
  const el = document.createElement('style'); el.id = '__aurora_css'
  el.textContent = '@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}*{box-sizing:border-box;margin:0}body{overflow:hidden}::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:rgba(148,163,184,.15);border-radius:3px}::-webkit-scrollbar-thumb:hover{background:rgba(148,163,184,.3)}'
  document.head.appendChild(el)
}
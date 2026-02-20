import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'

/**
 * Enterprise Revamp — App.jsx
 * - Live slider adjustments (debounced, latest-wins)
 * - Tabs for control panels
 * - Toasts instead of alerts
 * - Better status + pipeline UX
 * - Spacebar hold to compare original
 */

// ----------------------------
// Constants
// ----------------------------
const ENHANCE_MODES = [
  { id: 'pro', label: 'Pro Detail', desc: 'Sharpness + micro-contrast enhancement' },
  { id: 'color', label: 'Color Restore', desc: 'Recover fading / balance color' },
  { id: 'smooth', label: 'Denoise', desc: 'Reduce noise / compression artifacts' },
  { id: 'light', label: 'Lighting Fix', desc: 'Shadow/highlight recovery and lift' },
  { id: 'portrait', label: 'Portrait', desc: 'Skin-safe tuning, facial detail preservation' }
]

const XRAY_MODES = [
  { id: 'none', label: 'Off' },
  { id: 'structure', label: 'Structure' },
  { id: 'depth', label: 'Depth' },
  { id: 'frequency', label: 'Frequency' },
  { id: 'thermal', label: 'Thermal' },
  { id: 'bones', label: 'Bones' },
  { id: 'reveal', label: 'Reveal' },
  { id: 'bright', label: 'Bright' },
  { id: 'occlusion', label: 'Occlusion' }
]

const DEFAULTS = {
  exposure: 0,
  contrast: 0,
  highlights: 0,
  shadows: 0,
  whites: 0,
  blacks: 0,
  temperature: 0,
  tint: 0,
  vibrance: 0,
  saturation: 0,
  clarity: 0,
  dehaze: 0,
  sharpness: 0,
  grain: 0,
  vignette: 0,
  xray: 'none',
  xray_blend: 100
}

const TABS = [
  { id: 'intake', label: 'Intake' },
  { id: 'adjust', label: 'Adjust' },
  { id: 'enhance', label: 'Enhance' },
  { id: 'analyze', label: 'Analyze' },
  { id: 'export', label: 'Export' }
]

// ----------------------------
// App
// ----------------------------
export default function App() {
  // Core image state
  const [originalPath, setOriginalPath] = useState(null)
  const [originalPreview, setOriginalPreview] = useState(null)

  const [resultPreview, setResultPreview] = useState(null)
  const [tempPath, setTempPath] = useState(null)
  const [resultLabel, setResultLabel] = useState('')

  // Adjustments / layers
  const [adj, setAdj] = useState({ ...DEFAULTS })
  const [activeLayers, setActiveLayers] = useState(['pro'])
  const [liveAdjust, setLiveAdjust] = useState(true)

  // Analysis output
  const [diagnosis, setDiagnosis] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [rec, setRec] = useState(null)

  // UI
  const [tab, setTab] = useState('intake')
  const [busy, setBusy] = useState(false)
  const [busyMsg, setBusyMsg] = useState('')
  const [saving, setSaving] = useState(false)
  const [uiError, setUiError] = useState('')
  const [toast, setToast] = useState(null)

  // Compare slider + hold-to-compare
  const [holdOriginal, setHoldOriginal] = useState(false)
  const [splitPos, setSplitPos] = useState(55)
  const [zoom, setZoom] = useState(1) // simple zoom control (optional)
  const [fitMode, setFitMode] = useState('contain') // contain | cover

  // Derived
  const fileName = useMemo(
    () => (originalPath ? originalPath.split(/[/\\]/).pop() : null),
    [originalPath]
  )
  const dirty = useMemo(
    () => Object.keys(DEFAULTS).some((k) => adj[k] !== DEFAULTS[k]),
    [adj]
  )
  const hasResult = !!resultPreview && !!tempPath

  // ----------------------------
  // Toast helper
  // ----------------------------
  const pushToast = useCallback((type, title, message) => {
    setToast({ type, title, message, ts: Date.now() })
    window.clearTimeout(pushToast._t)
    pushToast._t = window.setTimeout(() => setToast(null), 2800)
  }, [])
  pushToast._t = pushToast._t || 0

  // ----------------------------
  // Keyboard shortcuts
  // ----------------------------
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.code === 'Space') {
        e.preventDefault()
        setHoldOriginal(true)
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'o') {
        e.preventDefault()
        handleOpen()
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault()
        if (tempPath) handleSave()
      }
      if (e.key === 'Escape') {
        setUiError('')
        setToast(null)
      }
    }
    const onKeyUp = (e) => {
      if (e.code === 'Space') {
        e.preventDefault()
        setHoldOriginal(false)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tempPath])

  // ----------------------------
  // API Guards (enterprise: fail loudly but cleanly)
  // ----------------------------
  useEffect(() => {
    if (!window.api) {
      setUiError(
        'Backend bridge unavailable (window.api missing). Check preload + contextBridge. (Tip: sandbox: false on Wayland)'
      )
    }
  }, [])

  // ----------------------------
  // Open image
  // ----------------------------
  const handleOpen = useCallback(async () => {
    try {
      setUiError('')
      if (!window.api?.selectImage) {
        setUiError('Backend API not available: window.api.selectImage missing.')
        pushToast('error', 'Backend', 'Preload bridge missing: window.api.selectImage')
        return
      }

      const f = await window.api.selectImage()
      if (!f) return
      if (f.error) {
        setUiError(f.error)
        pushToast('error', 'Load failed', f.error)
        return
      }

      // Reset state
      setOriginalPath(f.path)
      setOriginalPreview(f.preview)

      setResultPreview(null)
      setTempPath(null)
      setResultLabel('')

      setDiagnosis(null)
      setMetrics(null)
      setRec(null)

      setAdj({ ...DEFAULTS })
      setActiveLayers(['pro'])
      setSplitPos(55)
      setZoom(1)
      setFitMode('contain')

      setTab('adjust')
      pushToast('ok', 'Asset loaded', fileNameFromPath(f.path))
    } catch (err) {
      const msg = err?.message || 'Failed to open file'
      setUiError(msg)
      pushToast('error', 'Open failed', msg)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pushToast])

  // ----------------------------
  // Autopilot analysis
  // ----------------------------
  const handleAnalyze = useCallback(async () => {
    if (!originalPath) return
    setUiError('')
    setBusy(true)
    setBusyMsg('Analyzing')
    try {
      const r = await window.api.runAutopilot(originalPath)
      if (r?.error) {
        setUiError(r.error)
        pushToast('error', 'Analysis failed', r.error)
        return
      }
      if (r?.recommendations) {
        setDiagnosis(r.analysis || null)
        setMetrics(r.metrics || null)
        setRec(r.recommendations || null)

        if (r.recommendations.best_mode) setActiveLayers([r.recommendations.best_mode])

        // Seed adjustments (safe defaults)
        setAdj((p) => ({
          ...p,
          exposure: r.recommendations.exposure || 0,
          saturation: Math.round(((r.recommendations.saturation || 1) - 1) * 100) || 0
        }))
        setTab('adjust')
        pushToast('ok', 'Analysis ready', `Recommended: ${String(r.recommendations.best_mode || '').toUpperCase()}`)
      }
    } finally {
      setBusy(false)
      setBusyMsg('')
    }
  }, [originalPath, pushToast])

  // ----------------------------
  // Enhance pipeline
  // ----------------------------
  const toggleLayer = useCallback((id) => {
    setActiveLayers((p) => {
      if (p.includes(id)) {
        const n = p.filter((m) => m !== id)
        return n.length ? n : p
      }
      return [...p, id]
    })
  }, [])

  const handleEnhance = useCallback(async () => {
    if (!originalPath) return
    setUiError('')
    setBusy(true)
    setBusyMsg('Enhancing')
    try {
      const r = await window.api.enhanceImage(originalPath, activeLayers)
      if (r?.error) {
        setUiError(r.error)
        pushToast('error', 'Enhance failed', r.error)
        return
      }
      if (r?.temp_path) {
        setTempPath(r.temp_path)
        setResultPreview(r.preview)
        setResultLabel((r.applied_modes || activeLayers).join(' + '))
        setTab('export')
        pushToast('ok', 'Enhance complete', 'Result ready to export')
      }
    } finally {
      setBusy(false)
      setBusyMsg('')
    }
  }, [activeLayers, originalPath, pushToast])

  // ----------------------------
  // Save / Export
  // ----------------------------
  const handleSave = useCallback(async () => {
    if (!tempPath) return
    setSaving(true)
    try {
      const r = await window.api.saveImage(tempPath)
      if (r?.saved_path) {
        pushToast('ok', 'Saved', fileNameFromPath(r.saved_path))
        setSaving('done')
        setTimeout(() => setSaving(false), 1600)
      } else {
        if (r?.error && r.error !== 'Cancelled') {
          setUiError(r.error)
          pushToast('error', 'Save failed', r.error)
        }
        setSaving(false)
      }
    } catch (e) {
      const msg = e?.message || 'Save failed'
      setUiError(msg)
      pushToast('error', 'Save failed', msg)
      setSaving(false)
    }
  }, [tempPath, pushToast])

  // ----------------------------
  // Live Adjust engine call (debounced, latest-wins)
  // ----------------------------
  const adjustTimerRef = useRef(null)
  const adjustSeqRef = useRef(0)
  const inFlightRef = useRef(false)

  useEffect(() => {
    if (!liveAdjust) return
    if (!originalPath) return
    if (!dirty) return
    if (!window.api?.adjustImage) return

    if (adjustTimerRef.current) window.clearTimeout(adjustTimerRef.current)

    adjustTimerRef.current = window.setTimeout(async () => {
      if (inFlightRef.current) return

      const seq = ++adjustSeqRef.current
      inFlightRef.current = true

      setBusy(true)
      setBusyMsg('Adjusting')

      try {
        const r = await window.api.adjustImage(originalPath, adj)

        // ignore outdated responses
        if (seq !== adjustSeqRef.current) return

        if (r?.error) {
          setUiError(r.error)
          return
        }
        if (r?.temp_path) {
          setTempPath(r.temp_path)
          setResultPreview(r.preview)
          setResultLabel('Live Adjust')
        }
      } catch (e) {
        if (seq === adjustSeqRef.current) setUiError(e?.message || 'Adjust failed')
      } finally {
        if (seq === adjustSeqRef.current) {
          setBusy(false)
          setBusyMsg('')
        }
        inFlightRef.current = false
      }
    }, 140)

    return () => {
      if (adjustTimerRef.current) window.clearTimeout(adjustTimerRef.current)
    }
  }, [adj, dirty, originalPath, liveAdjust])

  // Manual apply if liveAdjust disabled
  const handleApply = useCallback(async () => {
    if (!originalPath || !dirty) return
    setUiError('')
    setBusy(true)
    setBusyMsg('Adjusting')
    try {
      const r = await window.api.adjustImage(originalPath, adj)
      if (r?.error) {
        setUiError(r.error)
        pushToast('error', 'Adjust failed', r.error)
        return
      }
      if (r?.temp_path) {
        setTempPath(r.temp_path)
        setResultPreview(r.preview)
        setResultLabel('Adjusted')
        pushToast('ok', 'Applied', 'Adjustments applied')
      }
    } finally {
      setBusy(false)
      setBusyMsg('')
    }
  }, [adj, dirty, originalPath, pushToast])

  const handleReset = useCallback(() => {
    setAdj({ ...DEFAULTS })
    pushToast('ok', 'Reset', 'Adjustments reset to defaults')
  }, [pushToast])

  // ----------------------------
  // UI helpers
  // ----------------------------
  const setA = useCallback((k, v) => setAdj((p) => ({ ...p, [k]: v })), [])
  const pipeline = useMemo(() => activeLayers.join(' → '), [activeLayers])

  const sessionStatus = useMemo(() => {
    if (!originalPath) return { label: 'Idle', tone: 'dim' }
    if (busy) return { label: busyMsg || 'Working', tone: 'amber' }
    if (hasResult) return { label: 'Result Ready', tone: 'green' }
    if (dirty) return { label: liveAdjust ? 'Live Adjust' : 'Draft Changes', tone: 'cyan' }
    return { label: 'Loaded', tone: 'cyan' }
  }, [busy, busyMsg, dirty, hasResult, liveAdjust, originalPath])

  // ----------------------------
  // Render
  // ----------------------------
  return (
    <div style={S.app}>
      <TopBar
        fileName={fileName}
        status={sessionStatus}
        onOpen={handleOpen}
        onAnalyze={handleAnalyze}
        onEnhance={handleEnhance}
        onSave={handleSave}
        canAnalyze={!!originalPath && !busy}
        canEnhance={!!originalPath && !busy}
        canSave={!!tempPath && !saving}
        busy={busy}
        saving={saving}
        zoom={zoom}
        setZoom={setZoom}
        fitMode={fitMode}
        setFitMode={setFitMode}
      />

      {uiError ? (
        <div style={S.bannerError}>
          <div style={{ fontWeight: 900, letterSpacing: 0.4 }}>ERROR</div>
          <div style={{ opacity: 0.9 }}>{uiError}</div>
          <button onClick={() => setUiError('')} style={S.bannerBtn}>
            Dismiss
          </button>
        </div>
      ) : null}

      <div style={S.body}>
        <aside style={S.sidebar}>
          <div style={S.brand}>
            <div style={S.brandIcon} />
            <div style={{ minWidth: 0 }}>
              <div style={S.brandTitle}>AURORA OPS</div>
              <div style={S.brandSub}>Enterprise Imaging Console</div>
            </div>
          </div>

          <TabRow tab={tab} setTab={setTab} />

          <div style={S.panel}>
            {tab === 'intake' && (
              <Panel title="Intake" subtitle="Load an asset and start a workflow">
                <PrimaryButton onClick={handleOpen}>
                  {originalPath ? 'Change Image' : 'Open Image'}
                </PrimaryButton>

                <FieldRow label="Asset" value={fileName || '—'} mono />
                <FieldRow
                  label="Session"
                  value={originalPath ? 'Secure session active' : 'Awaiting asset'}
                />

                <Divider />

                <Hint>
                  Shortcuts: <K>Ctrl</K>+<K>O</K> open · <K>Space</K> hold compare · <K>Ctrl</K>+<K>S</K> save
                </Hint>
              </Panel>
            )}

            {tab === 'adjust' && (
              <Panel title="Adjust" subtitle="Real-time tonal and color corrections">
                <ToggleRow
                  label="Live Adjust"
                  value={liveAdjust}
                  onToggle={() => setLiveAdjust((v) => !v)}
                  hint="Updates preview while you drag sliders"
                />

                {!liveAdjust && dirty && (
                  <PrimaryButton onClick={handleApply} disabled={busy}>
                    {busy && busyMsg === 'Adjusting' ? 'Applying…' : 'Apply Changes'}
                  </PrimaryButton>
                )}

                <SecondaryButton onClick={handleReset} disabled={!dirty}>
                  Reset to Defaults
                </SecondaryButton>

                <Divider />

                <SectionTitle>Tone</SectionTitle>
                <Slider label="Exposure" value={adj.exposure} min={-3} max={3} step={0.05} onChange={(v) => setA('exposure', v)} />
                <Slider label="Contrast" value={adj.contrast} min={-100} max={100} step={1} onChange={(v) => setA('contrast', v)} />
                <Slider label="Highlights" value={adj.highlights} min={-100} max={100} step={1} onChange={(v) => setA('highlights', v)} />
                <Slider label="Shadows" value={adj.shadows} min={-100} max={100} step={1} onChange={(v) => setA('shadows', v)} />
                <Slider label="Whites" value={adj.whites} min={-100} max={100} step={1} onChange={(v) => setA('whites', v)} />
                <Slider label="Blacks" value={adj.blacks} min={-100} max={100} step={1} onChange={(v) => setA('blacks', v)} />

                <Divider />

                <SectionTitle>Color</SectionTitle>
                <Slider label="Temperature" value={adj.temperature} min={-100} max={100} step={1} onChange={(v) => setA('temperature', v)} />
                <Slider label="Tint" value={adj.tint} min={-100} max={100} step={1} onChange={(v) => setA('tint', v)} />
                <Slider label="Vibrance" value={adj.vibrance} min={-100} max={100} step={1} onChange={(v) => setA('vibrance', v)} />
                <Slider label="Saturation" value={adj.saturation} min={-100} max={100} step={1} onChange={(v) => setA('saturation', v)} />

                <Divider />

                <SectionTitle>Detail & Effects</SectionTitle>
                <Slider label="Clarity" value={adj.clarity} min={-100} max={100} step={1} onChange={(v) => setA('clarity', v)} />
                <Slider label="Dehaze" value={adj.dehaze} min={0} max={100} step={1} onChange={(v) => setA('dehaze', v)} />
                <Slider label="Sharpness" value={adj.sharpness} min={0} max={100} step={1} onChange={(v) => setA('sharpness', v)} />
                <Slider label="Grain" value={adj.grain} min={0} max={100} step={1} onChange={(v) => setA('grain', v)} />
                <Slider label="Vignette" value={adj.vignette} min={-100} max={100} step={1} onChange={(v) => setA('vignette', v)} />

                <Divider />

                <SectionTitle>X-Ray</SectionTitle>
                <Pills
                  items={XRAY_MODES}
                  value={adj.xray}
                  onChange={(id) => setA('xray', id)}
                />
                {adj.xray !== 'none' && (
                  <Slider label="X-Ray Blend" value={adj.xray_blend} min={0} max={100} step={1} onChange={(v) => setA('xray_blend', v)} />
                )}
              </Panel>
            )}

            {tab === 'enhance' && (
              <Panel title="Enhance" subtitle="Stackable enhancement layers">
                <FieldRow label="Pipeline" value={pipeline || '—'} mono />

                <Divider />

                <div style={S.layerGrid}>
                  {ENHANCE_MODES.map((m) => {
                    const on = activeLayers.includes(m.id)
                    const idx = on ? activeLayers.indexOf(m.id) + 1 : null
                    return (
                      <button
                        key={m.id}
                        onClick={() => toggleLayer(m.id)}
                        style={{
                          ...S.layerCard,
                          borderColor: on ? C.line2 : C.line,
                          background: on
                            ? 'linear-gradient(180deg, rgba(59,130,246,0.14), rgba(2,6,23,0.35))'
                            : 'rgba(2,6,23,0.25)'
                        }}
                      >
                        <div style={S.layerTop}>
                          <div style={S.layerIdx}>{idx || '—'}</div>
                          <div style={{ minWidth: 0 }}>
                            <div style={S.layerName}>{m.label}</div>
                            <div style={S.layerDesc}>{m.desc}</div>
                          </div>
                          <div style={{ ...S.badge, ...(on ? S.badgeOn : S.badgeOff) }}>
                            {on ? 'ON' : 'OFF'}
                          </div>
                        </div>
                      </button>
                    )
                  })}
                </div>

                <Divider />

                <PrimaryButton onClick={handleEnhance} disabled={!originalPath || busy}>
                  {busy && busyMsg === 'Enhancing' ? 'Processing…' : 'Run Enhance'}
                </PrimaryButton>

                <Hint>
                  Tip: Run <b>Analyze</b> first to auto-pick a best layer.
                </Hint>
              </Panel>
            )}

            {tab === 'analyze' && (
              <Panel title="Analyze" subtitle="Quality metrics + recommended pipeline">
                <PrimaryButton onClick={handleAnalyze} disabled={!originalPath || busy}>
                  {busy && busyMsg === 'Analyzing' ? 'Analyzing…' : 'Run Analysis'}
                </PrimaryButton>

                <Divider />

                <FieldRow label="Diagnosis" value={diagnosis || '—'} />

                <Divider />

                <div style={S.metricGrid}>
                  <Metric label="Noise" value={metrics?.noise ?? '—'} tone={toneNoise(metrics?.noise)} />
                  <Metric
                    label="Sharpness"
                    value={
                      metrics?.sharpness == null
                        ? '—'
                        : metrics.sharpness > 500
                          ? 'HI'
                          : metrics.sharpness > 100
                            ? 'MED'
                            : 'LO'
                    }
                    tone={toneSharp(metrics?.sharpness)}
                  />
                  <Metric label="Range" value={metrics?.dynamic_range ?? '—'} tone={toneRange(metrics?.dynamic_range)} />
                  <Metric label="Skin" value={metrics?.skin_pct == null ? '—' : `${metrics.skin_pct}%`} tone={toneSkin(metrics?.skin_pct)} />
                </div>

                <Divider />

                <div style={S.recCard}>
                  <div style={S.recHead}>
                    <div style={S.recTitle}>Recommendation</div>
                    <div style={S.badgeStrong}>
                      {rec?.best_mode ? String(rec.best_mode).toUpperCase() : '—'}
                    </div>
                  </div>
                  <div style={S.recBody}>{rec?.mode_reason || 'Run analysis to generate a recommendation.'}</div>
                  {!!rec?.best_mode && (
                    <SecondaryButton
                      onClick={() => {
                        setActiveLayers([rec.best_mode])
                        setTab('enhance')
                        pushToast('ok', 'Pipeline updated', `Set to ${String(rec.best_mode).toUpperCase()}`)
                      }}
                    >
                      Apply Recommendation
                    </SecondaryButton>
                  )}
                </div>
              </Panel>
            )}

            {tab === 'export' && (
              <Panel title="Export" subtitle="Save to disk and finalize output">
                <FieldRow label="Result" value={resultLabel || (hasResult ? 'Ready' : '—')} mono />

                <Divider />

                <PrimaryButton onClick={handleSave} disabled={!tempPath || saving === true}>
                  {saving === 'done' ? 'Saved' : saving ? 'Saving…' : 'Save As…'}
                </PrimaryButton>

                <SecondaryButton
                  onClick={() => {
                    setTab('enhance')
                    pushToast('ok', 'Back', 'Adjust pipeline or enhance layers')
                  }}
                  disabled={!originalPath}
                >
                  Back to Enhance
                </SecondaryButton>

                <Divider />

                <Hint>
                  If the preview looks good, export. If not, switch to <b>Adjust</b> for fine tuning.
                </Hint>
              </Panel>
            )}
          </div>
        </aside>

        <main style={S.canvas}>
          {busy ? <BusyOverlay label={busyMsg} /> : null}

          {!originalPath ? (
            <EmptyState onOpen={handleOpen} />
          ) : (
            <CompareCanvas
              originalPreview={originalPreview}
              resultPreview={resultPreview}
              holdOriginal={holdOriginal}
              splitPos={splitPos}
              setSplitPos={setSplitPos}
              hasResult={hasResult}
              label={holdOriginal ? 'ORIGINAL' : hasResult ? 'RESULT' : 'ORIGINAL'}
              resultLabel={resultLabel}
              zoom={zoom}
              fitMode={fitMode}
            />
          )}
        </main>
      </div>

      {toast ? <Toast toast={toast} /> : null}
    </div>
  )
}

// ----------------------------
// Components
// ----------------------------
function TopBar({
  fileName,
  status,
  onOpen,
  onAnalyze,
  onEnhance,
  onSave,
  canAnalyze,
  canEnhance,
  canSave,
  busy,
  saving,
  zoom,
  setZoom,
  fitMode,
  setFitMode
}) {
  return (
    <div style={S.topBar}>
      <div style={S.topLeft}>
        <div style={S.logoDot} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
          <div style={S.appTitle}>Enterprise Image Ops Console</div>
          <div style={S.appSub}>
            {fileName ? (
              <>
                <span style={{ opacity: 0.75 }}>Asset:</span> <span style={S.mono}>{fileName}</span>
              </>
            ) : (
              <span style={{ opacity: 0.75 }}>No asset loaded</span>
            )}
          </div>
        </div>
      </div>

      <div style={S.topRight}>
        <StatusPill label={status.label} tone={status.tone} />

        <div style={S.topActions}>
          <SmallButton onClick={onOpen} title="Ctrl+O">
            Open
          </SmallButton>
          <SmallButton onClick={onAnalyze} disabled={!canAnalyze} title="Analyze">
            Analyze
          </SmallButton>
          <SmallButton onClick={onEnhance} disabled={!canEnhance} title="Enhance">
            Enhance
          </SmallButton>
          <SmallButton onClick={onSave} disabled={!canSave} title="Ctrl+S">
            {saving === 'done' ? 'Saved' : saving ? 'Saving…' : 'Save'}
          </SmallButton>
        </div>

        <div style={S.viewGroup}>
          <SmallChip onClick={() => setFitMode((m) => (m === 'contain' ? 'cover' : 'contain'))}>
            Fit: {fitMode === 'contain' ? 'Contain' : 'Cover'}
          </SmallChip>
          <SmallChip onClick={() => setZoom((z) => clamp(round2(z + 0.1), 1, 2.2))}>Zoom +</SmallChip>
          <SmallChip onClick={() => setZoom((z) => clamp(round2(z - 0.1), 1, 2.2))}>Zoom −</SmallChip>
        </div>

        <div style={{ opacity: 0.6, fontSize: 12, paddingLeft: 8 }}>
          {busy ? 'Processing…' : ''}
        </div>
      </div>
    </div>
  )
}

function TabRow({ tab, setTab }) {
  return (
    <div style={S.tabRow}>
      {TABS.map((t) => {
        const active = tab === t.id
        return (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              ...S.tab,
              ...(active ? S.tabActive : S.tabIdle)
            }}
          >
            {t.label}
          </button>
        )
      })}
    </div>
  )
}

function Panel({ title, subtitle, children }) {
  return (
    <div style={S.panelInner}>
      <div style={S.panelHead}>
        <div>
          <div style={S.panelTitle}>{title}</div>
          <div style={S.panelSub}>{subtitle}</div>
        </div>
      </div>
      <div style={S.panelBody}>{children}</div>
    </div>
  )
}

function Divider() {
  return <div style={S.divider} />
}

function SectionTitle({ children }) {
  return <div style={S.sectionTitle}>{children}</div>
}

function FieldRow({ label, value, mono }) {
  return (
    <div style={S.fieldRow}>
      <div style={S.fieldLabel}>{label}</div>
      <div style={{ ...S.fieldValue, ...(mono ? S.mono : null) }}>{value}</div>
    </div>
  )
}

function Hint({ children }) {
  return <div style={S.hint}>{children}</div>
}

function K({ children }) {
  return <span style={S.kbd}>{children}</span>
}

function ToggleRow({ label, value, onToggle, hint }) {
  return (
    <div style={S.toggleRow}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <div style={S.toggleLabel}>{label}</div>
        <div style={S.toggleHint}>{hint}</div>
      </div>
      <button onClick={onToggle} style={{ ...S.toggleBtn, ...(value ? S.toggleOn : S.toggleOff) }}>
        {value ? 'ON' : 'OFF'}
      </button>
    </div>
  )
}

function PrimaryButton({ children, onClick, disabled }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{ ...S.btn, ...S.btnPrimary, ...(disabled ? S.btnDisabled : null) }}>
      {children}
    </button>
  )
}

function SecondaryButton({ children, onClick, disabled }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{ ...S.btn, ...S.btnSecondary, ...(disabled ? S.btnDisabled : null) }}>
      {children}
    </button>
  )
}

function SmallButton({ children, onClick, disabled, title }) {
  return (
    <button
      title={title}
      onClick={onClick}
      disabled={disabled}
      style={{ ...S.smallBtn, ...(disabled ? S.smallBtnDisabled : null) }}
    >
      {children}
    </button>
  )
}

function SmallChip({ children, onClick }) {
  return (
    <button onClick={onClick} style={S.smallChip}>
      {children}
    </button>
  )
}

function StatusPill({ label, tone }) {
  const palette = {
    dim: { bg: 'rgba(148,163,184,0.12)', bd: 'rgba(148,163,184,0.25)', fg: '#94a3b8' },
    cyan: { bg: 'rgba(59,130,246,0.14)', bd: 'rgba(59,130,246,0.30)', fg: '#93c5fd' },
    amber: { bg: 'rgba(245,158,11,0.12)', bd: 'rgba(245,158,11,0.28)', fg: '#fcd34d' },
    green: { bg: 'rgba(34,197,94,0.12)', bd: 'rgba(34,197,94,0.28)', fg: '#86efac' },
    red: { bg: 'rgba(239,68,68,0.12)', bd: 'rgba(239,68,68,0.28)', fg: '#fca5a5' }
  }[tone] || {
    bg: 'rgba(148,163,184,0.12)',
    bd: 'rgba(148,163,184,0.25)',
    fg: '#94a3b8'
  }

  return (
    <div style={{ ...S.pill, background: palette.bg, borderColor: palette.bd, color: palette.fg }}>
      {label}
    </div>
  )
}

function Pills({ items, value, onChange }) {
  return (
    <div style={S.pills}>
      {items.map((i) => {
        const active = value === i.id
        return (
          <button
            key={i.id}
            onClick={() => onChange(i.id)}
            style={{
              ...S.pillBtn,
              ...(active ? S.pillBtnOn : S.pillBtnOff)
            }}
          >
            {i.label}
          </button>
        )
      })}
    </div>
  )
}

function Slider({ label, value, min, max, step, onChange }) {
  const isCenter = min < 0
  const pct = ((value - min) / (max - min)) * 100
  const cPct = isCenter ? ((0 - min) / (max - min)) * 100 : 0
  const left = isCenter ? Math.min(pct, cPct) : 0
  const width = isCenter ? Math.abs(pct - cPct) : pct
  const display = step < 1 && step > 0 ? Number(value).toFixed(2) : Math.round(value)

  return (
    <div style={S.sliderRow}>
      <div style={S.sliderHead}>
        <div style={S.sliderLabel}>{label}</div>
        <div style={S.sliderValue}>{display}</div>
      </div>

      <div style={S.sliderTrack}>
        <div style={S.sliderBg} />
        <div style={{ ...S.sliderFill, left: `${left}%`, width: `${width}%` }} />
        {isCenter ? <div style={{ ...S.sliderCenter, left: `${cPct}%` }} /> : null}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          onDoubleClick={() => onChange(isCenter ? 0 : min)}
          style={S.sliderInput}
        />
      </div>
    </div>
  )
}

function Metric({ label, value, tone }) {
  const toneMap = {
    dim: { fg: '#94a3b8', bd: 'rgba(148,163,184,0.18)', bg: 'rgba(2,6,23,0.28)' },
    cyan: { fg: '#93c5fd', bd: 'rgba(59,130,246,0.25)', bg: 'rgba(59,130,246,0.08)' },
    amber: { fg: '#fcd34d', bd: 'rgba(245,158,11,0.22)', bg: 'rgba(245,158,11,0.08)' },
    green: { fg: '#86efac', bd: 'rgba(34,197,94,0.22)', bg: 'rgba(34,197,94,0.08)' },
    red: { fg: '#fca5a5', bd: 'rgba(239,68,68,0.22)', bg: 'rgba(239,68,68,0.08)' }
  }[tone || 'dim']

  return (
    <div style={{ ...S.metric, borderColor: toneMap.bd, background: toneMap.bg }}>
      <div style={S.metricLabel}>{label}</div>
      <div style={{ ...S.metricValue, color: toneMap.fg }}>{value}</div>
    </div>
  )
}

function BusyOverlay({ label }) {
  return (
    <div style={S.busyOverlay}>
      <div style={S.spinner} />
      <div style={S.busyText}>{label || 'Working'}…</div>
    </div>
  )
}

function EmptyState({ onOpen }) {
  return (
    <div style={S.empty}>
      <div style={S.emptyPill}>Ready for Intake</div>
      <div style={S.emptyTitle}>Load an image to begin</div>
      <div style={S.emptySub}>JPG, PNG, WebP, BMP, TIFF · up to 50MB</div>
      <div style={{ height: 12 }} />
      <PrimaryButton onClick={onOpen}>Open Image</PrimaryButton>
      <div style={{ height: 14 }} />
      <div style={S.emptyFoot}>
        <span style={{ opacity: 0.75 }}>Shortcut:</span> <K>Ctrl</K>+<K>O</K>
      </div>
    </div>
  )
}

function CompareCanvas({
  originalPreview,
  resultPreview,
  holdOriginal,
  splitPos,
  setSplitPos,
  hasResult,
  label,
  resultLabel,
  zoom,
  fitMode
}) {
  // Spacebar hold forces original view
  const showResult = hasResult && !holdOriginal

  return (
    <div style={S.canvasInner}>
      {/* base original */}
      <img
        src={originalPreview}
        alt=""
        style={{
          ...S.img,
          objectFit: fitMode,
          transform: `scale(${zoom})`
        }}
      />

      {/* overlay result with wipe */}
      {showResult ? (
        <img
          src={resultPreview}
          alt=""
          style={{
            ...S.img,
            objectFit: fitMode,
            transform: `scale(${zoom})`,
            clipPath: `polygon(0 0, ${splitPos}% 0, ${splitPos}% 100%, 0 100%)`
          }}
        />
      ) : null}

      {/* slider control */}
      {showResult ? (
        <>
          <div style={{ ...S.splitLine, left: `${splitPos}%` }}>
            <div style={S.splitKnob}>⇆</div>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={splitPos}
            onChange={(e) => setSplitPos(Number(e.target.value))}
            style={S.splitSlider}
          />
          <div style={{ ...S.tagLeft }}>RESULT</div>
          <div style={{ ...S.tagRight }}>ORIGINAL</div>
        </>
      ) : (
        <div style={{ ...S.tagLeft }}>{hasResult ? 'ORIGINAL (Hold Space)' : 'ORIGINAL'}</div>
      )}

      {/* bottom label */}
      {resultLabel ? <div style={S.bottomTag}>{String(resultLabel).toUpperCase()}</div> : null}
      <div style={S.helpTag}>
        Hold <span style={S.helpKey}>Space</span> to compare · Drag split to wipe
      </div>
    </div>
  )
}

function Toast({ toast }) {
  const theme =
    toast.type === 'ok'
      ? { bd: 'rgba(34,197,94,0.30)', bg: 'rgba(34,197,94,0.10)', fg: '#86efac' }
      : toast.type === 'error'
        ? { bd: 'rgba(239,68,68,0.30)', bg: 'rgba(239,68,68,0.10)', fg: '#fca5a5' }
        : { bd: 'rgba(59,130,246,0.30)', bg: 'rgba(59,130,246,0.10)', fg: '#93c5fd' }

  return (
    <div style={{ ...S.toast, borderColor: theme.bd, background: theme.bg }}>
      <div style={{ fontWeight: 900, color: theme.fg }}>{toast.title}</div>
      <div style={{ opacity: 0.9 }}>{toast.message}</div>
    </div>
  )
}

// ----------------------------
// Helpers
// ----------------------------
function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n))
}
function round2(n) {
  return Math.round(n * 100) / 100
}
function fileNameFromPath(p) {
  try {
    return String(p).split(/[/\\]/).pop()
  } catch {
    return String(p || '')
  }
}

function toneNoise(v) {
  if (v == null) return 'dim'
  if (v > 15) return 'red'
  if (v > 8) return 'amber'
  return 'green'
}
function toneSharp(v) {
  if (v == null) return 'dim'
  if (v > 500) return 'green'
  if (v > 100) return 'amber'
  return 'red'
}
function toneRange(v) {
  if (v == null) return 'dim'
  if (v > 200) return 'green'
  return 'amber'
}
function toneSkin(v) {
  if (v == null) return 'dim'
  if (v > 15) return 'cyan'
  return 'dim'
}

// ----------------------------
// Colors / Styles
// ----------------------------
const C = {
  bg: '#070b14',
  panel: '#0b1220',
  panel2: '#0a1222',
  canvas: '#070b14',
  text: '#e5e7eb',
  dim: '#94a3b8',
  dim2: '#64748b',
  line: 'rgba(148,163,184,0.14)',
  line2: 'rgba(59,130,246,0.25)',
  blue: '#60a5fa',
  green: '#22c55e',
  amber: '#f59e0b',
  red: '#ef4444'
}

const S = {
  app: {
    height: '100vh',
    background: C.bg,
    color: C.text,
    fontFamily:
      "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji','Segoe UI Emoji'",
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden'
  },

  topBar: {
    height: 56,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 14px',
    borderBottom: `1px solid ${C.line}`,
    background:
      'linear-gradient(90deg, rgba(15,23,42,0.95), rgba(2,6,23,0.92))',
    backdropFilter: 'blur(8px)'
  },
  topLeft: { display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 },
  logoDot: {
    width: 12,
    height: 12,
    borderRadius: 999,
    background: 'linear-gradient(135deg, #60a5fa, #2563eb)'
  },
  appTitle: { fontSize: 13, fontWeight: 900, letterSpacing: 0.2 },
  appSub: { fontSize: 12, color: C.dim, minWidth: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },

  topRight: { display: 'flex', alignItems: 'center', gap: 10 },
  pill: {
    padding: '6px 10px',
    borderRadius: 999,
    border: '1px solid',
    fontSize: 12,
    fontWeight: 800,
    letterSpacing: 0.2
  },
  topActions: { display: 'flex', gap: 8, alignItems: 'center' },
  smallBtn: {
    height: 34,
    padding: '0 12px',
    borderRadius: 10,
    border: `1px solid ${C.line}`,
    background: 'rgba(2,6,23,0.35)',
    color: C.text,
    fontSize: 12,
    fontWeight: 800,
    cursor: 'pointer'
  },
  smallBtnDisabled: { opacity: 0.45, cursor: 'default' },

  viewGroup: { display: 'flex', gap: 8, alignItems: 'center' },
  smallChip: {
    height: 34,
    padding: '0 10px',
    borderRadius: 10,
    border: `1px solid ${C.line}`,
    background: 'rgba(2,6,23,0.20)',
    color: C.dim,
    fontSize: 12,
    fontWeight: 800,
    cursor: 'pointer'
  },

  bannerError: {
    display: 'flex',
    gap: 12,
    alignItems: 'center',
    padding: '10px 14px',
    borderBottom: '1px solid rgba(239,68,68,0.25)',
    background: 'rgba(239,68,68,0.10)',
    color: '#fecaca'
  },
  bannerBtn: {
    marginLeft: 'auto',
    height: 30,
    padding: '0 10px',
    borderRadius: 10,
    border: '1px solid rgba(239,68,68,0.25)',
    background: 'rgba(2,6,23,0.20)',
    color: '#fecaca',
    fontWeight: 900,
    cursor: 'pointer'
  },

  body: { display: 'flex', flex: 1, minHeight: 0 },

  sidebar: {
    width: 420,
    borderRight: `1px solid ${C.line}`,
    background: `linear-gradient(180deg, rgba(15,23,42,0.85), rgba(2,6,23,0.92))`,
    display: 'flex',
    flexDirection: 'column',
    minHeight: 0
  },
  brand: {
    padding: '14px 14px 10px',
    display: 'flex',
    alignItems: 'center',
    gap: 12
  },
  brandIcon: {
    width: 38,
    height: 38,
    borderRadius: 14,
    background: 'linear-gradient(135deg, rgba(96,165,250,1), rgba(37,99,235,1))',
    boxShadow: '0 10px 25px rgba(37,99,235,0.25)'
  },
  brandTitle: { fontSize: 14, fontWeight: 950, letterSpacing: 0.3 },
  brandSub: { fontSize: 12, color: C.dim2, fontWeight: 700 },

  tabRow: {
    padding: '0 10px 10px',
    display: 'grid',
    gridTemplateColumns: 'repeat(5, 1fr)',
    gap: 8
  },
  tab: {
    height: 34,
    borderRadius: 12,
    border: `1px solid ${C.line}`,
    fontSize: 12,
    fontWeight: 900,
    cursor: 'pointer'
  },
  tabActive: {
    background: 'linear-gradient(180deg, rgba(59,130,246,0.20), rgba(2,6,23,0.30))',
    borderColor: 'rgba(59,130,246,0.30)',
    color: '#bfdbfe'
  },
  tabIdle: {
    background: 'rgba(2,6,23,0.18)',
    color: C.dim
  },

  panel: {
    flex: 1,
    minHeight: 0,
    overflow: 'hidden',
    padding: '0 10px 10px'
  },
  panelInner: {
    height: '100%',
    borderRadius: 18,
    border: `1px solid ${C.line}`,
    background: 'rgba(2,6,23,0.25)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden'
  },
  panelHead: {
    padding: '14px 14px 10px',
    borderBottom: `1px solid ${C.line}`,
    background: 'rgba(2,6,23,0.20)'
  },
  panelTitle: { fontSize: 14, fontWeight: 950 },
  panelSub: { marginTop: 2, fontSize: 12, color: C.dim2, fontWeight: 700 },
  panelBody: {
    padding: 14,
    overflowY: 'auto',
    minHeight: 0
  },

  divider: {
    height: 1,
    background: C.line,
    margin: '12px 0'
  },

  fieldRow: {
    display: 'flex',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    gap: 10,
    padding: '6px 0'
  },
  fieldLabel: { fontSize: 12, color: C.dim2, fontWeight: 800 },
  fieldValue: { fontSize: 12, color: C.text, fontWeight: 800, textAlign: 'right', maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },

  mono: {
    fontFamily:
      "'JetBrains Mono','SF Mono','Menlo','Consolas',monospace"
  },

  hint: {
    marginTop: 10,
    fontSize: 12,
    color: C.dim,
    lineHeight: 1.5
  },
  kbd: {
    display: 'inline-block',
    padding: '2px 6px',
    borderRadius: 8,
    border: `1px solid ${C.line}`,
    background: 'rgba(2,6,23,0.25)',
    fontSize: 12,
    fontWeight: 900,
    margin: '0 2px'
  },

  toggleRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    padding: '8px 0'
  },
  toggleLabel: { fontSize: 12, fontWeight: 950 },
  toggleHint: { fontSize: 12, color: C.dim2, fontWeight: 700 },
  toggleBtn: {
    height: 34,
    padding: '0 12px',
    borderRadius: 999,
    border: `1px solid ${C.line}`,
    fontWeight: 950,
    cursor: 'pointer'
  },
  toggleOn: {
    background: 'rgba(59,130,246,0.16)',
    borderColor: 'rgba(59,130,246,0.30)',
    color: '#bfdbfe'
  },
  toggleOff: {
    background: 'rgba(2,6,23,0.20)',
    color: C.dim
  },

  btn: {
    width: '100%',
    height: 40,
    borderRadius: 14,
    border: `1px solid ${C.line}`,
    fontSize: 12,
    fontWeight: 950,
    cursor: 'pointer'
  },
  btnPrimary: {
    background: 'linear-gradient(180deg, rgba(59,130,246,0.22), rgba(2,6,23,0.28))',
    borderColor: 'rgba(59,130,246,0.30)',
    color: '#bfdbfe'
  },
  btnSecondary: {
    background: 'rgba(2,6,23,0.20)',
    color: C.dim
  },
  btnDisabled: { opacity: 0.5, cursor: 'default' },

  sectionTitle: { marginTop: 6, marginBottom: 8, fontSize: 12, fontWeight: 950, color: '#cbd5e1' },

  sliderRow: { marginBottom: 12 },
  sliderHead: { display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' },
  sliderLabel: { fontSize: 12, color: '#cbd5e1', fontWeight: 900 },
  sliderValue: { fontSize: 12, color: C.dim, fontWeight: 900, ...({ fontVariantNumeric: 'tabular-nums' }) },

  sliderTrack: { position: 'relative', height: 18, marginTop: 6, borderRadius: 10 },
  sliderBg: { position: 'absolute', left: 0, right: 0, top: '50%', height: 3, transform: 'translateY(-50%)', background: 'rgba(148,163,184,0.18)', borderRadius: 10 },
  sliderFill: { position: 'absolute', top: '50%', height: 3, transform: 'translateY(-50%)', background: 'rgba(96,165,250,0.95)', borderRadius: 10 },
  sliderCenter: { position: 'absolute', top: '50%', width: 2, height: 10, transform: 'translateY(-50%)', background: 'rgba(148,163,184,0.55)', borderRadius: 10 },
  sliderInput: { position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', opacity: 0, cursor: 'pointer' },

  pills: { display: 'flex', flexWrap: 'wrap', gap: 8 },
  pillBtn: {
    height: 34,
    padding: '0 12px',
    borderRadius: 999,
    border: `1px solid ${C.line}`,
    fontSize: 12,
    fontWeight: 900,
    cursor: 'pointer'
  },
  pillBtnOn: {
    background: 'rgba(59,130,246,0.16)',
    borderColor: 'rgba(59,130,246,0.30)',
    color: '#bfdbfe'
  },
  pillBtnOff: {
    background: 'rgba(2,6,23,0.20)',
    color: C.dim
  },

  layerGrid: { display: 'grid', gridTemplateColumns: '1fr', gap: 10 },
  layerCard: {
    width: '100%',
    borderRadius: 16,
    border: `1px solid ${C.line}`,
    padding: 12,
    cursor: 'pointer',
    textAlign: 'left'
  },
  layerTop: { display: 'flex', gap: 12, alignItems: 'flex-start' },
  layerIdx: {
    width: 34,
    height: 34,
    borderRadius: 12,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    border: `1px solid ${C.line}`,
    background: 'rgba(2,6,23,0.28)',
    fontWeight: 950,
    color: C.dim
  },
  layerName: { fontWeight: 950, fontSize: 13 },
  layerDesc: { marginTop: 4, fontSize: 12, color: C.dim2, fontWeight: 700, lineHeight: 1.35 },

  badge: {
    marginLeft: 'auto',
    height: 28,
    padding: '0 10px',
    borderRadius: 999,
    display: 'flex',
    alignItems: 'center',
    border: `1px solid ${C.line}`,
    fontSize: 12,
    fontWeight: 950
  },
  badgeOn: { background: 'rgba(34,197,94,0.12)', borderColor: 'rgba(34,197,94,0.24)', color: '#86efac' },
  badgeOff: { background: 'rgba(148,163,184,0.10)', borderColor: 'rgba(148,163,184,0.18)', color: C.dim },
  badgeStrong: {
    height: 30,
    padding: '0 12px',
    borderRadius: 999,
    border: '1px solid rgba(59,130,246,0.30)',
    background: 'rgba(59,130,246,0.14)',
    color: '#bfdbfe',
    fontWeight: 950,
    display: 'flex',
    alignItems: 'center'
  },

  metricGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 },
  metric: { padding: 12, borderRadius: 16, border: `1px solid ${C.line}` },
  metricLabel: { fontSize: 12, color: C.dim2, fontWeight: 900 },
  metricValue: { marginTop: 6, fontSize: 16, fontWeight: 950, letterSpacing: 0.2 },

  recCard: {
    borderRadius: 18,
    border: `1px solid ${C.line}`,
    background: 'rgba(2,6,23,0.25)',
    padding: 14
  },
  recHead: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 },
  recTitle: { fontSize: 12, color: C.dim2, fontWeight: 950, letterSpacing: 0.2 },
  recBody: { marginTop: 10, fontSize: 12, color: C.dim, lineHeight: 1.5, fontWeight: 700 },

  canvas: {
    flex: 1,
    minWidth: 0,
    background:
      'radial-gradient(circle at 20% 10%, rgba(59,130,246,0.18) 0%, rgba(2,6,23,0.92) 45%, rgba(2,6,23,1) 100%)',
    position: 'relative',
    overflow: 'hidden'
  },
  canvasInner: { position: 'absolute', inset: 0 },

  img: {
    position: 'absolute',
    inset: 0,
    width: '100%',
    height: '100%'
  },

  splitLine: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
    background: 'rgba(255,255,255,0.45)',
    transform: 'translateX(-50%)',
    pointerEvents: 'none'
  },
  splitKnob: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%,-50%)',
    width: 34,
    height: 34,
    borderRadius: 999,
    background: 'rgba(255,255,255,0.92)',
    color: '#0b1220',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 950,
    boxShadow: '0 10px 25px rgba(0,0,0,0.35)'
  },
  splitSlider: {
    position: 'absolute',
    inset: 0,
    width: '100%',
    height: '100%',
    opacity: 0,
    cursor: 'ew-resize'
  },

  tagLeft: {
    position: 'absolute',
    top: 14,
    left: 14,
    padding: '6px 10px',
    borderRadius: 999,
    border: `1px solid ${C.line}`,
    background: 'rgba(2,6,23,0.35)',
    fontWeight: 950,
    fontSize: 12,
    color: '#cbd5e1'
  },
  tagRight: {
    position: 'absolute',
    top: 14,
    right: 14,
    padding: '6px 10px',
    borderRadius: 999,
    border: `1px solid ${C.line}`,
    background: 'rgba(2,6,23,0.35)',
    fontWeight: 950,
    fontSize: 12,
    color: '#cbd5e1'
  },
  bottomTag: {
    position: 'absolute',
    bottom: 18,
    left: '50%',
    transform: 'translateX(-50%)',
    padding: '6px 12px',
    borderRadius: 999,
    border: '1px solid rgba(59,130,246,0.30)',
    background: 'rgba(59,130,246,0.14)',
    color: '#bfdbfe',
    fontWeight: 950,
    fontSize: 12,
    letterSpacing: 0.3
  },
  helpTag: {
    position: 'absolute',
    bottom: 18,
    right: 18,
    padding: '6px 12px',
    borderRadius: 999,
    border: `1px solid ${C.line}`,
    background: 'rgba(2,6,23,0.35)',
    color: C.dim,
    fontWeight: 800,
    fontSize: 12
  },
  helpKey: {
    padding: '1px 8px',
    borderRadius: 999,
    border: `1px solid ${C.line}`,
    background: 'rgba(2,6,23,0.25)',
    fontWeight: 950,
    color: '#cbd5e1'
  },

  busyOverlay: {
    position: 'absolute',
    inset: 0,
    background: 'rgba(2,6,23,0.55)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    zIndex: 10
  },
  spinner: {
    width: 14,
    height: 14,
    borderRadius: 999,
    border: '2px solid rgba(255,255,255,0.20)',
    borderTopColor: 'rgba(255,255,255,0.75)',
    animation: 'spin 0.9s linear infinite'
  },
  busyText: { fontSize: 13, fontWeight: 900, color: '#cbd5e1' },

  empty: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 20,
    textAlign: 'center'
  },
  emptyPill: {
    padding: '6px 12px',
    borderRadius: 999,
    border: '1px solid rgba(59,130,246,0.30)',
    background: 'rgba(59,130,246,0.12)',
    color: '#bfdbfe',
    fontWeight: 950,
    fontSize: 12
  },
  emptyTitle: { marginTop: 8, fontSize: 24, fontWeight: 950, letterSpacing: 0.2 },
  emptySub: { fontSize: 13, color: C.dim, fontWeight: 800 },
  emptyFoot: { fontSize: 12, color: C.dim, fontWeight: 800 },

  toast: {
    position: 'fixed',
    bottom: 16,
    left: 16,
    width: 360,
    borderRadius: 16,
    border: `1px solid ${C.line}`,
    padding: 12,
    background: 'rgba(2,6,23,0.35)',
    zIndex: 999,
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
    boxShadow: '0 18px 45px rgba(0,0,0,0.35)'
  }
}

// Add keyframes via inline <style> once (safe)
if (typeof document !== 'undefined' && !document.getElementById('__aurora_styles')) {
  const el = document.createElement('style')
  el.id = '__aurora_styles'
  el.textContent = `
    @keyframes spin { from { transform: rotate(0deg);} to { transform: rotate(360deg);} }
  `
  document.head.appendChild(el)
}
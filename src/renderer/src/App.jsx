import React, { useState, useCallback } from 'react'

const ENHANCE_MODES = [
  { id: 'pro', label: 'Pro Detail' },
  { id: 'color', label: 'Color Restore' },
  { id: 'smooth', label: 'Denoise' },
  { id: 'light', label: 'Lighting Fix' },
  { id: 'portrait', label: 'Portrait' },
]

const XRAY_MODES = [
  { id: 'none', label: 'Off' },
  { id: 'structure', label: 'Structure' },
  { id: 'depth', label: 'Depth' },
  { id: 'frequency', label: 'Frequency' },
  { id: 'thermal', label: 'Thermal' },
  { id: 'bones', label: 'Bones' },
]

const DEFAULTS = {
  exposure: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0,
  temperature: 0, tint: 0, vibrance: 0, saturation: 0,
  clarity: 0, dehaze: 0, sharpness: 0,
  grain: 0, vignette: 0,
  xray: 'none', xray_blend: 100,
}

const App = () => {
  const [originalPath, setOriginalPath] = useState(null)
  const [originalPreview, setOriginalPreview] = useState(null)
  const [resultPreview, setResultPreview] = useState(null)
  const [tempPath, setTempPath] = useState(null)
  const [resultLabel, setResultLabel] = useState('')

  const [busy, setBusy] = useState(false)
  const [busyMsg, setBusyMsg] = useState('')
  const [saving, setSaving] = useState(false)
  const [holdOriginal, setHoldOriginal] = useState(false)
  const [splitPos, setSplitPos] = useState(50)

  const [adj, setAdj] = useState({ ...DEFAULTS })
  const [activeLayers, setActiveLayers] = useState(['pro'])

  const [diagnosis, setDiagnosis] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [rec, setRec] = useState(null)

  const [panels, setPanels] = useState({
    source: true, analysis: false, tone: true, color: false,
    detail: false, effects: false, xray: false, layers: true,
  })

  const togglePanel = (k) => setPanels(p => ({ ...p, [k]: !p[k] }))
  const setA = (k, v) => setAdj(p => ({ ...p, [k]: v }))
  const toggleLayer = (id) => setActiveLayers(p => {
    if (p.includes(id)) { const n = p.filter(m => m !== id); return n.length ? n : p }
    return [...p, id]
  })

  const dirty = Object.keys(DEFAULTS).some(k => adj[k] !== DEFAULTS[k])
  const fileName = originalPath ? originalPath.split(/[/\\]/).pop() : null

  const handleOpen = useCallback(async () => {
    const f = await window.api.selectImage()
    if (!f) return
    setOriginalPath(f.path); setOriginalPreview(f.preview)
    setResultPreview(null); setTempPath(null); setResultLabel('')
    setDiagnosis(null); setMetrics(null); setRec(null)
    setAdj({ ...DEFAULTS })
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!originalPath) return
    setBusy(true); setBusyMsg('Analyzing')
    try {
      const r = await window.api.runAutopilot(originalPath)
      if (r.recommendations) {
        setDiagnosis(r.analysis); setMetrics(r.metrics); setRec(r.recommendations)
        if (r.recommendations.best_mode) setActiveLayers([r.recommendations.best_mode])
        setAdj(p => ({
          ...p,
          exposure: r.recommendations.exposure || 0,
          saturation: Math.round((r.recommendations.saturation - 1) * 100) || 0,
        }))
        setPanels(p => ({ ...p, analysis: true }))
      }
    } finally { setBusy(false) }
  }, [originalPath])

  const handleApply = useCallback(async () => {
    if (!originalPath || !dirty) return
    setBusy(true); setBusyMsg('Adjusting')
    try {
      const r = await window.api.adjustImage(originalPath, adj)
      if (r.temp_path) {
        setTempPath(r.temp_path); setResultPreview(r.preview); setResultLabel('Adjusted')
      } else if (r.error) alert(r.error)
    } finally { setBusy(false) }
  }, [originalPath, adj, dirty])

  const handleEnhance = useCallback(async () => {
    if (!originalPath) return
    setBusy(true); setBusyMsg('Enhancing')
    try {
      const r = await window.api.enhanceImage(originalPath, activeLayers)
      if (r.temp_path) {
        setTempPath(r.temp_path); setResultPreview(r.preview)
        setResultLabel((r.applied_modes || activeLayers).join(' + '))
      } else if (r.error) alert(r.error)
    } finally { setBusy(false) }
  }, [originalPath, activeLayers])

  const handleSave = useCallback(async () => {
    if (!tempPath) return
    setSaving(true)
    try {
      const r = await window.api.saveImage(tempPath)
      if (r.saved_path) { setSaving('done'); setTimeout(() => setSaving(false), 2000) }
      else { if (r.error !== 'Cancelled') alert(r.error); setSaving(false) }
    } catch { setSaving(false) }
  }, [tempPath])

  const handleReset = () => setAdj({ ...DEFAULTS })

  // ════════════════════════════════════════════════════════════════════
  return (
    <div style={R.root}>
      {/* ── SIDEBAR ── */}
      <div style={R.sidebar}>
        <div style={R.sideScroll}>
          {/* Brand */}
          <div style={R.brand}>
            <div style={R.brandIcon}/>
            <div>
              <div style={R.brandName}>PROENHANCE</div>
              <div style={R.brandSub}>processing studio</div>
            </div>
          </div>
          <div style={R.sep}/>

          {/* Source */}
          <PH title="SOURCE" open={panels.source} toggle={() => togglePanel('source')}>
            <Btn onClick={handleOpen}>{originalPath ? 'CHANGE IMAGE' : 'OPEN IMAGE'}</Btn>
            {fileName && <div style={R.fileInfo}>{fileName}</div>}
            {metrics && <div style={R.fileDim}>{metrics.width}x{metrics.height}</div>}
          </PH>

          {/* Autopilot */}
          <PH title="AUTOPILOT" open={panels.analysis} toggle={() => togglePanel('analysis')}>
            <Btn onClick={handleAnalyze} disabled={busy || !originalPath} accent>
              {busy && busyMsg === 'Analyzing' ? 'ANALYZING...' : 'RUN ANALYSIS'}
            </Btn>
            {diagnosis && <div style={R.diagBox}>{diagnosis}</div>}
            {metrics && (
              <div style={R.mGrid}>
                <MC l="NOISE" v={metrics.noise} c={metrics.noise > 15 ? C.red : metrics.noise > 8 ? C.amber : C.green}/>
                <MC l="SHARP" v={metrics.sharpness > 500 ? 'HI' : metrics.sharpness > 100 ? 'MED' : 'LO'} c={metrics.sharpness > 500 ? C.green : metrics.sharpness > 100 ? C.amber : C.red}/>
                <MC l="RANGE" v={metrics.dynamic_range} c={metrics.dynamic_range > 200 ? C.green : C.amber}/>
                <MC l="SKIN" v={`${metrics.skin_pct}%`} c={metrics.skin_pct > 15 ? C.cyan : C.dim}/>
              </div>
            )}
            {rec && (
              <div style={R.recBox}>
                <span style={R.recLbl}>REC </span>
                <span style={R.recVal}>{rec.best_mode.toUpperCase()}</span>
                <div style={R.recWhy}>{rec.mode_reason}</div>
              </div>
            )}
          </PH>

          <div style={R.sep}/>

          {/* Tone */}
          <PH title="TONE" open={panels.tone} toggle={() => togglePanel('tone')}>
            <Sl l="Exposure" v={adj.exposure} min={-3} max={3} step={0.05} set={v=>setA('exposure',v)}/>
            <Sl l="Contrast" v={adj.contrast} min={-100} max={100} step={1} set={v=>setA('contrast',v)}/>
            <Sl l="Highlights" v={adj.highlights} min={-100} max={100} step={1} set={v=>setA('highlights',v)}/>
            <Sl l="Shadows" v={adj.shadows} min={-100} max={100} step={1} set={v=>setA('shadows',v)}/>
            <Sl l="Whites" v={adj.whites} min={-100} max={100} step={1} set={v=>setA('whites',v)}/>
            <Sl l="Blacks" v={adj.blacks} min={-100} max={100} step={1} set={v=>setA('blacks',v)}/>
          </PH>

          {/* Color */}
          <PH title="COLOR" open={panels.color} toggle={() => togglePanel('color')}>
            <Sl l="Temp" v={adj.temperature} min={-100} max={100} step={1} set={v=>setA('temperature',v)}/>
            <Sl l="Tint" v={adj.tint} min={-100} max={100} step={1} set={v=>setA('tint',v)}/>
            <Sl l="Vibrance" v={adj.vibrance} min={-100} max={100} step={1} set={v=>setA('vibrance',v)}/>
            <Sl l="Saturation" v={adj.saturation} min={-100} max={100} step={1} set={v=>setA('saturation',v)}/>
          </PH>

          {/* Detail */}
          <PH title="DETAIL" open={panels.detail} toggle={() => togglePanel('detail')}>
            <Sl l="Clarity" v={adj.clarity} min={-100} max={100} step={1} set={v=>setA('clarity',v)}/>
            <Sl l="Dehaze" v={adj.dehaze} min={0} max={100} step={1} set={v=>setA('dehaze',v)}/>
            <Sl l="Sharpen" v={adj.sharpness} min={0} max={100} step={1} set={v=>setA('sharpness',v)}/>
          </PH>

          {/* Effects */}
          <PH title="EFFECTS" open={panels.effects} toggle={() => togglePanel('effects')}>
            <Sl l="Grain" v={adj.grain} min={0} max={100} step={1} set={v=>setA('grain',v)}/>
            <Sl l="Vignette" v={adj.vignette} min={-100} max={100} step={1} set={v=>setA('vignette',v)}/>
          </PH>

          {/* X-Ray */}
          <PH title="X-RAY VISION" open={panels.xray} toggle={() => togglePanel('xray')}>
            <div style={R.xrayRow}>
              {XRAY_MODES.map(m => (
                <button key={m.id}
                  style={{
                    ...R.xrayBtn,
                    backgroundColor: adj.xray === m.id ? (m.id === 'none' ? '#1a1a1e' : '#1a1630') : 'transparent',
                    color: adj.xray === m.id ? (m.id === 'none' ? C.dim : C.cyan) : C.dim,
                    borderColor: adj.xray === m.id && m.id !== 'none' ? '#2d2660' : '#1a1a1e',
                  }}
                  onClick={() => setA('xray', m.id)}
                >{m.label}</button>
              ))}
            </div>
            {adj.xray !== 'none' && (
              <Sl l="Blend" v={adj.xray_blend} min={0} max={100} step={1} set={v=>setA('xray_blend',v)}/>
            )}
          </PH>

          <div style={R.sep}/>

          {/* Apply adjustments */}
          {dirty && (
            <div style={R.adjBar}>
              <Btn onClick={handleApply} disabled={busy} primary>
                {busy && busyMsg === 'Adjusting' ? 'APPLYING...' : 'APPLY ADJUSTMENTS'}
              </Btn>
              <Btn onClick={handleReset}>RESET</Btn>
            </div>
          )}

          <div style={R.sep}/>

          {/* Enhancement Layers */}
          <PH title="ENHANCEMENT LAYERS" open={panels.layers} toggle={() => togglePanel('layers')}>
            <div style={R.layerCol}>
              {ENHANCE_MODES.map(m => {
                const on = activeLayers.includes(m.id)
                const idx = on ? activeLayers.indexOf(m.id) + 1 : null
                return (
                  <button key={m.id} style={{
                    ...R.layerChip,
                    backgroundColor: on ? '#0c1a2e' : 'transparent',
                    borderColor: on ? '#1d3461' : '#141418',
                  }} onClick={() => toggleLayer(m.id)}>
                    <span style={{ ...R.layerIdx, color: on ? C.cyan : '#222' }}>{idx || '-'}</span>
                    <span style={R.layerLbl}>{m.label}</span>
                    <span style={{ ...R.layerBadge, backgroundColor: on ? '#1d3461' : '#141418', color: on ? C.cyan : '#333' }}>
                      {on ? 'ON' : 'OFF'}
                    </span>
                  </button>
                )
              })}
            </div>
            {activeLayers.length > 1 && (
              <div style={R.pipeline}>{activeLayers.join(' \u2192 ')}</div>
            )}
            <Btn onClick={handleEnhance} disabled={busy || !originalPath} primary style={{ marginTop: 6 }}>
              {busy && busyMsg === 'Enhancing' ? 'PROCESSING...' : 'RUN ENHANCE'}
            </Btn>
          </PH>
        </div>

        {/* Bottom pinned */}
        <div style={R.bottomBar}>
          {tempPath && (
            <Btn onClick={handleSave} disabled={saving === true} full>
              {saving === 'done' ? 'SAVED' : saving ? 'SAVING...' : 'SAVE AS...'}
            </Btn>
          )}
        </div>
      </div>

      {/* ── CANVAS ── */}
      <div style={R.canvas}>
        {busy && (
          <div style={R.busyOverlay}>
            <div style={R.busyDot}/><span style={R.busyText}>{busyMsg}...</span>
          </div>
        )}

        {!originalPath && (
          <div style={R.empty}>
            <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#1a1a1e" strokeWidth="1">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <path d="M21 15l-5-5L5 21"/>
            </svg>
            <div style={R.emptyTitle}>Open an image to begin</div>
            <div style={R.emptyHint}>JPG / PNG / WebP / BMP / TIFF</div>
          </div>
        )}

        {originalPath && !resultPreview && (
          <>
            <img src={originalPreview} style={R.img} alt=""/>
            <div style={R.canvasTag}>ORIGINAL</div>
          </>
        )}

        {originalPath && resultPreview && (
          <>
            <img src={holdOriginal ? originalPreview : originalPreview} style={R.img} alt=""/>
            {!holdOriginal && (
              <img src={resultPreview} style={{
                ...R.img,
                clipPath: `polygon(0 0, ${splitPos}% 0, ${splitPos}% 100%, 0 100%)`
              }} alt=""/>
            )}
            {!holdOriginal && (
              <>
                <div style={{ ...R.splitLine, left: `${splitPos}%` }}>
                  <div style={R.splitKnob}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#09090b" strokeWidth="3">
                      <path d="M9 4l-7 8 7 8"/><path d="M15 4l7 8-7 8"/>
                    </svg>
                  </div>
                </div>
                <input type="range" min="0" max="100" value={splitPos}
                  onChange={e => setSplitPos(e.target.value)} style={R.splitSlider}/>
              </>
            )}
            <div style={{ ...R.label, top: 12, left: 12 }}>
              {holdOriginal ? 'ORIGINAL' : 'RESULT'}
            </div>
            {!holdOriginal && (
              <div style={{ ...R.label, top: 12, right: 12, left: 'auto' }}>ORIGINAL</div>
            )}
            {resultLabel && (
              <div style={R.resultTag}>{resultLabel.toUpperCase()}</div>
            )}
            <button style={R.compareBtn}
              onMouseDown={() => setHoldOriginal(true)}
              onMouseUp={() => setHoldOriginal(false)}
              onMouseLeave={() => setHoldOriginal(false)}
            >HOLD TO COMPARE</button>
          </>
        )}
      </div>
    </div>
  )
}


// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

function PH({ title, open, toggle, children }) {
  return (
    <div style={{ marginBottom: 2 }}>
      <button style={R.ph} onClick={toggle}>
        <span>{title}</span>
        <span style={{ color: '#2a2a2e', fontSize: 10 }}>{open ? '\u25BC' : '\u25B6'}</span>
      </button>
      {open && <div style={R.pb}>{children}</div>}
    </div>
  )
}

function Btn({ children, onClick, disabled, accent, primary, full, style }) {
  let bg = '#111114', bd = '#1a1a1e', fg = '#555'
  if (accent) { bg = '#0f1520'; bd = '#1a2a40'; fg = C.cyan }
  if (primary) { bg = '#0a1628'; bd = '#1d3461'; fg = '#7db8f0' }
  return (
    <button style={{
      padding: '7px 0', fontSize: 10, fontWeight: 700, letterSpacing: '0.8px',
      fontFamily: "'JetBrains Mono','SF Mono','Consolas',monospace",
      backgroundColor: bg, color: fg, border: `1px solid ${bd}`, borderRadius: 3,
      cursor: disabled ? 'default' : 'pointer', opacity: disabled ? 0.4 : 1,
      width: full || true ? '100%' : 'auto', transition: 'opacity 0.15s', ...style,
    }} onClick={onClick} disabled={disabled}>{children}</button>
  )
}

function Sl({ l, v, min, max, step, set }) {
  const isCenter = min < 0
  const pct = ((v - min) / (max - min)) * 100
  const cPct = isCenter ? ((0 - min) / (max - min)) * 100 : 0
  const left = isCenter ? Math.min(pct, cPct) : 0
  const width = isCenter ? Math.abs(pct - cPct) : pct
  const display = (step < 1 && step > 0) ? v.toFixed(2) : Math.round(v)

  return (
    <div style={R.slRow}>
      <div style={R.slHead}>
        <span style={R.slLabel}>{l}</span>
        <span style={R.slVal}>{display}</span>
      </div>
      <div style={R.slTrack}>
        <div style={R.slBg}/>
        <div style={{ ...R.slFill, left: `${left}%`, width: `${width}%` }}/>
        {isCenter && <div style={{ ...R.slCenter, left: `${cPct}%` }}/>}
        <input type="range" min={min} max={max} step={step} value={v}
          onChange={e => set(parseFloat(e.target.value))}
          onDoubleClick={() => set(isCenter ? 0 : min)}
          style={R.slInput}/>
      </div>
    </div>
  )
}

function MC({ l, v, c }) {
  return (
    <div style={R.mc}>
      <div style={R.mcL}>{l}</div>
      <div style={{ ...R.mcV, color: c }}>{v}</div>
    </div>
  )
}

const C = {
  cyan: '#5eafd6', amber: '#d4a344', red: '#c44', green: '#4a4', dim: '#333',
  bg: '#09090b', sidebar: '#0b0b0e', border: '#141418',
}

// ═══════════════════════════════════════════════════════════════════════════════
// STYLES
// ═══════════════════════════════════════════════════════════════════════════════

const R = {
  root: {
    display: 'flex', height: '100vh', margin: 0,
    fontFamily: "'JetBrains Mono','SF Mono','Consolas',monospace",
    backgroundColor: C.bg, color: '#888', userSelect: 'none', fontSize: 11,
  },
  sidebar: {
    width: 284, minWidth: 284, display: 'flex', flexDirection: 'column',
    backgroundColor: C.sidebar, borderRight: `1px solid ${C.border}`,
  },
  sideScroll: { flex: 1, overflowY: 'auto', overflowX: 'hidden', padding: '12px 12px 6px' },

  brand: { display: 'flex', alignItems: 'center', gap: 10, padding: '2px 0 10px' },
  brandIcon: {
    width: 24, height: 24, borderRadius: 4, flexShrink: 0,
    background: `linear-gradient(135deg, ${C.cyan} 0%, #3a6080 100%)`,
  },
  brandName: { fontSize: 12, fontWeight: 800, color: '#ccc', letterSpacing: '1.5px' },
  brandSub: { fontSize: 8, color: '#2a2a2e', letterSpacing: '1.5px', textTransform: 'uppercase' },

  sep: { height: 1, backgroundColor: C.border, margin: '4px 0' },

  ph: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    width: '100%', padding: '5px 1px', background: 'none', border: 'none',
    color: '#3a3a3e', cursor: 'pointer', fontSize: 9, fontWeight: 700,
    letterSpacing: '1.2px',
    fontFamily: "'JetBrains Mono','SF Mono','Consolas',monospace",
  },
  pb: { padding: '3px 0 4px' },

  fileInfo: { fontSize: 9, color: '#333', marginTop: 4, wordBreak: 'break-all', lineHeight: 1.4 },
  fileDim: { fontSize: 9, color: '#222', marginTop: 1 },

  diagBox: {
    marginTop: 5, padding: '6px 7px', fontSize: 9, lineHeight: 1.6,
    color: '#556', backgroundColor: '#0d0d10', borderRadius: 3, border: `1px solid ${C.border}`,
  },
  mGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, marginTop: 5 },
  mc: { padding: '4px 6px', backgroundColor: '#0d0d10', borderRadius: 2, border: `1px solid ${C.border}` },
  mcL: { fontSize: 8, color: '#2a2a2e', letterSpacing: '0.5px' },
  mcV: { fontSize: 11, fontWeight: 800, marginTop: 1 },

  recBox: { marginTop: 5, padding: '5px 7px', backgroundColor: '#0a0e0a', borderRadius: 3, border: '1px solid #141a14' },
  recLbl: { fontSize: 8, color: '#333' },
  recVal: { fontSize: 10, color: C.green, fontWeight: 800 },
  recWhy: { fontSize: 8, color: '#2a3a2a', marginTop: 1 },

  // Slider
  slRow: { marginBottom: 8 },
  slHead: { display: 'flex', justifyContent: 'space-between', marginBottom: 2 },
  slLabel: { fontSize: 10, color: '#444' },
  slVal: { fontSize: 10, color: '#777', fontWeight: 700 },
  slTrack: { position: 'relative', height: 14, display: 'flex', alignItems: 'center' },
  slBg: { position: 'absolute', left: 0, right: 0, height: 2, backgroundColor: '#1a1a1e', borderRadius: 1 },
  slFill: { position: 'absolute', height: 2, backgroundColor: C.cyan, borderRadius: 1 },
  slCenter: { position: 'absolute', width: 1, height: 6, backgroundColor: '#2a2a2e', top: '50%', transform: 'translateY(-50%)' },
  slInput: { position: 'absolute', width: '100%', height: '100%', margin: 0, opacity: 0, cursor: 'pointer', zIndex: 2 },

  adjBar: { display: 'flex', flexDirection: 'column', gap: 3, padding: '4px 0' },

  // X-Ray
  xrayRow: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 2, marginBottom: 6 },
  xrayBtn: {
    padding: '5px 2px', fontSize: 9, fontWeight: 600, letterSpacing: '0.3px',
    border: '1px solid', borderRadius: 3, cursor: 'pointer',
    fontFamily: "'JetBrains Mono','SF Mono','Consolas',monospace",
    background: 'none', transition: 'all 0.1s',
  },

  // Layers
  layerCol: { display: 'flex', flexDirection: 'column', gap: 2 },
  layerChip: {
    display: 'flex', alignItems: 'center', gap: 6, padding: '5px 6px',
    border: '1px solid', borderRadius: 3, cursor: 'pointer', width: '100%',
    background: 'none', color: '#888', fontSize: 11,
    fontFamily: "'JetBrains Mono','SF Mono','Consolas',monospace",
    transition: 'all 0.1s',
  },
  layerIdx: { fontSize: 9, fontWeight: 800, width: 12, textAlign: 'center', flexShrink: 0 },
  layerLbl: { flex: 1, textAlign: 'left', fontWeight: 500 },
  layerBadge: { fontSize: 8, fontWeight: 800, padding: '1px 5px', borderRadius: 2, letterSpacing: '0.5px' },
  pipeline: { marginTop: 4, fontSize: 9, color: C.cyan, opacity: 0.5 },

  bottomBar: { padding: '8px 12px', borderTop: `1px solid ${C.border}` },

  // Canvas
  canvas: {
    flex: 1, position: 'relative', overflow: 'hidden',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    backgroundColor: C.bg,
  },
  img: { position: 'absolute', width: '100%', height: '100%', objectFit: 'contain' },

  busyOverlay: {
    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
    backgroundColor: 'rgba(9,9,11,0.6)', zIndex: 20,
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
  },
  busyDot: {
    width: 8, height: 8, borderRadius: '50%', backgroundColor: C.cyan,
    animation: 'pulse 1s infinite',
  },
  busyText: { fontSize: 11, color: '#555', letterSpacing: '1px' },

  empty: { textAlign: 'center' },
  emptyTitle: { fontSize: 12, color: '#1a1a1e', marginTop: 14, letterSpacing: '0.5px' },
  emptyHint: { fontSize: 9, color: '#141418', marginTop: 4 },

  canvasTag: {
    position: 'absolute', bottom: 14, left: '50%', transform: 'translateX(-50%)',
    fontSize: 9, color: '#333', backgroundColor: 'rgba(9,9,11,0.8)', padding: '3px 10px',
    borderRadius: 2, border: `1px solid ${C.border}`, letterSpacing: '1px',
  },

  splitLine: {
    position: 'absolute', top: 0, bottom: 0, width: 1,
    backgroundColor: 'rgba(255,255,255,0.5)', transform: 'translateX(-50%)',
    pointerEvents: 'none', zIndex: 5,
  },
  splitKnob: {
    position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)',
    width: 30, height: 30, backgroundColor: '#fff', borderRadius: '50%',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    boxShadow: '0 2px 10px rgba(0,0,0,0.6)',
  },
  splitSlider: {
    position: 'absolute', width: '100%', height: '100%',
    top: 0, left: 0, margin: 0, zIndex: 10, cursor: 'ew-resize', opacity: 0,
  },
  label: {
    position: 'absolute', zIndex: 4, fontSize: 9, fontWeight: 700, letterSpacing: '1px',
    padding: '3px 8px', borderRadius: 2,
    backgroundColor: 'rgba(9,9,11,0.7)', color: '#555', border: `1px solid ${C.border}`,
  },
  resultTag: {
    position: 'absolute', bottom: 14, left: '50%', transform: 'translateX(-50%)',
    zIndex: 4, fontSize: 9, fontWeight: 700, letterSpacing: '1px',
    padding: '3px 10px', borderRadius: 2,
    backgroundColor: 'rgba(10,22,40,0.85)', color: C.cyan, border: '1px solid #1d3461',
  },
  compareBtn: {
    position: 'absolute', bottom: 14, right: 14, zIndex: 11,
    fontSize: 9, fontWeight: 700, letterSpacing: '0.8px', color: '#333',
    backgroundColor: 'rgba(11,11,14,0.9)', padding: '5px 10px',
    borderRadius: 2, border: `1px solid ${C.border}`, cursor: 'pointer',
    fontFamily: "'JetBrains Mono','SF Mono','Consolas',monospace",
  },
}

export default App
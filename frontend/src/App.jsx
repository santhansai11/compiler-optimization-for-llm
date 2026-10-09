import { useEffect, useMemo, useState } from 'react'
import {
  Activity, ArrowUpRight, Boxes, Braces, ChartNoAxesColumn,
  Check, ChevronDown, CircleAlert, Cpu, Download, Gauge, GitBranch,
  Layers3, Play, RefreshCw, Server, SlidersHorizontal, Workflow, Zap,
} from 'lucide-react'
import './App.css'

const quickCases = [
  [1, 32, 64], [2, 64, 64], [4, 128, 128], [2, 256, 256],
]

async function api(path, body) {
  const response = await fetch(`/api/${path}`, {
    method: body ? 'POST' : 'GET',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(payload.error || payload.message || `API error (${response.status})`)
  return payload
}

function Panel({ title, eyebrow, icon: Icon, children, action }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div className="panel-heading">
          {Icon && <span className="panel-icon"><Icon size={16} /></span>}
          <div>{eyebrow && <div className="eyebrow">{eyebrow}</div>}<h2>{title}</h2></div>
        </div>
        {action}
      </div>
      {children}
    </section>
  )
}

function BarChart({ rows, unit = '', precision = 2, empty = 'Run an optimization to populate this chart.' }) {
  const max = Math.max(0, ...rows.map((row) => Math.max(0, Number(row.value) || 0)))
  if (!rows.length) return <div className="chart-empty">{empty}</div>
  return (
    <div className="bar-list">
      {rows.map((row, index) => {
        const value = Math.max(0, Number(row.value) || 0)
        const width = max === 0 ? 0 : Math.max(2, (value / max) * 100)
        return (
          <div className="bar-row" key={`${row.label}-${index}`}>
            <div className="bar-row-top"><span>{row.label}</span><strong>{value.toLocaleString(undefined, { maximumFractionDigits: precision })}{unit}</strong></div>
            <div className="bar-track"><div className={`bar-fill ${row.tone || ''}`} style={{ width: `${width}%` }} /></div>
            {row.detail && <div className="bar-detail">{row.detail}</div>}
          </div>
        )
      })}
    </div>
  )
}

function MetricCard({ label, value, foot, icon: Icon, tone = '' }) {
  return (
    <div className="metric-card panel">
      <div className="metric-header"><span>{label}</span><span className={`metric-icon ${tone}`}><Icon size={16} /></span></div>
      <div className="metric-value">{value}</div>
      <div className="metric-footer">{foot}</div>
    </div>
  )
}

function GraphView({ graph, title, optimized = false }) {
  const [zoom, setZoom] = useState(1)
  const positions = useMemo(() => {
    if (!graph?.nodes?.length) return { nodes: [], edges: [], width: 780, height: 250 }
    const byId = new Map(graph.nodes.map((node) => [node.id, node]))
    const indegree = new Map(graph.nodes.map((node) => [node.id, 0]))
    const children = new Map(graph.nodes.map((node) => [node.id, []]))
    for (const edge of graph.edges) {
      if (children.has(edge.source) && indegree.has(edge.target)) {
        children.get(edge.source).push(edge.target)
        indegree.set(edge.target, indegree.get(edge.target) + 1)
      }
    }
    const depth = new Map(graph.nodes.map((node) => [node.id, 0]))
    const queue = graph.nodes.filter((node) => indegree.get(node.id) === 0).map((node) => node.id)
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const source = queue[cursor]
      for (const target of children.get(source)) {
        depth.set(target, Math.max(depth.get(target), depth.get(source) + 1))
        indegree.set(target, indegree.get(target) - 1)
        if (indegree.get(target) === 0) queue.push(target)
      }
    }
    const levels = new Map()
    for (const node of graph.nodes) {
      const level = depth.get(node.id) || 0
      if (!levels.has(level)) levels.set(level, [])
      levels.get(level).push(node)
    }
    const sorted = [...levels.entries()].sort((a, b) => a[0] - b[0])
    const maxNodes = Math.max(1, ...sorted.map(([, nodes]) => nodes.length))
    const width = Math.max(800, sorted.length * 250 + 120)
    const height = Math.max(360, maxNodes * 108 + 100)
    const points = new Map()
    sorted.forEach(([, nodes], levelIndex) => {
      nodes.forEach((node, index) => {
        points.set(node.id, {
          x: 62 + levelIndex * ((width - 124) / Math.max(1, sorted.length - 1)),
          y: ((index + 1) * height) / (nodes.length + 1),
        })
      })
    })
    return { nodes: graph.nodes, edges: graph.edges, points, width, height, byId }
  }, [graph])

  if (!positions.nodes.length) return <div className="chart-empty">Graph appears after optimization.</div>
  return (
    <div className="dag-wrap">
      <div className="dag-title"><strong>{title}</strong><span>{positions.nodes.length} nodes · {positions.edges.length} dependencies · zoom {Math.round(zoom * 100)}%</span><div className="graph-zoom"><button aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(.55, value - .15))}>−</button><button aria-label="Reset zoom" onClick={() => setZoom(1)}>Reset</button><button aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(2.2, value + .15))}>+</button></div></div>
      <div className="dag-canvas"><svg className="dag-svg" width={positions.width * zoom} height={positions.height * zoom} viewBox={`0 0 ${positions.width} ${positions.height}`} role="img" aria-label={title}>
        <defs><marker id={optimized ? 'arrowOpt' : 'arrowOrig'} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#65708b" /></marker></defs>
        {positions.edges.map((edge, index) => {
          const source = positions.points.get(edge.source)
          const target = positions.points.get(edge.target)
          if (!source || !target) return null
          return <path key={`${edge.source}-${edge.target}-${index}`} d={`M ${source.x + 66} ${source.y} C ${source.x + 100} ${source.y}, ${target.x - 100} ${target.y}, ${target.x - 66} ${target.y}`} className="dag-edge" markerEnd={`url(#${optimized ? 'arrowOpt' : 'arrowOrig'})`} />
        })}
        {positions.nodes.map((node) => {
          const point = positions.points.get(node.id)
          const fused = node.op_type?.startsWith('Fused') || node.op_type?.startsWith('Linear') || node.op_type === 'GEMM'
          return <g key={node.id} transform={`translate(${point.x},${point.y})`}>
            <rect x="-66" y="-27" width="132" height="54" rx="11" className={`dag-node ${fused ? 'fused' : ''}`} />
            <text textAnchor="middle" y="4" className="dag-node-text">{node.op_type || 'Operation'}</text>
            <title>{node.label} · {node.op_type}{node.schedule_level != null ? ` · level ${node.schedule_level}` : ''}</title>
          </g>
        })}
      </svg></div>
    </div>
  )
}

function App() {
  const [models, setModels] = useState([])
  const [modelId, setModelId] = useState('postln_2b')
  const [batch, setBatch] = useState(4)
  const [seq, setSeq] = useState(64)
  const [dModel, setDModel] = useState(64)
  const [activeTab, setActiveTab] = useState('overview')
  const [health, setHealth] = useState('checking')
  const [result, setResult] = useState(null)
  const [xla, setXla] = useState(null)
  const [sweep, setSweep] = useState(null)
  const [report, setReport] = useState(null)
  const [error, setError] = useState('')
  const [xlaError, setXlaError] = useState('')
  const [loading, setLoading] = useState(false)
  const [sweeping, setSweeping] = useState(false)
  const [passes] = useState({ normalize: true, merge: true })

  useEffect(() => {
    api('status').then(() => setHealth('online')).catch(() => setHealth('offline'))
    api('models').then((data) => setModels(data.models || [])).catch(() => {})
  }, [])

  const workload = useMemo(() => ({ batch: Number(batch), seq: Number(seq), d_model: Number(dModel) }), [batch, seq, dModel])
  const metrics = result?.metrics
  const counts = metrics?.counts
  const speedup = metrics?.speedup
  const graphSteps = result?.stages || []
  const latencyRows = xla?.backends?.map((backend, index) => ({
    label: backend.name,
    value: backend.latency_ms,
    tone: index === 2 ? 'tone-cyan' : index === 1 ? 'tone-purple' : '',
    detail: `${backend.throughput_tokens_s.toLocaleString(undefined, { maximumFractionDigits: 0 })} tokens/s`,
  })) || []
  const throughputRows = xla?.backends?.map((backend, index) => ({
    label: backend.name,
    value: backend.throughput_tokens_s,
    tone: index === 2 ? 'tone-cyan' : index === 1 ? 'tone-purple' : '',
    detail: `${backend.latency_ms.toFixed(3)} ms median`,
  })) || []

  async function runOptimization() {
    setLoading(true)
    setError('')
    setXlaError('')
    setXla(null)
    setSweep(null)
    setReport(null)
    setActiveTab('overview')
    const payload = {
      model_id: modelId,
      ...workload,
      passes: { ...passes, partition: false },
      partition_mode: 'Off',
    }
    try {
      const response = await api('optimize', payload)
      setResult(response)
      setActiveTab('graphs')
      let backendResult = null
      try {
        backendResult = await api('xla_benchmark', { ...workload, warmup: 4, runs: 20 })
        setXla(backendResult)
      } catch (backendError) {
        setXlaError(backendError.message)
      }
      try {
        const saved = await api('export_report', {
          kind: 'attention_pytorch_xla_comparison', model_id: 'AttentionModel', device: 'CPU', workload,
          warmup_runs: 4, timed_runs: 20, compiler_metrics: response.metrics, stages: response.stages,
          pass_info: response.pass_info, xla: backendResult,
        })
        setReport(saved)
      } catch (exportError) {
        setError(`Comparison finished, but Excel export failed: ${exportError.message}`)
      }
      setActiveTab('overview')
    } catch (runError) {
      setError(runError.message)
    } finally {
      setLoading(false)
    }
  }

  async function runSweep(full = false) {
    setSweeping(true)
    setError('')
    setReport(null)
    try {
      const response = await api('benchmark_sweep', {
        ...(full ? {} : { cases: quickCases.map(([b, s, d]) => ({ batch: b, seq: s, d_model: d })) }),
        warmup: 2,
        runs: 8,
      })
      setSweep(response)
      const saved = await api('export_report', {
        kind: 'attention_workload_sweep', model_id: 'AttentionModel', device: 'CPU',
        warmup_runs: response.warmup_runs, timed_runs: response.timed_runs, sweep: response,
      })
      setReport(saved)
      setActiveTab('sweep')
    } catch (runError) {
      setError(runError.message)
    } finally {
      setSweeping(false)
    }
  }

  const passRows = Object.entries(metrics?.pass_durations_s || {}).map(([label, value]) => ({ label: label.replaceAll('_', ' '), value: value * 1000, detail: `${(value * 1000).toFixed(2)} ms` }))
  const memoryRows = metrics ? [
    { label: 'Measured · original', value: metrics.peak_memory_mb.original, detail: 'NumPy reference executor' },
    { label: 'Measured · optimized', value: metrics.peak_memory_mb.optimized, tone: 'tone-purple', detail: 'NumPy reference executor' },
    { label: 'Modeled GPU · original', value: metrics.modeled.memory_original_mb, tone: 'tone-cyan', detail: 'Cost model estimate' },
    { label: 'Modeled GPU · optimized', value: metrics.modeled.memory_optimized_mb, tone: 'tone-green', detail: 'Cost model estimate' },
  ] : []

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top"><span className="brand-mark"><Zap size={19} fill="currentColor" /></span><span>Graph<span className="brand-light">Forge</span><small>LLM COMPILER LAB</small></span></a>
        <div className="topbar-right"><span className={`api-pill ${health}`}><span />API {health}</span><span className="topbar-divider" /><span className="topbar-caption"><Cpu size={14} /> Research dashboard</span></div>
      </header>

      <main className="page" id="top">
        <section className="hero-panel">
          <div className="hero-copy"><div className="hero-kicker"><span className="pulse-dot" /> COMPILER WORKBENCH <span className="hero-version">v2.1</span></div>
            <h1>Make every operation<br /><span>earn its place.</span></h1>
            <p>Build a model’s computation DAG, normalize it, merge recognized operators, and compare the same workload with XLA.</p>
            <div className="hero-chips"><span><Workflow size={14} /> DAG construction</span><span><Layers3 size={14} /> Graph normalization</span><span><ChartNoAxesColumn size={14} /> Semantic merge</span><span><Cpu size={14} /> XLA benchmark</span></div>
          </div>
          <div className="hero-art" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit orbit-three" /><div className="hero-core"><Zap size={34} /></div><span className="orbit-node node-a">matmul</span><span className="orbit-node node-b">softmax</span><span className="orbit-node node-c">fused</span><span className="orbit-node node-d">normalize</span><div className="hero-art-caption">GRAPH → OPTIMIZE → MEASURE</div></div>
        </section>

        <section className="control-row panel">
          <div className="control-title"><span className="panel-icon"><SlidersHorizontal size={16} /></span><div><div className="eyebrow">EXPERIMENT SETUP</div><h2>Choose a workload</h2></div></div>
          <label className="field model-field"><span>Model graph</span><span className="select-wrap"><select value={modelId} onChange={(event) => setModelId(event.target.value)}>{models.map((model) => <option value={model.id} key={model.id}>{model.name}</option>)}</select><ChevronDown size={15} /></span></label>
          <label className="field"><span>Batch</span><input type="number" min="1" max="32" value={batch} onChange={(event) => setBatch(event.target.value)} /></label>
          <label className="field"><span>Sequence</span><input type="number" min="8" max="512" step="8" value={seq} onChange={(event) => setSeq(event.target.value)} /></label>
          <label className="field"><span>d model</span><input type="number" min="16" max="512" step="16" value={dModel} onChange={(event) => setDModel(event.target.value)} /></label>
          <button className="btn-primary run-btn" onClick={runOptimization} disabled={loading || health === 'offline'}>{loading ? <RefreshCw size={17} className="spin" /> : <Play size={16} fill="currentColor" />}{loading ? 'Running…' : 'Run comparison'}</button>
        </section>

        <section className="passes-row"><span className="passes-label">ACTIVE PASSES</span><span className="pass-chip selected"><span className="pass-check"><Check size={11} /></span>DAG construction</span><span className="pass-chip selected"><span className="pass-check"><Check size={11} /></span>Graph normalization</span><span className="pass-chip selected"><span className="pass-check"><Check size={11} /></span>Semantic operator merging</span><span className="passes-note">Same workload · same weights · CPU reference</span></section>

        {health === 'offline' && <div className="notice notice-error"><CircleAlert size={17} /> Backend API is offline. Start the Python API and React development server with <code>run-server.bat</code>.</div>}
        {error && <div className="notice notice-error"><CircleAlert size={17} />{error}</div>}

        {result && <>
          <nav className="tabs-nav" aria-label="Dashboard sections">
            {[
              ['overview', 'Overview', Gauge], ['graphs', 'Graph transforms', GitBranch], ['xla', 'XLA comparison', Cpu], ['sweep', 'Stress sweep', Activity],
            ].map(([id, label, Icon]) => <button key={id} className={`tab-btn ${activeTab === id ? 'active' : ''}`} onClick={() => setActiveTab(id)}><Icon size={16} />{label}{id === 'xla' && xla?.all_correct && <span className="tab-check"><Check size={11} /></span>}</button>)}
            {report && <a className="download-btn" href={report.download_url}><Download size={15} /> Download XLSX</a>}
          </nav>

          {activeTab === 'overview' && <div className="dashboard animate-fade">
            <div className="result-strip"><span className="result-dot" /> LAST RUN <b>{result.original_graph?.nodes?.length || counts?.nodes_original || 0} → {counts?.nodes_optimized || 0} IR nodes</b><span className="result-sep">·</span>batch {workload.batch} / seq {workload.seq} / d {workload.d_model}<span className="result-spacer" />{metrics.accuracy.preserved ? <span className="accuracy-tag"><Check size={13} /> outputs preserved</span> : <span className="accuracy-tag bad"><CircleAlert size={13} /> output changed</span>}</div>
            <div className="metrics-grid">
              <MetricCard label="Graph reduction" value={`${(metrics.graph_reduction_ratio * 100).toFixed(1)}%`} foot={`${counts.nodes_original} → ${counts.nodes_optimized} nodes`} icon={GitBranch} />
              <MetricCard label="Measured speedup" value={`${speedup.toFixed(2)}×`} foot="NumPy reference executor" icon={Gauge} tone="purple" />
              <MetricCard label="Accuracy preserved" value={metrics.accuracy.preserved ? '100%' : 'Mismatch'} foot={`max error ${metrics.accuracy.max_abs_diff.toExponential(1)}`} icon={Check} tone={metrics.accuracy.preserved ? 'green' : 'amber'} />
              <MetricCard label="Kernel reduction" value={`${(metrics.kernel_launch_reduction * 100).toFixed(1)}%`} foot={`${counts.kernel_launches_original} → ${counts.kernel_launches_optimized} launches`} icon={Boxes} tone="cyan" />
              <MetricCard label="Modeled GPU speedup" value={`${metrics.modeled.speedup.toFixed(2)}×`} foot="Cost model estimate · not measured GPU" icon={ArrowUpRight} tone="amber" />
              <MetricCard label="Compile time" value={`${(metrics.compilation_time_s * 1000).toFixed(1)} ms`} foot={`${Object.keys(metrics.pass_durations_s || {}).length} passes`} icon={Activity} />
            </div>
            <div className="charts-grid">
              <Panel title="Graph size through the pipeline" eyebrow="NORMALIZATION + FUSION" icon={GitBranch}><BarChart rows={graphSteps.map((step, index) => ({ label: step.stage, value: step.ops, tone: index === graphSteps.length - 1 ? 'tone-purple' : '', detail: `${step.ops} operations` }))} unit=" ops" precision={0} /></Panel>
              <Panel title="Latency comparison" eyebrow="REFERENCE EXECUTOR · MS" icon={Gauge}><BarChart rows={[
                { label: 'Original graph', value: metrics.latency_ms.original, detail: 'NumPy interpreter' },
                { label: 'Optimized graph', value: metrics.latency_ms.optimized, tone: 'tone-purple', detail: `${speedup.toFixed(2)}× measured speedup` },
                { label: 'Modeled GPU · original', value: metrics.modeled.latency_original_us / 1000, tone: 'tone-cyan', detail: 'Cost model only' },
                { label: 'Modeled GPU · optimized', value: metrics.modeled.latency_optimized_us / 1000, tone: 'tone-green', detail: `${metrics.modeled.speedup.toFixed(2)}× modeled` },
              ]} unit=" ms" /></Panel>
              <Panel title="Throughput" eyebrow="SAMPLES PER SECOND" icon={Activity}><BarChart rows={[
                { label: 'Original graph', value: metrics.throughput.original, detail: `batch ${metrics.throughput.batch}` },
                { label: 'Optimized graph', value: metrics.throughput.optimized, tone: 'tone-purple', detail: `${((metrics.throughput.optimized / metrics.throughput.original - 1) * 100).toFixed(1)}% change` },
              ]} unit=" /s" precision={0} /></Panel>
              <Panel title="Memory footprint" eyebrow="MEASURED VS MODELED · MB" icon={Boxes}><BarChart rows={memoryRows} unit=" MB" /></Panel>
              <Panel title="Optimization ratios" eyebrow="STRUCTURAL METRICS" icon={ChartNoAxesColumn}><BarChart rows={[
                { label: 'Graph reduction · GRR', value: metrics.graph_reduction_ratio * 100, detail: `${counts.nodes_original - counts.nodes_optimized} nodes removed` },
                { label: 'Operator merge · OMR', value: metrics.operator_merge_ratio * 100, tone: 'tone-purple' },
                { label: 'Attention merge · ACR', value: (metrics.attention_canonicalization_rate || 0) * 100, tone: 'tone-cyan', detail: `${metrics.attention_subgraphs.canonicalized}/${metrics.attention_subgraphs.detected} patterns merged inside semantic merge` },
                { label: 'Kernel launch reduction', value: metrics.kernel_launch_reduction * 100, tone: 'tone-green' },
              ]} unit="%" /></Panel>
              <Panel title="Pass time" eyebrow="COMPILATION BREAKDOWN · MS" icon={Activity}><BarChart rows={passRows} unit=" ms" /></Panel>
            </div>
          </div>}

          {activeTab === 'graphs' && <div className="charts-grid graph-grid animate-fade">
            <Panel title="Original computation DAG" eyebrow="BEFORE PASSES" icon={Workflow}><GraphView graph={result.original_graph} title="Original IR" /></Panel>
            <Panel title="Optimized computation DAG" eyebrow="AFTER SELECTED PASSES" icon={Workflow}><GraphView graph={result.optimized_graph} title="Optimized IR" optimized /></Panel>
            <Panel title="Pass-by-pass node count" eyebrow="NORMALIZATION OUTPUT INCLUDED" icon={Layers3}><BarChart rows={graphSteps.map((step, index) => ({ label: step.stage, value: step.ops, tone: index === graphSteps.length - 1 ? 'tone-purple' : '' }))} unit=" ops" precision={0} /></Panel>
          </div>}

          {activeTab === 'xla' && <div className="dashboard animate-fade">
            <div className="section-intro"><div><div className="eyebrow">SAME INPUT · SAME WEIGHTS · FLOAT32 · CPU</div><h2>How does XLA compile this attention graph?</h2><p>JAX lowers this attention function through OpenXLA. Compilation time is separated from steady-state execution time.</p></div><span className={`benchmark-state ${xla?.all_correct ? 'good' : ''}`}>{xla?.all_correct ? <><Check size={15} /> Numerical outputs match</> : <><Cpu size={15} /> XLA backend status</>}</span></div>
            {xla ? <>
              <div className="metric-note"><Braces size={15} /> FX node counts and StableHLO operation counts describe different IRs; compare latency and correctness across backends, not those counts as if they were equivalent.</div>
              <div className="metrics-grid xla-cards">{xla.backends.map((backend, index) => <MetricCard key={backend.name} label={backend.name} value={`${backend.latency_ms.toFixed(3)} ms`} foot={`${backend.throughput_tokens_s.toLocaleString(undefined, { maximumFractionDigits: 0 })} tokens/s${backend.compile_ms ? ` · compile ${backend.compile_ms.toFixed(1)} ms` : ''}`} icon={index === 2 ? Cpu : index === 1 ? Layers3 : Activity} tone={index === 2 ? 'cyan' : index === 1 ? 'purple' : ''} />)}</div>
              <div className="charts-grid">
                <Panel title="Per-call latency" eyebrow="MEDIAN AFTER WARMUP · MS" icon={Gauge}><BarChart rows={latencyRows} unit=" ms" precision={3} /></Panel>
                <Panel title="Token throughput" eyebrow="TOKENS PER SECOND" icon={Activity}><BarChart rows={throughputRows} unit=" tok/s" precision={0} /></Panel>
                <Panel title="PyTorch FX graph size" eyebrow="FX COMPUTE NODES" icon={Braces}><BarChart rows={[
                  { label: 'Before normalization', value: xla.normalization.fx_nodes_before },
                  { label: 'After normalization', value: xla.normalization.fx_nodes_after, tone: 'tone-cyan', detail: `${xla.normalization.nodes_removed} identity/dead nodes removed` },
                  { label: 'After semantic fusion', value: xla.backends[1].fx_compute_ops, tone: 'tone-purple', detail: `${xla.backends[1].graph_reduction_pct.toFixed(1)}% fewer than normalized FX` },
                ]} unit=" ops" precision={0} /></Panel>
                <Panel title="XLA StableHLO graph" eyebrow="XLA LOWERED IR · DISTINCT COUNT" icon={Braces}><BarChart rows={[
                  { label: 'StableHLO operations', value: xla.backends[2].stablehlo_ops, tone: 'tone-cyan', detail: 'Operation count in XLA lowered module' },
                ]} unit=" ops" precision={0} /></Panel>
                <Panel title="Output agreement" eyebrow="MAX ABSOLUTE ERROR VS PYTORCH" icon={Check}><BarChart rows={xla.backends.map((backend, index) => ({ label: backend.name, value: backend.max_abs_error_vs_torch, tone: index === 2 ? 'tone-cyan' : index === 1 ? 'tone-purple' : '', detail: backend.correct_vs_torch ? 'Within tolerance' : 'Outside tolerance' }))} precision={7} /></Panel>
              </div>
              <div className="metric-note"><Cpu size={15} /> Device: {xla.backends[2].device}. {xla.comparison_note} {xla.warmup_runs} warmup · {xla.timed_runs} measured calls.</div>
              <details className="panel ir-details"><summary>Inspect XLA StableHLO output</summary><pre>{xla.xla_ir}</pre></details>
            </> : <div className="panel xla-empty"><div className="empty-icon"><Cpu size={22} /></div><h3>XLA comparison unavailable</h3><p>{xlaError || 'Run the comparison after enabling the JAX CPU dependency.'}</p><code>pip install -r requirements.txt</code></div>}
          </div>}

          {activeTab === 'sweep' && <div className="dashboard animate-fade">
            <div className="section-intro"><div><div className="eyebrow">MULTI-SHAPE ATTENTION STRESS TEST</div><h2>See how the rewrite behaves as workloads grow</h2><p>Same weights and input for each eager-versus-semantic pair, with output error reported per case.</p></div><div className="sweep-actions"><button className="btn-secondary" onClick={() => runSweep(false)} disabled={sweeping}>{sweeping ? <RefreshCw size={15} className="spin" /> : <Activity size={15} />}{sweeping ? 'Sweeping…' : 'Quick · 4 shapes'}</button><button className="btn-secondary" onClick={() => runSweep(true)} disabled={sweeping}>{sweeping ? <RefreshCw size={15} className="spin" /> : <ChartNoAxesColumn size={15} />}{sweeping ? 'Sweeping…' : 'All 13 shapes'}</button></div></div>
            {sweep ? <>
              <div className="metric-note"><Activity size={15} /> {sweep.results.length} workload shapes · {sweep.warmup_runs} warmup · {sweep.timed_runs} timed runs per graph.</div>
              <div className="metrics-grid sweep-summary"><MetricCard label="Median speedup" value={`${sweep.summary.median_speedup.toFixed(2)}×`} foot="Across workload shapes" icon={Gauge} tone="purple" /><MetricCard label="Average graph reduction" value={`${sweep.summary.avg_grr.toFixed(1)}%`} foot="FX compute nodes" icon={GitBranch} /><MetricCard label="Numerical correctness" value={sweep.summary.all_correct ? 'All match' : 'Mismatch'} foot={`${sweep.results.filter((row) => row.correct).length}/${sweep.results.length} workloads`} icon={Check} tone="green" /></div>
              <div className="charts-grid"><Panel title="Speedup by workload" eyebrow="EAGER / SEMANTIC LATENCY" icon={ArrowUpRight}><BarChart rows={sweep.results.map((row) => ({ label: `B${row.batch} · S${row.seq} · D${row.d_model}`, value: row.speedup, tone: row.speedup >= 1 ? 'tone-green' : 'tone-rose', detail: `${row.original_latency_ms.toFixed(3)} → ${row.optimized_latency_ms.toFixed(3)} ms` }))} unit="×" precision={3} /></Panel><Panel title="Latency across workload shapes" eyebrow="MILLISECONDS · LOWER IS BETTER" icon={Gauge}><BarChart rows={sweep.results.flatMap((row) => [
                { label: `Original · S${row.seq}/D${row.d_model}`, value: row.original_latency_ms },
                { label: `Semantic · S${row.seq}/D${row.d_model}`, value: row.optimized_latency_ms, tone: 'tone-purple' },
              ])} unit=" ms" precision={3} /></Panel><Panel title="Graph reduction" eyebrow="FX COMPUTE OPS" icon={Boxes}><BarChart rows={sweep.results.map((row) => ({ label: `S${row.seq} · D${row.d_model}`, value: row.grr, tone: 'tone-cyan' }))} unit="%" /></Panel></div>
              <div className="panel table-panel"><div className="panel-head"><div className="panel-heading"><span className="panel-icon"><ChartNoAxesColumn size={16} /></span><div><div className="eyebrow">STRESS TEST DATA</div><h2>Workload results</h2></div></div></div><div className="table-scroll"><table className="data-table"><thead><tr><th>Batch</th><th>Sequence</th><th>d model</th><th>Latency · original</th><th>Latency · semantic</th><th>Speedup</th><th>Reduction</th><th>Max error</th><th>Correct</th></tr></thead><tbody>{sweep.results.map((row, index) => <tr key={`${row.batch}-${row.seq}-${row.d_model}-${index}`}><td>{row.batch}</td><td>{row.seq}</td><td>{row.d_model}</td><td>{row.original_latency_ms.toFixed(3)} ms</td><td>{row.optimized_latency_ms.toFixed(3)} ms</td><td>{row.speedup.toFixed(2)}×</td><td>{row.grr.toFixed(1)}%</td><td>{row.max_error.toExponential(2)}</td><td className={row.correct ? 'correct-cell' : 'incorrect-cell'}>{row.correct ? 'PASS' : 'CHECK'}</td></tr>)}</tbody></table></div></div>
            </> : <div className="panel xla-empty"><div className="empty-icon"><Activity size={22} /></div><h3>Run an attention workload sweep</h3><p>Choose four representative workloads for a quick check or use all 13 sizes from the supplied stress-test matrix.</p><div className="sweep-actions centered"><button className="btn-primary" onClick={() => runSweep(false)} disabled={sweeping}>{sweeping ? <RefreshCw size={15} className="spin" /> : <Play size={15} />}{sweeping ? 'Sweeping…' : 'Quick · 4 shapes'}</button><button className="btn-secondary" onClick={() => runSweep(true)} disabled={sweeping}>{sweeping ? <RefreshCw size={15} className="spin" /> : <ChartNoAxesColumn size={15} />}{sweeping ? 'Sweeping…' : 'All 13 shapes'}</button></div></div>}
          </div>}
        </>}

        {!result && <section className="empty-workspace panel"><span className="empty-icon"><Workflow size={24} /></span><div className="eyebrow">READY WHEN YOU ARE</div><h2>Choose a model and run the compiler</h2><p>The dashboard will show graph normalization, semantic merging, before/after DAGs, and an XLA comparison for the attention workload.</p><button className="btn-primary" onClick={runOptimization} disabled={loading || health === 'offline'}>{loading ? <RefreshCw size={17} className="spin" /> : <Play size={16} fill="currentColor" />}{loading ? 'Running comparison…' : 'Run first comparison'}</button></section>}

        <footer className="footer"><span><Zap size={13} /> GraphForge · compiler research UI</span><span>Reference executor metrics and hardware/compiler benchmarks are reported separately.</span></footer>
      </main>
    </div>
  )
}

export default App

import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [outputPath] = process.argv.slice(2);
if (!outputPath) throw new Error("Usage: export_metrics.mjs <output.xlsx>");

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const report = JSON.parse(Buffer.concat(chunks).toString("utf8"));

function flatten(value, prefix = "", rows = []) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const [key, child] of Object.entries(value)) {
      flatten(child, prefix ? `${prefix}.${key}` : key, rows);
    }
  } else if (Array.isArray(value)) {
    rows.push([prefix, JSON.stringify(value)]);
  } else {
    const metricLabels = {
      "compiler.graph_reduction_ratio": ["GRR · Graph reduction (%)", 100],
      "compiler.operator_merge_ratio": ["OMR · Operator merge (%)", 100],
      "compiler.attention_canonicalization_rate": ["ACR · Attention patterns merged (%)", 100],
      "compiler.kernel_launch_reduction": ["Estimated kernel-launch reduction (%)", 100],
      "compiler.modeled.memory_reduction_pct": ["Modeled memory reduction (%)", 100],
      "compiler.compilation_time_s": ["Pipeline compilation time (ms)", 1000],
    };
    if (metricLabels[prefix] && typeof value === "number") {
      const [label, multiplier] = metricLabels[prefix];
      rows.push([label, value * multiplier]);
    } else if (prefix.startsWith("compiler.pass_durations_s.") && typeof value === "number") {
      rows.push([`Pass duration · ${prefix.split(".").at(-1)} (ms)`, value * 1000]);
    } else {
      rows.push([prefix, value ?? "N/A - not implemented in this run"]);
    }
  }
  return rows;
}

function writeSheet(workbook, name, title, rows) {
  const sheet = workbook.worksheets.add(name);
  const values = [[title, ""], ["Metric", "Value"], ...rows];
  sheet.getRange(`A1:B${values.length}`).values = values;
  sheet.getRange("A1:B1").merge();
  sheet.getRange("A1:B1").format = {
    fill: "#1E293B",
    font: { bold: true, color: "#FFFFFF", size: 16 },
    rowHeight: 30,
    verticalAlignment: "center",
  };
  sheet.getRange("A2:B2").format = {
    fill: "#DBEAFE",
    font: { bold: true, color: "#1E3A8A" },
  };
  if (values.length > 2) {
    sheet.getRange(`A3:B${values.length}`).format.wrapText = true;
    sheet.getRange(`A3:A${values.length}`).format.columnWidth = 42;
    sheet.getRange(`B3:B${values.length}`).format.columnWidth = 72;
  }
  sheet.freezePanes.freezeRows(2);
  return sheet;
}

const workbook = Workbook.create();
const overviewRows = [
  ["Run ID", report.run_id],
  ["Run type", report.kind],
  ["Created (UTC)", report.created_utc ? `${report.created_utc} UTC` : ""],
  ["Model", report.model_id ?? "AttentionModel"],
  ["Device", report.device ?? "CPU"],
  ["Batch", report.workload?.batch ?? ""],
  ["Sequence length", report.workload?.seq ?? ""],
  ["Model width", report.workload?.d_model ?? ""],
  ["Warmup runs", report.warmup_runs ?? ""],
  ["Timed runs", report.timed_runs ?? ""],
  ...flatten(report.compiler_metrics ?? {}, "compiler"),
];
writeSheet(workbook, "Summary", "LLM graph optimization run", overviewRows);

if (report.xla) {
  const sheet = workbook.worksheets.add("XLA");
  const headers = [
    "Backend", "Median latency ms", "Throughput tokens/s", "Compile ms",
    "Max abs error", "Output matches", "FX compute ops", "Graph reduction %",
    "StableHLO ops", "Device",
  ];
  const backendRows = (report.xla.backends ?? []).map((backend) => [
    backend.name, backend.latency_ms, backend.throughput_tokens_s,
    backend.compile_ms ?? "N/A", backend.max_abs_error_vs_torch,
    backend.correct_vs_torch, backend.fx_compute_ops ?? "N/A",
    backend.graph_reduction_pct ?? "N/A", backend.stablehlo_ops ?? "N/A",
    backend.device ?? report.device ?? "CPU",
  ]);
  const comparisonRows = [
    ["Comparison note", report.xla.comparison_note ?? ""],
    ["Warmup runs", report.xla.warmup_runs ?? ""],
    ["Timed runs", report.xla.timed_runs ?? ""],
    ["FX nodes before normalization", report.xla.normalization?.fx_nodes_before ?? ""],
    ["FX nodes after normalization", report.xla.normalization?.fx_nodes_after ?? ""],
    ["Normalization nodes removed", report.xla.normalization?.nodes_removed ?? ""],
    ["IR operation counts directly comparable", "No — FX and StableHLO are different IRs"],
  ];
  sheet.getRange(`A1:J${backendRows.length + 2}`).values = [
    ["Same-input XLA comparison"], headers, ...backendRows,
  ];
  sheet.getRange("A1:J1").merge();
  sheet.getRange("A1:J1").format = {
    fill: "#1E293B", font: { bold: true, color: "#FFFFFF", size: 16 },
    rowHeight: 30, verticalAlignment: "center",
  };
  sheet.getRange("A2:J2").format = {
    fill: "#DBEAFE", font: { bold: true, color: "#1E3A8A" }, wrapText: true,
  };
  if (backendRows.length) {
    sheet.getRange(`A3:J${backendRows.length + 2}`).format.wrapText = true;
  }
  sheet.getRange("A1:A100").format.columnWidth = 30;
  sheet.getRange("B1:J100").format.columnWidth = 18;
  const contextStart = backendRows.length + 5;
  sheet.getRange(`A${contextStart}:B${contextStart + comparisonRows.length - 1}`).values = comparisonRows;
  sheet.getRange(`A${contextStart}:B${contextStart + comparisonRows.length - 1}`).format.wrapText = true;
  sheet.getRange(`B${contextStart}:J${contextStart}`).merge();
  sheet.freezePanes.freezeRows(2);
}

if (report.sweep?.results?.length) {
  const sheet = workbook.worksheets.add("Workload sweep");
  const headers = [
    "Batch", "Sequence", "d_model", "Original nodes", "Normalized nodes",
    "Optimized nodes", "Graph reduction %", "Original latency ms",
    "Optimized latency ms", "Speedup x", "Original throughput tokens/s",
    "Optimized throughput tokens/s", "OMR %", "ACR % (N/A)",
    "Original kernel launches (estimated)", "Optimized kernel launches (estimated)",
    "Kernel launch reduction % (estimated)", "Peak GPU memory MB (N/A on CPU)",
    "Compilation ms", "Max absolute error", "Correct",
  ];
  const rows = report.sweep.results.map((item) => [
    item.batch, item.seq, item.d_model, item.original_nodes, item.normalized_nodes,
    item.optimized_nodes, item.grr, item.original_latency_ms,
    item.optimized_latency_ms, item.speedup, item.throughput_orig,
    item.throughput_opt, item.omr, item.acr ?? "N/A - not implemented",
    item.kernel_launches_original_estimate, item.kernel_launches_optimized_estimate,
    item.kernel_launch_reduction_estimate, "N/A - CPU benchmark",
    item.compilation_time_ms, item.max_error, item.correct,
  ]);
  const table = [headers, ...rows];
  sheet.getRange(`A1:U${table.length}`).values = table;
  sheet.getRange("A1:U1").format = {
    fill: "#1E293B", font: { bold: true, color: "#FFFFFF" },
    wrapText: true, rowHeight: 34,
  };
  sheet.getRange(`A2:U${table.length}`).format.wrapText = true;
  sheet.getRange(`A1:U${table.length}`).format.columnWidth = 17;
  sheet.freezePanes.freezeRows(1);
}

if (report.stages?.length) {
  writeSheet(workbook, "Graph stages", "DAG node counts by pass", [
    ...report.stages.map((stage) => [stage.stage, stage.ops]),
  ]);
}

await workbook.recalculate();
await fs.mkdir(path.dirname(path.resolve(outputPath)), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, sheets: workbook.worksheets.items.map((sheet) => sheet.name) }));

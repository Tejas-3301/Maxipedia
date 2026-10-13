import { fmt, short } from "../util.js";

const W = 1000;
const FX = 250;
const MX = 600;
const VX = 880;
const ROW = 22;
const TOP = 34;

const PAIR_STYLE = {
  corroborates: { stroke: "#7c3aed", dash: undefined },
  conflicts: { stroke: "#ea580c", dash: "5 3" },
  redundant_with: { stroke: "#94a3b8", dash: "2 3" },
};

/**
 * Evidence graph: finding -> modality -> verdict with supports (red) / contradicts (green) edges sized by their
 * log-odds weight, plus corroborates / conflicts / redundant_with arcs between findings. Click a finding's dot to
 * remove it (counterfactual); click its label to open its evidence.
 */
export default function GraphView({ graph, cf, excluded, onToggle, onSelect, selectedId, reviews, onUseRejections, onReset }) {
  if (!graph) return null;
  const evidence = graph.nodes
    .filter((n) => n.type === "finding" && n.role === "evidence")
    .sort((a, b) => a.modality.localeCompare(b.modality) || b.llr - a.llr);
  const mods = graph.nodes.filter((n) => n.type === "modality");
  const verdict = graph.nodes.find((n) => n.type === "verdict");
  const context = graph.nodes.filter((n) => n.type === "finding" && n.role === "context").length;
  const H = Math.max(TOP + evidence.length * ROW + 20, TOP + mods.length * 60 + 20);
  const fy = Object.fromEntries(evidence.map((n, i) => [n.id, TOP + i * ROW + 10]));
  const my = Object.fromEntries(mods.map((n, i) => [n.id, TOP + 30 + i * ((H - TOP - 60) / Math.max(mods.length - 1, 1))]));
  const vy = H / 2;
  const after = cf?.assessment;
  const afterMod = after ? Object.fromEntries(Object.entries(after.modalities).map(([m, v]) => [`mod:${m}`, v])) : {};
  const rejected = Object.entries(reviews || {}).filter(([, r]) => r.decision === "reject").map(([id]) => id);

  const curve = (x1, y1, x2, y2) => `M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`;
  const arc = (y1, y2) => {
    const bow = 18 + Math.abs(y2 - y1) * 0.25;
    return `M ${FX + 6} ${y1} C ${FX + 6 + bow} ${y1}, ${FX + 6 + bow} ${y2}, ${FX + 6} ${y2}`;
  };

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-1 flex flex-wrap items-center gap-3 text-xs text-slate-600">
        <b className="text-slate-800">Evidence graph</b>
        <span><span className="text-red-600">━</span> supports</span>
        <span><span className="text-green-600">━</span> contradicts</span>
        <span><span className="text-violet-600">━</span> corroborates</span>
        <span><span className="text-orange-600">╍</span> conflicts</span>
        <span><span className="text-slate-400">┈</span> redundant (discounted)</span>
        <span>· {context} neutral / not-assessable findings carry no weight</span>
        <span className="ml-auto flex gap-2">
          <button disabled={!rejected.length} onClick={() => onUseRejections(rejected)}
                  className="rounded border border-slate-300 px-2 py-0.5 disabled:opacity-40">
            remove my {rejected.length} rejected
          </button>
          <button disabled={!excluded.size} onClick={onReset} className="rounded border border-slate-300 px-2 py-0.5 disabled:opacity-40">
            reset
          </button>
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <text x={FX} y={16} fontSize="10" textAnchor="end" fill="#64748b">evidence (dot = remove/restore)</text>
        <text x={MX} y={16} fontSize="10" textAnchor="middle" fill="#64748b">modality</text>
        <text x={VX} y={16} fontSize="10" textAnchor="middle" fill="#64748b">verdict</text>

        {graph.edges.filter((e) => PAIR_STYLE[e.relation] && fy[e.source] && fy[e.target]).map((e, i) => (
          <path key={`p${i}`} d={arc(fy[e.source], fy[e.target])} fill="none" stroke={PAIR_STYLE[e.relation].stroke}
                strokeDasharray={PAIR_STYLE[e.relation].dash} strokeWidth="1.2"
                opacity={excluded.has(e.source) || excluded.has(e.target) ? 0.15 : 0.7}>
            <title>{e.relation}</title>
          </path>
        ))}
        {graph.edges.filter((e) => fy[e.source] && e.target.startsWith("mod:")).map((e, i) => {
          const off = excluded.has(e.source);
          return (
            <path key={`f${i}`} d={curve(FX + 6, fy[e.source], MX - 52, my[e.target])} fill="none"
                  stroke={off ? "#cbd5e1" : e.relation === "supports" ? "#dc2626" : "#16a34a"}
                  strokeWidth={off ? 1 : 1 + Math.min(4, Math.abs(e.weight) * 1.4)} strokeDasharray={off ? "4 3" : undefined}
                  opacity={off ? 0.6 : 0.55}>
              <title>{`${e.relation} ${fmt(e.weight)}`}</title>
            </path>
          );
        })}
        {graph.edges.filter((e) => e.source.startsWith("mod:") && e.target === "verdict").map((e, i) => (
          <path key={`m${i}`} d={curve(MX + 52, my[e.source], VX - 70, vy)} fill="none"
                stroke={e.relation === "supports" ? "#dc2626" : "#16a34a"}
                strokeWidth={1 + Math.min(5, Math.abs(e.weight))} opacity="0.5">
            <title>{`${e.relation} ${fmt(e.weight)}`}</title>
          </path>
        ))}

        {evidence.map((n) => {
          const off = excluded.has(n.id);
          const rv = reviews?.[n.id]?.decision;
          return (
            <g key={n.id}>
              <text x={FX - 10} y={fy[n.id] + 4} fontSize="10.5" textAnchor="end" style={{ cursor: "pointer" }}
                    fill={off ? "#94a3b8" : "#0f172a"} fontWeight={n.id === selectedId ? 700 : 400}
                    textDecoration={off ? "line-through" : undefined} onClick={() => onSelect(n.id)}>
                {short(n.label)}{n.segment ? ` ${fmt(n.segment.start, 1)}–${fmt(n.segment.end, 1)}s` : ""}
                {rv ? ` [${rv}]` : ""}
                {n.impact ? `  ${n.impact.delta_p >= 0 ? "+" : ""}${fmt(n.impact.delta_p)}${n.impact.flips_verdict ? "✱" : ""}` : ""}
              </text>
              <circle cx={FX} cy={fy[n.id]} r={6} style={{ cursor: "pointer" }} onClick={() => onToggle(n.id)}
                      fill={off ? "#fff" : n.stance === "supports_manipulation" ? "#dc2626" : "#16a34a"}
                      stroke={n.id === selectedId ? "#0f172a" : "#475569"} strokeWidth={n.id === selectedId ? 2 : 1}>
                <title>{off ? "restore this evidence" : "remove this evidence (counterfactual)"}</title>
              </circle>
            </g>
          );
        })}

        {mods.map((n) => {
          const a = afterMod[n.id];
          const changed = a && Math.abs(a.posterior - n.posterior) > 0.0005;
          return (
            <g key={n.id}>
              <rect x={MX - 52} y={my[n.id] - 18} width={104} height={36} rx={6}
                    fill={n.assessed ? "#f8fafc" : "#ffffff"} stroke={n.conflict ? "#ea580c" : "#94a3b8"}
                    strokeDasharray={n.assessed ? undefined : "3 3"} />
              <text x={MX} y={my[n.id] - 3} fontSize="11" textAnchor="middle" fontWeight="600">{n.label}</text>
              <text x={MX} y={my[n.id] + 11} fontSize="9.5" textAnchor="middle" fill="#475569">
                P {fmt(n.posterior)}{changed ? ` → ${fmt(a.posterior)}` : ""}{n.assessed ? "" : " (prior)"}
              </text>
            </g>
          );
        })}

        <g>
          <rect x={VX - 70} y={vy - 34} width={140} height={after ? 68 : 46} rx={8}
                fill={verdict.label === "likely manipulated" ? "#fef2f2" : verdict.label === "likely authentic" ? "#ecfdf5" : "#fffbeb"}
                stroke="#475569" />
          <text x={VX} y={vy - 16} fontSize="11" textAnchor="middle" fontWeight="700">{verdict.label}</text>
          <text x={VX} y={vy} fontSize="10" textAnchor="middle">P {fmt(verdict.p_manipulated)}</text>
          {after && (
            <>
              <text x={VX} y={vy + 16} fontSize="9.5" textAnchor="middle" fill="#7c2d12">without {excluded.size} removed:</text>
              <text x={VX} y={vy + 29} fontSize="10.5" textAnchor="middle" fontWeight="700" fill="#7c2d12">
                {after.state} · P {fmt(after.probability_manipulated)}
              </text>
            </>
          )}
        </g>
      </svg>
    </div>
  );
}

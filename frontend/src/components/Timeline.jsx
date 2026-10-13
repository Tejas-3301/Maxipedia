import { LANES, STANCE_COLOR, fmt, short } from "../util.js";

const W = 1000;
const LABEL = 92;
const CHIPS = 150;
const LANE_H = 54;
const SCHED_H = 18;
const TOP = 16;

const LANE_CURVE = { visual: "video", audio: "audio" };

/**
 * Swimlanes: screening curve per lane first (grey area), deep findings fill in as segments
 * (red = supports manipulation, green = clean, grey hatched = not assessable). File-level findings sit
 * as chips on the right. Top row: deep-analysis schedule (analysed solid, unanalysed outlined).
 */
export default function Timeline({ duration, screening, schedule, findings, reviews, syncCurve, time, selectedId,
                                   onSelect, onSeek }) {
  const dur = Math.max(duration || 0, 0.1);
  const X0 = LABEL;
  const X1 = W - CHIPS - 8;
  const x = (t) => X0 + (Math.min(Math.max(t, 0), dur) / dur) * (X1 - X0);
  const laneY = (i) => TOP + SCHED_H + 6 + i * LANE_H;
  const H = laneY(LANES.length) + 18;

  const seekFromEvent = (e) => {
    const svg = e.currentTarget.ownerSVGElement || e.currentTarget;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    if (p.x >= X0 && p.x <= X1) onSeek(((p.x - X0) / (X1 - X0)) * dur);
  };

  const sched = schedule || {};
  const analysed = sched.analysed || [];
  const unanalysed = sched.unanalysed || [];
  const pending = !sched.analysed && sched.candidates ? sched.candidates : [];
  const ticks = [];
  const step = dur > 30 ? 5 : dur > 10 ? 2 : 1;
  for (let t = 0; t <= dur + 1e-6; t += step) ticks.push(t);

  const segFindings = (lane) =>
    findings.filter((f) => f.modality === lane && f.segment && !f.check.startsWith("screen."));
  const fileFindings = (lane) =>
    findings.filter((f) => f.modality === lane && !f.segment && f.check !== "ingest_inventory");

  const curvePath = (curve, i) => {
    if (!curve?.length) return null;
    const bin = screening.bin_s;
    const base = laneY(i) + LANE_H - 4;
    const h = LANE_H - 10;
    let d = `M ${x(0)} ${base}`;
    curve.forEach((v, k) => {
      d += ` L ${x(k * bin)} ${base - v * h} L ${x((k + 1) * bin)} ${base - v * h}`;
    });
    return `${d} L ${x(curve.length * bin)} ${base} Z`;
  };

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-1 flex flex-wrap items-center gap-3 text-xs text-slate-500">
        <b className="text-slate-700">Timeline</b>
        <Legend color="#94a3b8" label="screening curve" area />
        <Legend color={STANCE_COLOR.supports_manipulation} label="supports manipulation" />
        <Legend color={STANCE_COLOR.contradicts} label="clean (deep)" />
        <Legend color={STANCE_COLOR.not_assessable} label="not assessable" />
        <span>click a finding to jump · click a lane to seek</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none">
        <defs>
          <pattern id="hatch" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
            <line x1="0" y1="0" x2="0" y2="6" stroke="#94a3b8" strokeWidth="2" />
          </pattern>
        </defs>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={TOP} y2={H - 14} stroke="#f1f5f9" />
            <text x={x(t)} y={H - 3} fontSize="9" textAnchor="middle" fill="#64748b">
              {t}s
            </text>
          </g>
        ))}
        <text x={4} y={TOP + 12} fontSize="10" fill="#475569">
          deep
        </text>
        {pending.map((iv, k) => (
          <rect key={`p${k}`} x={x(iv.start)} y={TOP} width={Math.max(1, x(iv.end) - x(iv.start))} height={SCHED_H}
                fill="none" stroke="#cbd5e1" strokeDasharray="2 2" />
        ))}
        {analysed.map((iv, k) => (
          <rect key={`a${k}`} x={x(iv.start)} y={TOP} width={Math.max(1, x(iv.end) - x(iv.start))} height={SCHED_H}
                fill={iv.kind === "flagged" ? "#fde68a" : iv.kind === "manual" ? "#c4b5fd" : "#e2e8f0"} stroke="#fff">
            <title>{`${iv.kind} ${fmt(iv.start)}–${fmt(iv.end)}s analysed in depth`}</title>
          </rect>
        ))}
        {unanalysed.map((iv, k) => (
          <rect key={`u${k}`} x={x(iv.start)} y={TOP} width={Math.max(1, x(iv.end) - x(iv.start))} height={SCHED_H}
                fill="url(#hatch)" stroke={iv.kind === "flagged" ? "#dc2626" : "#94a3b8"}>
            <title>{`${iv.kind} ${fmt(iv.start)}–${fmt(iv.end)}s NOT analysed (deadline)`}</title>
          </rect>
        ))}

        {LANES.map((lane, i) => {
          const y = laneY(i);
          const curve = screening?.curves?.[LANE_CURVE[lane]];
          const chips = fileFindings(lane);
          return (
            <g key={lane}>
              <rect x={X0} y={y} width={X1 - X0} height={LANE_H - 4} fill={i % 2 ? "#f8fafc" : "#ffffff"}
                    stroke="#e2e8f0" onClick={seekFromEvent} style={{ cursor: "crosshair" }} />
              <text x={4} y={y + LANE_H / 2} fontSize="11" fontWeight="600" fill="#334155">
                {lane}
              </text>
              {curve && <path d={curvePath(curve, i)} fill="#94a3b8" opacity="0.35" pointerEvents="none" />}
              {curve && (
                <line x1={X0} x2={X1} y1={y + LANE_H - 4 - screening.flag_level * (LANE_H - 10)}
                      y2={y + LANE_H - 4 - screening.flag_level * (LANE_H - 10)} stroke="#94a3b8"
                      strokeDasharray="3 3" pointerEvents="none" />
              )}
              {lane === "sync" && syncCurve?.length > 1 && (
                <polyline pointerEvents="none" fill="none" stroke="#7c3aed" strokeWidth="1.5"
                          points={syncCurve.map((p) => `${x(p.t)},${y + LANE_H - 4 - Math.max(0, Math.min(1, p.r)) * (LANE_H - 10)}`).join(" ")} />
              )}
              {segFindings(lane).map((f) => {
                const rv = reviews?.[f.id]?.decision;
                const sup = f.stance === "supports_manipulation";
                const na = f.stance === "not_assessable";
                const hgt = sup ? LANE_H - 12 : (LANE_H - 12) / 2;
                const yy = sup ? y + 4 : y + LANE_H - 8 - hgt;
                return (
                  <rect key={f.id} x={x(f.segment.start)} y={yy}
                        width={Math.max(3, x(f.segment.end) - x(f.segment.start))} height={hgt}
                        fill={na ? "url(#hatch)" : STANCE_COLOR[f.stance]}
                        opacity={rv === "reject" ? 0.2 : sup ? 0.35 + 0.6 * (f.score || 0) : 0.35}
                        stroke={f.id === selectedId ? "#0f172a" : rv === "accept" ? "#2563eb" : "none"}
                        strokeWidth={f.id === selectedId ? 2.5 : 1.5}
                        strokeDasharray={rv === "reject" ? "3 2" : undefined}
                        style={{ cursor: "pointer" }} onClick={() => onSelect(f)}>
                    <title>{`${f.check} · ${f.stance} · ${fmt(f.segment.start)}–${fmt(f.segment.end)}s${rv ? ` · ${rv}ed` : ""}`}</title>
                  </rect>
                );
              })}
              {chips.slice(0, 6).map((f, k) => {
                const rv = reviews?.[f.id]?.decision;
                const cx = X1 + 10 + (k % 2) * 72;
                const cy = y + 4 + Math.floor(k / 2) * 15;
                return (
                  <g key={f.id} style={{ cursor: "pointer" }} onClick={() => onSelect(f)} opacity={rv === "reject" ? 0.35 : 1}>
                    <rect x={cx} y={cy} width={68} height={13} rx={3} fill={STANCE_COLOR[f.stance]} opacity={0.25}
                          stroke={f.id === selectedId ? "#0f172a" : "none"} />
                    <text x={cx + 3} y={cy + 10} fontSize="8.5" fill="#0f172a">
                      {short(f.check).slice(0, 15)}
                    </text>
                    <title>{`${f.check} · ${f.stance} (file-level)`}</title>
                  </g>
                );
              })}
            </g>
          );
        })}
        <line x1={x(time)} x2={x(time)} y1={TOP - 4} y2={H - 14} stroke="#0f172a" strokeWidth="1.5" pointerEvents="none" />
      </svg>
    </div>
  );
}

function Legend({ color, label, area }) {
  return (
    <span className="flex items-center gap-1">
      <span className="inline-block h-2.5 w-4 rounded-sm" style={{ background: color, opacity: area ? 0.45 : 0.8 }} />
      {label}
    </span>
  );
}

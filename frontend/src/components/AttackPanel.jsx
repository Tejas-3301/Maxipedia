import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { STATE_STYLE, TERMINAL, fmt } from "../util.js";

const STATUS_STYLE = {
  lost: "text-red-700 font-semibold",
  weaker: "text-amber-700",
  kept: "text-slate-700",
  stronger: "text-violet-700",
  new: "text-violet-700",
};

/** Attack mode: recompress / resize / add noise, re-analyse, and show the evidence degrading honestly. */
export default function AttackPanel({ jobId, actor, onOpenJob }) {
  const [level, setLevel] = useState("medium");
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(null);
  const [err, setErr] = useState(null);
  const timer = useRef(null);

  const load = useCallback(async () => {
    clearTimeout(timer.current);
    try {
      const d = await api.attacks(jobId);
      setData(d);
      if (d.attacks.some((a) => !TERMINAL.includes(a.status))) timer.current = setTimeout(load, 1200);
    } catch (e) {
      setErr(e.message);
    }
  }, [jobId]);

  useEffect(() => {
    setData(null);
    load();
    return () => clearTimeout(timer.current);
  }, [load]);

  const launch = async (kind) => {
    setBusy(true);
    setErr(null);
    try {
      await api.attack(jobId, kind, level, actor);
      await load();
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <b>Attack mode</b>
        <span className="text-xs text-slate-500">degrade a copy, re-analyse, compare</span>
        <select value={level} onChange={(e) => setLevel(e.target.value)} className="ml-auto rounded border border-slate-300 px-1 py-0.5 text-xs">
          <option>light</option>
          <option>medium</option>
          <option>strong</option>
        </select>
        {["recompress", "resize", "noise"].map((k) => (
          <button key={k} disabled={busy} onClick={() => launch(k)}
                  className="rounded bg-orange-600 px-2 py-1 text-xs text-white disabled:opacity-40">
            {k}
          </button>
        ))}
      </div>
      {err && <p className="mt-1 text-xs text-red-700">{err}</p>}
      {data && data.attacks.length > 0 && (
        <table className="mt-2 w-full text-left text-xs">
          <thead className="text-slate-500">
            <tr><th className="pr-2">Copy</th><th className="pr-2">Verdict</th><th className="pr-2">P</th><th className="pr-2">Conf.</th><th /></tr>
          </thead>
          <tbody>
            <tr className="border-t border-slate-100">
              <td className="py-1 pr-2 font-semibold">original</td>
              <td className="pr-2">{data.parent.state}</td>
              <td className="pr-2">{fmt(data.parent.probability_manipulated)}</td>
              <td className="pr-2">{fmt(data.parent.confidence)}</td>
              <td />
            </tr>
            {data.attacks.map((a) => (
              <Fragment key={a.job_id}>
                <tr className="border-t border-slate-100 align-top">
                  <td className="py-1 pr-2" title={a.attack?.description}>{a.attack?.kind} ({a.attack?.level})</td>
                  {a.status === "done" ? (
                    <>
                      <td className="pr-2"><span className={`rounded border px-1 ${STATE_STYLE[a.state] ?? ""}`}>{a.state}</span></td>
                      <td className="pr-2">{fmt(a.probability_manipulated)}</td>
                      <td className="pr-2">{a.confidence === null ? "–" : fmt(a.confidence)}</td>
                      <td className="whitespace-nowrap">
                        <button className="underline" onClick={() => setOpen(open === a.job_id ? null : a.job_id)}>evidence</button>{" "}
                        <button className="underline" onClick={() => onOpenJob(a.job_id)}>open</button>
                      </td>
                    </>
                  ) : (
                    <td colSpan={4} className="animate-pulse text-slate-500">{a.status}…</td>
                  )}
                </tr>
                {open === a.job_id && a.checks && (
                  <tr>
                    <td colSpan={5} className="pb-2">
                      <div className="mb-1 text-slate-500">{a.attack?.description}</div>
                      <table className="w-full">
                        <thead className="text-slate-500">
                          <tr><th className="pr-2 text-left">Check</th><th className="pr-2">supports: original → attacked</th><th className="pr-2">clean</th><th>status</th></tr>
                        </thead>
                        <tbody>
                          {a.checks.filter((r) => r.status || r.parent_clean !== r.attacked_clean).map((r) => (
                            <tr key={r.check}>
                              <td className="pr-2 font-mono">{r.check}</td>
                              <td className="pr-2 text-center">{fmt(r.parent_supports)} → {fmt(r.attacked_supports)}</td>
                              <td className="pr-2 text-center">{fmt(r.parent_clean)} → {fmt(r.attacked_clean)}</td>
                              <td className={STATUS_STYLE[r.status] ?? ""}>{r.status || "-"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

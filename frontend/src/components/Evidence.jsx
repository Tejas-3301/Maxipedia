import { useState } from "react";
import { api } from "../api.js";
import { STANCE_STYLE, fmt, seg } from "../util.js";

const IMAGE_KEYS = ["heatmap", "overview", "spectrogram", "plot"];

/** Evidence view for one finding: reasoning, images, metadata fields, raw evidence, review controls + history. */
export default function Evidence({ jobId, finding, review, actor, onReviewed }) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  if (!finding) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-500">
        Select a finding on the timeline, in the graph or in the table to see its evidence.
      </div>
    );
  }
  const f = finding;
  const ev = f.evidence || {};
  const images = IMAGE_KEYS.filter((k) => typeof ev[k] === "string" && ev[k]);
  const fields = Array.isArray(ev.fields) ? ev.fields : [];

  const send = async (decision) => {
    setBusy(true);
    setErr(null);
    try {
      await api.review(jobId, f.id, decision, decision === "note" ? note : null, actor);
      if (decision === "note") setNote("");
      onReviewed();
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3 rounded-lg border border-slate-200 bg-white p-4 text-sm">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h3 className="font-mono font-semibold">{f.check}</h3>
        <span className={`rounded px-1.5 py-0.5 text-xs ${STANCE_STYLE[f.stance]}`}>{f.stance}</span>
        <span>score {fmt(f.score)}</span>
        <span>reliability {fmt(f.reliability)}</span>
        <span>depth {f.depth}</span>
        {f.coverage !== null && f.coverage !== undefined && <span>coverage {Math.round(f.coverage * 100)}%</span>}
        <span>{seg(f.segment)}</span>
        {f.region && <span className="font-mono text-xs">region [{f.region.join(", ")}]</span>}
        {review?.decision && (
          <span className={`rounded px-1.5 py-0.5 text-xs font-semibold ${review.decision === "accept" ? "bg-blue-100 text-blue-800" : "bg-slate-800 text-white"}`}>
            {review.decision}ed
          </span>
        )}
      </div>
      <p>
        <b>Rationale:</b> {f.rationale}
      </p>
      <p className="text-slate-600">
        <b>Limitations:</b> {f.limitations}
      </p>

      {images.length > 0 && (
        <div className="grid gap-2 md:grid-cols-2">
          {images.map((k) => (
            <figure key={k}>
              <img src={api.artifactUrl(jobId, ev[k])} alt={k} className="w-full rounded border border-slate-200" />
              <figcaption className="text-xs text-slate-500">{k}</figcaption>
            </figure>
          ))}
        </div>
      )}

      {fields.length > 0 && (
        <table className="w-full text-left text-xs">
          <thead className="text-slate-500">
            <tr>
              <th className="pr-3">Field</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {fields.map((fl, i) => (
              <tr key={i} className="border-t border-slate-100 align-top">
                <td className="py-0.5 pr-3 font-mono">{fl.field}</td>
                <td className="break-all font-mono">{fl.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <details>
        <summary className="cursor-pointer text-xs text-slate-500">raw evidence</summary>
        <pre className="max-h-56 overflow-auto rounded bg-slate-50 p-2 text-xs">{JSON.stringify(ev, null, 2)}</pre>
      </details>

      <div className="rounded border border-slate-200 bg-slate-50 p-2">
        <div className="mb-1 text-xs font-semibold text-slate-600">Investigator review (logged to the signed custody chain as {actor})</div>
        <div className="flex flex-wrap gap-2">
          <button disabled={busy} onClick={() => send("accept")} className="rounded bg-blue-700 px-3 py-1 text-xs text-white disabled:opacity-40">
            Accept
          </button>
          <button disabled={busy} onClick={() => send("reject")} className="rounded bg-slate-800 px-3 py-1 text-xs text-white disabled:opacity-40">
            Reject
          </button>
          <button disabled={busy || !review?.decision} onClick={() => send("clear")} className="rounded border border-slate-300 px-3 py-1 text-xs disabled:opacity-40">
            Clear decision
          </button>
        </div>
        <div className="mt-2 flex gap-2">
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Add an annotation…"
                 className="flex-1 rounded border border-slate-300 px-2 py-1 text-xs" />
          <button disabled={busy || !note.trim()} onClick={() => send("note")} className="rounded border border-slate-400 px-3 py-1 text-xs disabled:opacity-40">
            Annotate
          </button>
        </div>
        {err && <p className="mt-1 text-xs text-red-700">{err}</p>}
        {review?.history?.length > 0 && (
          <ul className="mt-2 space-y-0.5 text-xs text-slate-600">
            {review.history.map((h, i) => (
              <li key={i}>
                <span className="font-mono">{h.ts.slice(11, 19)}</span> {h.actor}: <b>{h.decision}</b>
                {h.decision === "note" && review.notes.find((n) => n.ts === h.ts) ? ` — ${review.notes.find((n) => n.ts === h.ts).note}` : ""}
                <span className="ml-1 font-mono text-slate-400">{h.entry_hash.slice(0, 10)}…</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

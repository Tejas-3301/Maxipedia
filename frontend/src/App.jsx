import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api.js";
import AttackPanel from "./components/AttackPanel.jsx";
import Custody from "./components/Custody.jsx";
import Evidence from "./components/Evidence.jsx";
import GraphView from "./components/GraphView.jsx";
import Player from "./components/Player.jsx";
import Timeline from "./components/Timeline.jsx";
import { STANCE_STYLE, STATE_STYLE, TERMINAL, fmt, loadPref, savePref, seg } from "./util.js";

function Upload({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const submit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.upload(file);
      onUploaded(r.job_id);
      setFile(null);
      e.target.reset();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <form onSubmit={submit} className="space-y-3 rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="font-semibold">Upload media</h2>
      <input type="file" accept="video/*,audio/*,image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)}
             className="block w-full text-sm file:mr-3 file:rounded file:border-0 file:bg-slate-800 file:px-3 file:py-1.5 file:text-white" />
      <button disabled={!file || busy} className="rounded bg-slate-800 px-4 py-1.5 text-sm text-white disabled:opacity-40">
        {busy ? "Uploading…" : "Analyze"}
      </button>
      {error && <p className="text-sm text-red-700">{error}</p>}
    </form>
  );
}

function JobList({ jobs, selected, onSelect }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="mb-2 font-semibold">Recent jobs</h2>
      {jobs.length === 0 && <p className="text-sm text-slate-500">No jobs yet.</p>}
      <ul className="space-y-1">
        {jobs.map((j) => (
          <li key={j.id}>
            <button onClick={() => onSelect(j.id)}
                    className={`w-full rounded px-2 py-1.5 text-left text-sm hover:bg-slate-100 ${selected === j.id ? "bg-slate-100 ring-1 ring-slate-300" : ""}`}>
              <div className="truncate font-medium">{j.filename}</div>
              <div className="text-xs text-slate-500">{j.status}{j.verdict ? ` · ${j.verdict}` : ""}</div>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

const STAGES = ["queued", "preprocessing", "screening", "analyzing", "deep", "done"];

function JobStatus({ job, findings }) {
  const stage = STAGES.indexOf(job.status);
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3 text-sm">
      <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
        <h2 className="text-base font-semibold">{job.filename}</h2>
        <span>Type: {job.media_type ?? "–"}</span>
        <span>Duration: {job.duration_s ? `${fmt(job.duration_s)} s` : "–"}</span>
        <span>{findings.length} findings</span>
        <div className="flex items-center gap-1 text-xs">
          {STAGES.slice(1).map((s, i) => (
            <span key={s} className={`rounded px-1.5 py-0.5 ${i + 1 < stage || job.status === "done" ? "bg-slate-800 text-white"
              : i + 1 === stage ? "animate-pulse bg-amber-300" : "bg-slate-100 text-slate-400"}`}>{s}</span>
          ))}
          {job.status === "deepening" && <span className="animate-pulse rounded bg-violet-300 px-1.5 py-0.5">deepening</span>}
          {job.status === "failed" && <span className="rounded bg-red-600 px-1.5 py-0.5 text-white">failed</span>}
        </div>
      </div>
      <div className="mt-1 break-all font-mono text-xs text-slate-500">SHA-256 {job.sha256}</div>
      {job.error && <p className="mt-1 text-red-700">Error: {job.error}</p>}
    </div>
  );
}

function Assessment({ a, reviewedCf }) {
  const cov = a.coverage || {};
  return (
    <div className={`rounded-lg border-2 p-3 text-sm ${STATE_STYLE[a.state] ?? ""}`}>
      <div className="flex flex-wrap items-baseline gap-x-4">
        <h2 className="text-lg font-bold capitalize">{a.state}</h2>
        <span>P(manipulated) <b>{fmt(a.probability_manipulated)}</b></span>
        <span>Confidence <b>{a.confidence === null ? "– (abstained)" : fmt(a.confidence)}</b></span>
      </div>
      <p className="mt-1">{a.summary}</p>
      {cov.deep_coverage !== null && cov.deep_coverage !== undefined && (
        <p className="mt-1 text-xs">
          Deep coverage <b>{Math.round(cov.deep_coverage * 100)}%</b> in {cov.deep_elapsed_s}s (deadline {cov.budget_s}s) ·
          controls {cov.controls} ({cov.control_hits} hits) · {cov.unanalysed_intervals?.length || 0} intervals not analysed
        </p>
      )}
      {reviewedCf && (
        <div className={`mt-2 rounded border border-dashed p-2 text-xs ${STATE_STYLE[reviewedCf.after.state] ?? ""}`}>
          <b>Reviewed verdict</b> (without {reviewedCf.excluded.length} rejected finding(s)):{" "}
          <b className="capitalize">{reviewedCf.after.state}</b>, P {fmt(reviewedCf.after.probability_manipulated)}
          {reviewedCf.state_changed && " — differs from the automatic verdict"}
        </div>
      )}
      <details className="mt-1 text-xs">
        <summary className="cursor-pointer">limitations ({a.limitations.length})</summary>
        <ul className="mt-1 list-disc pl-5">
          {a.limitations.map((l, i) => <li key={i}>{l}</li>)}
        </ul>
      </details>
    </div>
  );
}

function FindingsTable({ findings, impact, excluded, onToggle, onSelect, selectedId, reviews }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white p-3">
      <table className="w-full text-left text-sm">
        <thead className="border-b text-xs uppercase text-slate-500">
          <tr>
            <th className="py-1 pr-2">Remove</th><th className="pr-3">Check</th><th className="pr-3">Modality</th>
            <th className="pr-3">Stance</th><th className="pr-3">Depth</th><th className="pr-3">Score</th>
            <th className="pr-3">Rel.</th><th className="pr-3">Segment</th><th className="pr-3">Review</th>
            <th className="pr-3">If removed</th>
          </tr>
        </thead>
        <tbody>
          {findings.map((f) => {
            const im = impact[f.id];
            const off = excluded.has(f.id);
            return (
              <tr key={f.id} onClick={() => onSelect(f.id)}
                  className={`cursor-pointer border-b border-slate-100 hover:bg-slate-50 ${f.id === selectedId ? "bg-amber-50" : ""} ${off ? "opacity-40 line-through" : ""}`}>
                <td className="pr-2" onClick={(e) => e.stopPropagation()}>
                  {im && <input type="checkbox" checked={off} onChange={() => onToggle(f.id)} />}
                </td>
                <td className="py-1 pr-3 font-mono text-xs">{f.check}</td>
                <td className="pr-3">{f.modality}</td>
                <td className="pr-3"><span className={`rounded px-1.5 py-0.5 text-xs ${STANCE_STYLE[f.stance]}`}>{f.stance}</span></td>
                <td className="pr-3 text-xs">{f.depth}</td>
                <td className="pr-3">{fmt(f.score)}</td>
                <td className="pr-3">{fmt(f.reliability)}</td>
                <td className="whitespace-nowrap pr-3">{seg(f.segment)}</td>
                <td className="pr-3 text-xs">{reviews[f.id]?.decision ?? ""}{reviews[f.id]?.notes?.length ? ` ✎${reviews[f.id].notes.length}` : ""}</td>
                <td className="whitespace-nowrap pr-3 text-xs">
                  {im ? `${fmt(im.p_without)} (${im.delta_p >= 0 ? "+" : ""}${fmt(im.delta_p)})${im.flips_verdict ? " ✱" : ""}` : "–"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const TABS = ["evidence", "graph", "findings", "custody"];

export default function App() {
  const [jobs, setJobs] = useState([]);
  // deep link: #<job>[/<finding>[/<tab>]]
  const [selected, setSelectedState] = useState(() => window.location.hash.slice(1).split("/")[0] || null);
  const pendingFocus = useRef(window.location.hash.slice(1).split("/")[1] || null);
  const pendingTab = useRef(window.location.hash.slice(1).split("/")[2] || null);
  const [job, setJob] = useState(null);
  const [info, setInfo] = useState(null);
  const [screening, setScreening] = useState(null);
  const [schedule, setSchedule] = useState(null);
  const [findings, setFindings] = useState([]);
  const [assessment, setAssessment] = useState(null);
  const [graph, setGraph] = useState(null);
  const [reviews, setReviews] = useState({});
  const [custody, setCustody] = useState(null);
  const [syncCurve, setSyncCurve] = useState(null);
  const [excluded, setExcluded] = useState(new Set());
  const [cf, setCf] = useState(null);
  const [reviewedCf, setReviewedCf] = useState(null);
  const [focusId, setFocusId] = useState(null);
  const [time, setTime] = useState(0);
  const [tab, setTab] = useState("evidence");
  const [actor, setActor] = useState(() => loadPref("actor", "investigator"));
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const [pollKey, setPollKey] = useState(0);
  const player = useRef(null);
  const timer = useRef(null);
  const infoLoaded = useRef(false);

  const setSelected = useCallback((id) => {
    setSelectedState(id);
    window.history.replaceState(null, "", id ? `#${id}` : " ");
  }, []);
  const refreshJobs = useCallback(() => api.listJobs().then(setJobs).catch((e) => setError(e.message)), []);

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(`Backend unreachable: ${e.message}`));
    refreshJobs();
  }, [refreshJobs]);

  const loadReviews = useCallback(async (id) => {
    const [r, c] = await Promise.all([api.reviews(id), api.custody(id)]);
    setReviews(r.reviews);
    setCustody(c);
    const rejected = Object.entries(r.reviews).filter(([, v]) => v.decision === "reject").map(([k]) => k);
    setReviewedCf(rejected.length ? await api.counterfactual(id, rejected, null) : null);
  }, []);

  useEffect(() => {
    infoLoaded.current = false;
    setJob(null); setInfo(null); setScreening(null); setSchedule(null); setFindings([]); setAssessment(null);
    setGraph(null); setReviews({}); setCustody(null); setSyncCurve(null); setExcluded(new Set()); setCf(null);
    setReviewedCf(null); setFocusId(null); setTime(0);
  }, [selected]);

  // Progressive polling: screening curve first, then whole-file findings, then deep findings, then the verdict.
  useEffect(() => {
    clearTimeout(timer.current);
    if (!selected) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const j = await api.job(selected);
        if (cancelled) return;
        setJob(j);
        if (!["queued", "preprocessing"].includes(j.status)) {
          const [fs, sc] = await Promise.all([api.findings(selected), api.screening(selected)]);
          if (cancelled) return;
          setFindings(fs);
          setScreening(sc.screening);
          setSchedule(sc.schedule);
          if (!infoLoaded.current) {
            infoLoaded.current = true;
            api.info(selected).then((i) => !cancelled && setInfo(i)).catch(() => { infoLoaded.current = false; });
          }
        }
        if (!TERMINAL.includes(j.status)) {
          timer.current = setTimeout(poll, 800);
          return;
        }
        refreshJobs();
        if (j.status === "done") {
          const [a, g] = await Promise.all([api.assessment(selected), api.graph(selected)]);
          if (cancelled) return;
          setAssessment(a);
          setGraph(g);
          await loadReviews(selected);
        }
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    };
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer.current);
    };
  }, [selected, pollKey, refreshJobs, loadReviews]);

  // lip-sync offset curve for the sync lane
  useEffect(() => {
    const sf = findings.find((f) => f.check === "sync.lipsync" && f.evidence?.curve_path);
    if (!sf || !selected) return;
    api.artifactJson(selected, sf.evidence.curve_path)
      .then((c) => setSyncCurve(c.windows.filter((w) => w.status === "ok" && w.r_at_offset !== undefined)
        .map((w) => ({ t: (w.start + w.end) / 2, r: w.r_at_offset }))))
      .catch(() => {});
  }, [findings, selected]);

  const byId = useMemo(() => Object.fromEntries(findings.map((f) => [f.id, f])), [findings]);
  const impact = useMemo(() => Object.fromEntries((graph?.nodes || []).filter((n) => n.impact).map((n) => [n.id, n.impact])), [graph]);
  const focus = focusId ? byId[focusId] : null;

  useEffect(() => {
    if (pendingFocus.current && byId[pendingFocus.current] && info) {
      const f = byId[pendingFocus.current];
      pendingFocus.current = null;
      setFocusId(f.id);
      if (f.segment) setTimeout(() => player.current?.seek(f.segment.start + 0.05), 300);
    }
    if (pendingTab.current && TABS.includes(pendingTab.current) && (pendingTab.current !== "graph" || graph)) {
      setTab(pendingTab.current);
      pendingTab.current = null;
    }
  }, [byId, info, graph]);

  const select = (fOrId) => {
    const f = typeof fOrId === "string" ? byId[fOrId] : fOrId;
    if (!f) return;
    setFocusId(f.id);
    setTab("evidence");
    window.history.replaceState(null, "", `#${selected}/${f.id}`);
    if (f.segment) player.current?.seek(f.segment.start + Math.min(0.05, (f.segment.end - f.segment.start) / 2));
  };

  const runCf = async (next) => {
    setExcluded(next);
    if (!next.size) {
      setCf(null);
      return;
    }
    try {
      setCf(await api.counterfactual(selected, [...next], null)); // exploration only: not a custody event
    } catch (e) {
      setError(e.message);
    }
  };
  const toggleExclude = (id) => {
    const next = new Set(excluded);
    next.has(id) ? next.delete(id) : next.add(id);
    runCf(next);
  };

  const analyzeDeeper = async () => {
    const t = player.current?.time() ?? time;
    try {
      await api.deep(selected, Math.max(0, t - 1.5), t + 1.5, actor);
      setAssessment(null);
      setGraph(null);
      setPollKey((k) => k + 1);
    } catch (e) {
      setError(e.message);
    }
  };

  const isVideo = job?.media_type === "video";

  return (
    <div className="mx-auto max-w-[1500px] p-4">
      <header className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h1 className="text-2xl font-bold">Media Forensics Workspace</h1>
        <span className="flex items-center gap-3 text-xs text-slate-500">
          <label>
            Investigator{" "}
            <input value={actor} onChange={(e) => { setActor(e.target.value); savePref("actor", e.target.value); }}
                   className="w-28 rounded border border-slate-300 px-1 py-0.5" />
          </label>
          {health ? `backend ${health.status}` : "connecting…"}
        </span>
      </header>
      {error && (
        <div className="mb-3 rounded border border-red-300 bg-red-50 p-2 text-sm text-red-800">
          {error} <button className="underline" onClick={() => setError(null)}>dismiss</button>
        </div>
      )}
      <div className="grid gap-4 lg:grid-cols-[16rem_minmax(0,1fr)]">
        <aside className="space-y-4">
          <Upload onUploaded={(id) => { refreshJobs(); setSelected(id); }} />
          <JobList jobs={jobs} selected={selected} onSelect={setSelected} />
        </aside>
        <main className="min-w-0 space-y-4">
          {!job && <p className="text-slate-500">Upload a file or pick a job.</p>}
          {job && <JobStatus job={job} findings={findings} />}
          {job && isVideo && (
            <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
              <div className="space-y-2">
                <Player ref={player} jobId={selected} info={info} focus={focus} onTime={setTime} />
                {job.status === "done" && (
                  <button onClick={analyzeDeeper} className="rounded border border-violet-400 px-3 py-1 text-xs text-violet-800 hover:bg-violet-50">
                    Analyze deeper here ({fmt(Math.max(0, time - 1.5), 1)}–{fmt(time + 1.5, 1)} s)
                  </button>
                )}
              </div>
              <div className="space-y-2">
                {assessment ? <Assessment a={assessment} reviewedCf={reviewedCf} /> : (
                  <div className="rounded-lg border border-dashed border-slate-300 p-4 text-sm text-slate-500">
                    The verdict appears when deep analysis finishes; the screening curve and findings fill in below.
                  </div>
                )}
                {assessment && (
                  <div className="flex flex-wrap gap-2">
                    <a href={api.reportUrl(selected, actor)}
                       onClick={() => setTimeout(() => loadReviews(selected), 1500)}
                       className="rounded bg-slate-800 px-3 py-1 text-xs text-white">
                      Export PDF report (with chain of custody)
                    </a>
                    {job.attack && (
                      <span className="rounded bg-orange-100 px-2 py-1 text-xs text-orange-800">
                        attacked copy: {job.attack.description} ·{" "}
                        <button className="underline" onClick={() => setSelected(job.derived_from)}>open original</button>
                      </span>
                    )}
                  </div>
                )}
                {assessment && !job.attack && <AttackPanel jobId={selected} actor={actor} onOpenJob={setSelected} />}
                {cf && (
                  <div className={`rounded-lg border-2 border-dashed p-2 text-xs ${STATE_STYLE[cf.after.state] ?? ""}`}>
                    <b>Counterfactual</b> without {cf.excluded.length} finding(s): <b className="capitalize">{cf.after.state}</b>,
                    P {fmt(cf.after.probability_manipulated)} (ΔP {cf.delta_p >= 0 ? "+" : ""}{fmt(cf.delta_p)})
                    {cf.state_changed && " — verdict changes"}{" "}
                    <button className="underline" onClick={() => runCf(new Set())}>reset</button>
                  </div>
                )}
              </div>
            </div>
          )}
          {job && !isVideo && assessment && <Assessment a={assessment} reviewedCf={reviewedCf} />}
          {job && isVideo && (screening || findings.length > 0) && (
            <Timeline duration={job.duration_s} screening={screening} schedule={schedule} findings={findings}
                      reviews={reviews} syncCurve={syncCurve} time={time} selectedId={focusId}
                      onSelect={select} onSeek={(t) => player.current?.seek(t)} />
          )}
          {job && findings.length > 0 && (
            <div>
              <div className="mb-2 flex gap-1">
                {TABS.map((t) => (
                  <button key={t} onClick={() => setTab(t)}
                          className={`rounded-t px-3 py-1 text-sm capitalize ${tab === t ? "bg-slate-800 text-white" : "bg-slate-200 text-slate-700"}`}>
                    {t}
                  </button>
                ))}
              </div>
              {tab === "evidence" && (
                <Evidence jobId={selected} finding={focus} review={focus ? reviews[focus.id] : null} actor={actor}
                          onReviewed={() => loadReviews(selected)} />
              )}
              {tab === "graph" && (graph ? (
                <GraphView graph={graph} cf={cf} excluded={excluded} onToggle={toggleExclude} onSelect={select}
                           selectedId={focusId} reviews={reviews} onUseRejections={(ids) => runCf(new Set(ids))}
                           onReset={() => runCf(new Set())} />
              ) : <p className="text-sm text-slate-500">The graph is available once the verdict is computed.</p>)}
              {tab === "findings" && (
                <FindingsTable findings={findings} impact={impact} excluded={excluded} onToggle={toggleExclude}
                               onSelect={select} selectedId={focusId} reviews={reviews} />
              )}
              {tab === "custody" && <Custody custody={custody} />}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

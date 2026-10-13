const BASE = "/api";

async function req(path, opts) {
  const r = await fetch(BASE + path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${r.status}: ${detail}`);
  }
  return r.json();
}

const post = (path, body) =>
  req(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  health: () => req("/health"),
  listJobs: () => req("/jobs?limit=30"),
  upload: (file) => {
    const fd = new FormData();
    fd.append("file", file);
    return req("/jobs", { method: "POST", body: fd });
  },
  job: (id) => req(`/jobs/${id}`),
  info: (id) => req(`/jobs/${id}/info`),
  findings: (id) => req(`/jobs/${id}/findings`),
  screening: (id) => req(`/jobs/${id}/screening`),
  assessment: (id) => req(`/jobs/${id}/assessment`),
  graph: (id) => req(`/jobs/${id}/graph`),
  reviews: (id) => req(`/jobs/${id}/reviews`),
  custody: (id) => req(`/jobs/${id}/custody`),
  counterfactual: (id, exclude, actor) => post(`/jobs/${id}/counterfactual`, { exclude, actor }),
  review: (id, fid, decision, note, actor) => post(`/jobs/${id}/findings/${fid}/review`, { decision, note, actor }),
  deep: (id, start, end, actor) => post(`/jobs/${id}/deep`, { start, end, actor }),
  attack: (id, kind, level, actor) => post(`/jobs/${id}/attack`, { kind, level, actor }),
  attacks: (id) => req(`/jobs/${id}/attacks`),
  reportUrl: (id, actor) => `${BASE}/jobs/${id}/report.pdf?actor=${encodeURIComponent(actor)}`,
  mediaUrl: (id) => `${BASE}/jobs/${id}/media`,
  artifactUrl: (id, path) => `${BASE}/jobs/${id}/artifact?path=${encodeURIComponent(path)}`,
  artifactJson: (id, path) => req(`/jobs/${id}/artifact?path=${encodeURIComponent(path)}`),
};

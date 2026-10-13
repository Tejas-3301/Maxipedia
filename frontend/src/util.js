export const TERMINAL = ["done", "failed"];
export const LANES = ["visual", "audio", "sync", "metadata", "provenance"];

export const STATE_STYLE = {
  "likely manipulated": "bg-red-50 border-red-300 text-red-900",
  "likely authentic": "bg-emerald-50 border-emerald-300 text-emerald-900",
  inconclusive: "bg-amber-50 border-amber-300 text-amber-900",
};

export const STANCE_STYLE = {
  supports_manipulation: "bg-red-100 text-red-800",
  contradicts: "bg-emerald-100 text-emerald-800",
  neutral: "bg-slate-100 text-slate-700",
  not_assessable: "bg-slate-200 text-slate-500",
};

export const STANCE_COLOR = {
  supports_manipulation: "#dc2626",
  contradicts: "#16a34a",
  neutral: "#94a3b8",
  not_assessable: "#cbd5e1",
};

export const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
export const seg = (s) => (s ? `${fmt(s.start)}–${fmt(s.end)} ${s.unit}` : "file");
export const short = (check) => check.replace(/^(visual|audio|metadata|provenance|sync|screen)\./, "");

export function loadPref(key, fallback) {
  try {
    return window.localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

export function savePref(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable: keep in memory only */
  }
}

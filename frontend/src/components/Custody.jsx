/** Chain-of-custody log with verification status. */
export default function Custody({ custody }) {
  if (!custody) return null;
  const v = custody.verification;
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm">
      <div className="mb-2 flex flex-wrap items-baseline gap-3">
        <h3 className="font-semibold">Chain of custody</h3>
        <span className={`rounded px-2 py-0.5 text-xs font-semibold ${v.intact ? "bg-emerald-100 text-emerald-800" : "bg-red-100 text-red-800"}`}>
          {v.intact ? "intact · all Ed25519 signatures valid" : "BROKEN"}
        </span>
        <span className="text-xs text-slate-500">
          {v.entries} entries · key {v.key_id} · head {v.head?.slice(0, 12)}…
        </span>
      </div>
      {v.problems.length > 0 && (
        <ul className="mb-2 list-disc pl-5 text-xs text-red-700">
          {v.problems.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b text-slate-500">
            <tr>
              <th className="py-1 pr-2">#</th>
              <th className="pr-2">Time (UTC)</th>
              <th className="pr-2">Actor</th>
              <th className="pr-2">Action</th>
              <th className="pr-2">Detail</th>
              <th>Entry hash</th>
            </tr>
          </thead>
          <tbody>
            {custody.entries.map((e) => (
              <tr key={e.seq} className="border-b border-slate-100 align-top">
                <td className="py-1 pr-2">{e.seq}</td>
                <td className="whitespace-nowrap pr-2 font-mono">{e.ts.replace("T", " ").slice(0, 19)}</td>
                <td className="pr-2">{e.actor}</td>
                <td className="pr-2 font-semibold">{e.action}</td>
                <td className="max-w-md break-all pr-2 font-mono text-slate-600">{e.detail}</td>
                <td className="font-mono text-slate-400">{e.entry_hash.slice(0, 12)}…</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

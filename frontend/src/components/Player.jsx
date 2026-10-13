import { forwardRef, useImperativeHandle, useRef, useState } from "react";
import { api } from "../api.js";
import { fmt, short } from "../util.js";

/** Video (proxy) with the selected finding's region box drawn in original-pixel coordinates. */
const Player = forwardRef(function Player({ jobId, info, focus, onTime }, ref) {
  const video = useRef(null);
  const [t, setT] = useState(0);
  const [showHeat, setShowHeat] = useState(false);

  useImperativeHandle(ref, () => ({
    seek(sec) {
      if (video.current) {
        video.current.pause();
        video.current.currentTime = Math.max(0, sec);
      }
      // show the box straight away; 'seeked' may arrive late (or never, before metadata has loaded)
      setT(Math.max(0, sec));
      onTime?.(Math.max(0, sec));
    },
    time: () => video.current?.currentTime ?? 0,
  }));

  const W = info?.width || 640;
  const H = info?.height || 480;
  const region = focus?.region;
  const s = focus?.segment;
  const inSegment = s && t >= s.start - 0.15 && t <= s.end + 0.15;
  const heat = focus?.evidence?.heatmap;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="relative w-full overflow-hidden rounded bg-black" style={{ aspectRatio: `${W} / ${H}` }}>
        <video
          ref={video}
          src={api.mediaUrl(jobId)}
          controls
          preload="metadata"
          className="absolute inset-0 h-full w-full"
          style={{ objectFit: "fill" }}
          onTimeUpdate={(e) => {
            setT(e.target.currentTime);
            onTime?.(e.target.currentTime);
          }}
          onSeeked={(e) => {
            setT(e.target.currentTime);
            onTime?.(e.target.currentTime);
          }}
        />
        {showHeat && heat && inSegment && (
          <img
            src={api.artifactUrl(jobId, heat)}
            alt="heatmap"
            className="pointer-events-none absolute inset-0 h-full w-full opacity-60"
          />
        )}
        {region && inSegment && (
          <div
            className="pointer-events-none absolute border-2 border-yellow-300 shadow-[0_0_0_2px_rgba(0,0,0,0.6)]"
            style={{
              left: `${(region[0] / W) * 100}%`,
              top: `${(region[1] / H) * 100}%`,
              width: `${(region[2] / W) * 100}%`,
              height: `${(region[3] / H) * 100}%`,
            }}
          >
            <span className="absolute -top-5 left-0 whitespace-nowrap bg-yellow-300 px-1 text-[10px] font-semibold text-black">
              {short(focus.check)} {fmt(focus.score)}
            </span>
          </div>
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-slate-600">
        <span className="font-mono">t = {fmt(t)} s</span>
        {focus && (
          <span>
            Selected: <b>{focus.check}</b> {s ? `${fmt(s.start)}–${fmt(s.end)} s` : "(file-level)"}
            {region ? ` · box [${region.join(", ")}]` : ""}
          </span>
        )}
        {heat && (
          <label className="ml-auto flex items-center gap-1">
            <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /> heatmap overlay
          </label>
        )}
      </div>
    </div>
  );
});

export default Player;

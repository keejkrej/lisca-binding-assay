import json
import os
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(
    os.environ.get(
        "LNP_RESULTS_DIR", str(Path.home() / "data" / "lnp-tracking-spotiflow")
    )
)
alltraces = []
summary = []
fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for p in sorted(root.glob("*/detections.csv")):
    d = pd.read_csv(p)
    qc = pd.read_csv(p.parent / "frame_qc.csv")
    recording = p.parent.name[-3:]
    tr = (
        d.groupby("frame")
        .agg(
            puncta_count=("intensity", "size"),
            mean_punctum_intensity_adu=("intensity", "mean"),
            median_punctum_intensity_adu=("intensity", "median"),
            std_punctum_intensity_adu=("intensity", "std"),
        )
        .reset_index()
    )
    tr = qc[["frame", "time_s"]].merge(tr, on="frame", how="left")
    tr["puncta_count"] = tr.puncta_count.fillna(0).astype(int)
    tr.insert(0, "recording", recording)
    tr["time_min"] = tr.time_s / 60
    assert len(tr) == len(qc)
    assert tr.puncta_count.sum() == len(d)
    tr.to_csv(p.parent / "puncta_traces.csv", index=False)
    alltraces.append(tr)
    for ax, col in zip(axs, ["puncta_count", "mean_punctum_intensity_adu"]):
        ax.plot(tr.time_min, tr[col], lw=1.3, label=recording)
    summary.append(
        {
            "recording": recording,
            "start_count": int(tr.puncta_count.iloc[0]),
            "end_count": int(tr.puncta_count.iloc[-1]),
            "start_mean_intensity": float(tr.mean_punctum_intensity_adu.iloc[0]),
            "end_mean_intensity": float(tr.mean_punctum_intensity_adu.iloc[-1]),
        }
    )
axs[0].set_ylabel("Detected puncta / field")
axs[1].set_ylabel("Mean punctum intensity (ADU)")
axs[1].set_xlabel("Elapsed time within recording (min)")
axs[0].legend(title="Recording")
fig.suptitle("Spotiflow puncta traces — full field, all detections")
fig.tight_layout()
fig.savefig(root / "puncta_traces.png", dpi=180)
pd.concat(alltraces, ignore_index=True).to_csv(root / "puncta_traces.csv", index=False)
pd.DataFrame(summary).to_csv(root / "puncta_trace_summary.csv", index=False)
data = [
    {
        "name": str(t.recording.iloc[0]),
        "points": t[["time_min", "puncta_count", "mean_punctum_intensity_adu"]]
        .round(3)
        .to_numpy()
        .tolist(),
    }
    for t in alltraces
]
html = """<!doctype html><html><style>svg{width:100%;height:260px;display:block}text{fill:var(--foreground);font:12px sans-serif}.note{font-size:13px;color:var(--muted-foreground)}.legend{display:flex;gap:20px;font-size:13px;margin:8px 0}.label{font-size:14px;margin-top:12px}#readout{font-size:13px;min-height:22px}</style><div class="legend"><span style="color:var(--chart-1)">● 005</span><span style="color:var(--chart-2)">● 006</span><span style="color:var(--chart-3)">● 007</span></div><div class="label">Number of puncta per field</div><svg id="count" viewBox="0 0 700 260"></svg><div class="label">Mean fluorescence at detected punctum centers (ADU)</div><svg id="intensity" viewBox="0 0 700 260"></svg><div id="readout">Hover over either plot for values.</div><p class="note">All Spotiflow detections, including short tracks. Each recording starts at its own zero; recordings are not stitched. Intensity is raw interpolated fluorescence at spot centers, without background subtraction or bleaching correction. These are full-field traces, not per-cell counts.</p><script>const data=DATA,ns='http://www.w3.org/2000/svg',colors=['var(--chart-1)','var(--chart-2)','var(--chart-3)'];function draw(id,col){const svg=document.getElementById(id);function add(tag,a,t){const e=document.createElementNS(ns,tag);for(const k in a)e.setAttribute(k,a[k]);if(t)e.textContent=t;svg.append(e);return e}const max=Math.max(...data.flatMap(s=>s.points.map(p=>p[col]))),unit=col==1?200:100,ymax=Math.ceil(max/unit)*unit;const yy=v=>210-v/ymax*175;for(let i=0;i<=4;i++){const v=ymax*i/4;add('line',{x1:62,x2:680,y1:yy(v),y2:yy(v),stroke:'var(--border)'});add('text',{x:5,y:yy(v)+4},Math.round(v))}for(let i=0;i<=6;i++)add('text',{x:60+i*100,y:232},i*5);add('text',{x:275,y:256},'Elapsed time (min)');data.forEach((s,i)=>add('polyline',{points:s.points.map(p=>[62+p[0]*20,yy(p[col])].join(',')).join(' '),fill:'none',stroke:colors[i],'stroke-width':1.5}));const cursor=add('line',{x1:62,x2:62,y1:30,y2:210,stroke:'var(--muted-foreground)',visibility:'hidden','stroke-dasharray':'3 3'});svg.addEventListener('mousemove',ev=>{const box=svg.getBoundingClientRect(),x=(ev.clientX-box.left)/box.width*700,t=Math.max(0,Math.min(30,(x-62)/20));cursor.setAttribute('x1',62+t*20);cursor.setAttribute('x2',62+t*20);cursor.setAttribute('visibility','visible');document.getElementById('readout').textContent=t.toFixed(2)+' min · '+data.map(s=>{if(t>s.points.at(-1)[0])return s.name+': ended';const p=s.points.reduce((a,b)=>Math.abs(a[0]-t)<Math.abs(b[0]-t)?a:b);return s.name+': '+p[1]+' puncta, '+p[2].toFixed(1)+' ADU'}).join(' | ')});}draw('count',1);draw('intensity',2)</script></html>""".replace(
    "DATA", json.dumps(data)
)
(root / "puncta_traces.html").write_text(html)
print(pd.DataFrame(summary).to_string(index=False))

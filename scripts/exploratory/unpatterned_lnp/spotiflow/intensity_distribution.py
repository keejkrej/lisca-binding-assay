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
series = []
tables = []
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for ax, p in zip(axs, sorted(root.glob("*/detections.csv"))):
    d = pd.read_csv(p)
    q = pd.read_csv(p.parent / "frame_qc.csv")
    name = p.parent.name[-3:]
    g = d.groupby("frame").intensity
    t = (
        g.agg(mean="mean", median="median")
        .join(g.quantile(0.25).rename("q25"))
        .join(g.quantile(0.75).rename("q75"))
        .reset_index()
    )
    t = q[["frame", "time_s"]].merge(t, on="frame", how="left")
    t["time_min"] = t.time_s / 60
    t.insert(0, "recording", name)
    assert ((t.q25 <= t["median"]) & (t["median"] <= t.q75)).all()
    tables.append(t)
    series.append(
        {
            "name": name,
            "points": t[["time_min", "mean", "median", "q25", "q75"]]
            .round(3)
            .values.tolist(),
        }
    )
    ax.fill_between(
        t.time_min, t.q25, t.q75, color="#3985b5", alpha=0.2, label="IQR (25–75%)"
    )
    ax.plot(t.time_min, t["mean"], color="#d56b29", lw=1.5, label="Mean")
    ax.plot(t.time_min, t["median"], color="#216ca3", lw=1.5, label="Median")
    ax.set(title=f"Recording {name}", ylabel="Punctum intensity (ADU)")
    ax.legend(loc="upper right")
axs[-1].set_xlabel("Elapsed time within recording (min)")
fig.tight_layout()
fig.savefig(root / "puncta_mean_median_iqr.png", dpi=180)
pd.concat(tables).to_csv(root / "puncta_mean_median_iqr.csv", index=False)
html = """<!doctype html><html><style>svg{width:100%;height:230px;display:block}text{fill:var(--foreground);font:12px sans-serif}.legend{display:flex;gap:18px;font-size:13px;margin-bottom:12px}.label{margin-top:10px;font-size:14px}.note,#readout{font-size:13px;color:var(--muted-foreground)}</style><div class="legend"><span style="color:var(--chart-1)">━ Mean</span><span style="color:var(--chart-2)">━ Median</span><span style="color:var(--chart-2)">▧ IQR (25–75%)</span></div><div id="charts"></div><div id="readout">Hover over a plot for values.</div><p class="note">Distribution across all detected puncta in each frame. The band shows variation among puncta, not uncertainty in the mean. Raw center intensity in ADU; no background subtraction. Each recording has its own time zero.</p><script>const data=DATA,ns='http://www.w3.org/2000/svg';const ymax=Math.ceil(Math.max(...data.flatMap(s=>s.points.flatMap(p=>p.slice(1))))/100)*100;for(const s of data){const host=document.getElementById('charts'),label=document.createElement('div');label.className='label';label.textContent='Recording '+s.name;host.append(label);const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 700 230');host.append(svg);function add(tag,a,t){const e=document.createElementNS(ns,tag);for(const k in a)e.setAttribute(k,a[k]);if(t)e.textContent=t;svg.append(e);return e}const x=t=>60+t*20,y=v=>185-v/ymax*150;for(let i=0;i<=4;i++){const v=ymax*i/4;add('line',{x1:60,x2:680,y1:y(v),y2:y(v),stroke:'var(--border)'});add('text',{x:4,y:y(v)+4},Math.round(v))}for(let i=0;i<=6;i++)add('text',{x:x(i*5),y:205},i*5);add('text',{x:270,y:228},'Elapsed time (min)');add('polygon',{points:s.points.map(p=>[x(p[0]),y(p[3])].join(',')).concat([...s.points].reverse().map(p=>[x(p[0]),y(p[4])].join(','))).join(' '),fill:'var(--chart-2)','fill-opacity':.2});for(const [col,color] of [[1,'var(--chart-1)'],[2,'var(--chart-2)']])add('polyline',{points:s.points.map(p=>[x(p[0]),y(p[col])].join(',')).join(' '),fill:'none',stroke:color,'stroke-width':1.6});svg.addEventListener('mousemove',ev=>{const b=svg.getBoundingClientRect(),t=Math.max(0,Math.min(s.points.at(-1)[0],((ev.clientX-b.left)/b.width*700-60)/20));const p=s.points.reduce((a,b)=>Math.abs(a[0]-t)<Math.abs(b[0]-t)?a:b);document.getElementById('readout').textContent=s.name+' · '+p[0].toFixed(2)+' min · mean '+p[1].toFixed(1)+' · median '+p[2].toFixed(1)+' · IQR '+p[3].toFixed(1)+'–'+p[4].toFixed(1)+' ADU'})}</script></html>""".replace(
    "DATA", json.dumps(series)
)
(root / "puncta_mean_median_iqr.html").write_text(html)

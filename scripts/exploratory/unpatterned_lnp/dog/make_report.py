import json
import os
from pathlib import Path

import pandas as pd

root = Path(
    os.environ.get("LNP_RESULTS_DIR", str(Path.home() / "data" / "lnp-tracking"))
)
series = []
for p in sorted(root.glob("*/frame_qc.csv")):
    q = pd.read_csv(p)
    series.append(
        {
            "name": p.parent.name[-3:],
            "points": list(zip((q.time_s / 60).round(3), q.detections)),
        }
    )
html = """<!doctype html><html><style>svg{width:100%;height:300px}text{fill:var(--foreground);font:12px sans-serif} .note{color:var(--muted-foreground);font:13px sans-serif}</style><div>Detected fluorescent puncta per frame</div><svg viewBox="0 0 700 300" id="chart"></svg><p class="note">Independent recordings; counts depend on focus and detection threshold. Time is relative to the start of each recording.</p><script>const data=DATA;const svg=document.getElementById('chart'),ns='http://www.w3.org/2000/svg';function add(tag,a,t){const e=document.createElementNS(ns,tag);for(const k in a)e.setAttribute(k,a[k]);if(t)e.textContent=t;svg.append(e)}const ymax=Math.ceil(Math.max(...data.flatMap(s=>s.points.map(p=>p[1])))/200)*200;for(let i=0;i<=4;i++){let y=250-i*50;add('line',{x1:55,x2:680,y1:y,y2:y,stroke:'var(--border)'});add('text',{x:5,y:y+4},Math.round(ymax*i/4))}for(let i=0;i<=6;i++)add('text',{x:55+i*100,y:275},i*5);add('text',{x:320,y:298},'Time (min)');data.forEach((s,i)=>{let color=['var(--chart-1)','var(--chart-2)','var(--chart-3)'][i];add('polyline',{points:s.points.map(p=>[55+p[0]*20,250-p[1]/ymax*200].join(',')).join(' '),fill:'none',stroke:color,'stroke-width':2});add('text',{x:100+i*190,y:22,style:'fill:'+color},'Recording '+s.name)})</script>""".replace(
    "DATA", json.dumps(series)
)
(root / "report.html").write_text(html)

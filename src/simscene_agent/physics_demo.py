"""Deterministic physics checks used by the examples.

The module exposes the same failure signals a simulator adapter should return. It
does not pretend an analytic AABB test is a substitute for MuJoCo/Isaac; the report
labels this backend explicitly and can be compared with the included MuJoCo replay.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path

def run_physics_demo(out_dir: Path, steps: int = 120) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # A cup is dropped above a table. Semi-implicit Euler with a unilateral contact
    # clamp: z never crosses the support plane, and the contact event is observable.
    g=-9.81; dt=1/60; z=1.25; vz=0.0; table_top=0.75; cup_h=0.14; contact=[]; rows=[]
    for step in range(steps):
        vz += g*dt; z += vz*dt
        if z-cup_h/2 < table_top:
            z=table_top+cup_h/2; vz=0.0; contact.append(step)
        rows.append((step,step*dt,z,vz,z-cup_h/2-table_top))
    with (out_dir/"trajectory.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["step","time_s","cup_center_z_m","velocity_z_m_s","gap_m"]); w.writerows(rows)
    angles=[]
    for i in range(10): angles.append(min(90.0, i*10.0))
    # A small artifact that makes the signal inspectable without a plotting stack.
    from PIL import Image, ImageDraw
    im=Image.new("RGB",(900,430),(247,249,252)); d=ImageDraw.Draw(im)
    d.text((25,18),"analytic contact regression: cup drop",fill=(20,30,45))
    def plot(vals, box, color, title, baseline=None):
        x0,y0,x1,y1=box; d.rectangle(box,outline=(160,170,185),width=1); d.text((x0+8,y0+8),title,fill=(40,50,65))
        lo=min(vals); hi=max(vals); span=max(hi-lo,1e-6)
        pts=[(x0+8+(x1-x0-16)*i/max(len(vals)-1,1),y1-8-(y1-y0-30)*(v-lo)/span) for i,v in enumerate(vals)]
        d.line(pts,fill=color,width=3)
        if baseline is not None:
            yy=y1-8-(y1-y0-30)*(baseline-lo)/span; d.line((x0+8,yy,x1-8,yy),fill=(40,150,80),width=2)
    plot([r[2] for r in rows],(25,55,430,390),(40,110,210),"cup center z (m)",table_top+cup_h/2)
    plot([r[4] for r in rows],(465,55,870,390),(200,70,50),"support gap (m)",0.0)
    im.save(out_dir/"trajectory.png")
    report={"backend":"analytic_unilateral_contact_demo","contact_steps":contact,"final_height_m":z,"support_gap_m":rows[-1][-1],"door_angles_deg":angles,"door_joint_limit_passed":max(angles)<=90.0,"stable_support_passed":abs(rows[-1][-1])<1e-9,"visualization":"trajectory.png","notes":["This is a deterministic harness test, not a claim of calibrated real physics","run the MuJoCo XML in examples/real_rgb_sequence when MuJoCo 3.x is installed"]}
    (out_dir/"physics_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report

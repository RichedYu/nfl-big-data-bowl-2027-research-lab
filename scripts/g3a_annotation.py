"""Partial blinded agreement with private player-cluster uncertainty."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import linear_sum_assignment

from g24_annotation import FIELDS, POINTS, load, pair_events
from g3a_measurement import G23,G24
from g23_run import verify_artifacts
from movement_probe import atomic_json


def read_partial(path,expected):
    with path.open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError("Required annotation columns are absent")
        selected=[]
        for row in reader:
            if row["case"] not in expected:
                raise ValueError("Case outside review packet")
            if row["review_status"].strip():
                selected.append(row)
            elif any(row[k].strip() for k in (*POINTS,"event_index","direction","confidence","reason")):
                raise ValueError("Blank status with annotation content")
    if not selected:
        return None,{}
    subset={row["case"]:expected[row["case"]] for row in selected}
    with TemporaryDirectory() as directory:
        p=Path(directory)/"submitted.csv"
        with p.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)
        return load(p,subset)


def match_partial(a,b):
    """Pair incomplete events on at least two jointly observed time anchors."""
    def incomplete(x):
        return any(x[k] is None for k in ("onset_s","exit_s","bend_center_s"))
    cost=np.zeros((len(a),len(b)))
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            if not (incomplete(x) or incomplete(y)):
                continue
            gaps=[abs(x[k]-y[k]) for k in POINTS if x[k] is not None and y[k] is not None]
            if len(gaps)>=2 and max(gaps)<=.35:
                cost[i,j]=2+min(len(a),len(b))-np.mean(gaps)/.35
    row,col=linear_sum_assignment(cost,maximize=True)
    return [(int(i),int(j)) for i,j in zip(row,col) if cost[i,j]>0]


def agreement(left,right,clusters):
    if not set(left)|set(right)<=set(clusters):
        raise ValueError("Cluster map is incomplete")
    seen=sorted(set(left)&set(right))
    cases=[]
    for c in seen:
        x,y=left[c],right[c]
        lx,ly=x[0][0],y[0][0]
        a,b=[[p for s,p in rows if s=="events"] for rows in (x,y)]
        usable="unjudgeable" not in (lx,ly)
        pairs=pair_events(a,b) if usable else []
        unmatched_a=[i for i in range(len(a)) if i not in {m[0] for m in pairs}]
        unmatched_b=[j for j in range(len(b)) if j not in {m[1] for m in pairs}]
        partial=match_partial([a[i] for i in unmatched_a],[b[j] for j in unmatched_b]) if usable else []
        result=[]
        def event(i,j,kind,iou):
            result.append(dict(left=a[i]["index"],right=b[j]["index"],kind=kind,iou=iou,
                               anchor_deltas={k:b[j][k]-a[i][k] for k in POINTS
                                              if a[i][k] is not None and b[j][k] is not None}))
        for i,j,iou in pairs:
            event(i,j,"complete",iou)
        for i,j in partial:
            event(unmatched_a[i],unmatched_b[j],"partial",None)
        cases.append(dict(case=c,left_status=lx,right_status=ly,usable=usable,
                          counts=[len(a),len(b)],matches=result))
    def summarize(rows):
        valid=[r for r in rows if r["usable"]]
        totals=sum(sum(r["counts"]) for r in valid)
        events=[m for r in valid for m in r["matches"]]
        full=sum(m["kind"]=="complete" for m in events)
        result=dict(case_status_agreement=float(np.mean([r["left_status"]==r["right_status"] for r in rows])),
                    complete_f1=2*full/totals if totals else None,
                    expanded_agreement=2*len(events)/totals if totals else None)
        for mode in ("complete","partial"):
            for k in POINTS:
                error=[m["anchor_deltas"][k] for m in events if m["kind"]==mode and k in m["anchor_deltas"]]
                result[f"{mode}_{k}_mae"]=float(np.mean(np.abs(error))) if error else None
                result[f"{mode}_{k}_pairs"]=len(error)
        return result
    groups=defaultdict(list)
    for r in cases:
        groups[clusters[r["case"]]].append(r)
    n=len(groups)
    stats=summarize(cases) if cases else {}
    distribution=[]
    if n>=2:
        units=list(groups.values())
        for ix in np.random.default_rng(31041).integers(0,n,(2000,n)):
            distribution.append(summarize([row for k in ix for row in units[k]]))
    ci={}
    for k in stats:
        if not k.endswith("pairs"):
            v=[x[k] for x in distribution if x[k] is not None]
            ci[k]=dict(defined_draws=len(v),interval=np.quantile(v,[.025,.975]).tolist() if v else None)
    return dict(expected=len(clusters),left_submitted=len(left),right_submitted=len(right),both_submitted=len(seen),
                comparable=sum(r["usable"] for r in cases),clusters=n,estimates=stats,cluster_bootstrap=ci,
                case_status_pairs=dict(Counter(r["left_status"]+" × "+r["right_status"] for r in cases)),cases=cases)


def mechanism(run):
    g23=json.loads((G23/"checkpoint.json").read_text())
    verify_artifacts(G23,g23["stages"]["motion"]["artifacts"])
    verify_artifacts(G23,g23["stages"]["football"]["artifacts"])
    g24=json.loads((G24/"checkpoint.json").read_text())
    verify_artifacts(G24,g24["files"])
    folder=run/"annotation_materials"
    for source in (G24/"annotation_materials").glob("*.csv"):
        (folder/source.name).write_bytes(source.read_bytes())
    key=json.loads((G23/"annotation_key_private.json").read_text())
    roster={p:i for i,p in enumerate(sorted({v[1] for v in key.values()}),1)}
    codes={case:f"group_{roster[row[1]]:03d}" for case,row in key.items()}
    assert len(codes)==23
    atomic_json(folder/"private_cluster_map.json",codes)
    details=json.loads((G24/"design_diagnostics.json").read_text())["policy_and_outcome"]
    old=json.loads((G23/"football_results.json").read_text())
    out=old["coverage"]["2023"]["opportunities"]
    lines=["# G3B mechanism transfer feasibility · PENDING","",
           "The 23 original PNGs are preserved in the G2.3 blind packet. The G2.4 blank independent forms and case manifest are copied into this run. No human readings were received. The private case-to-player cluster map must be held by the coordinator and withheld from raters. The packet is purposive; its agreement is a diagnostic, not population reliability.",
           f"The 23 existing cases map to {len(roster)} independent player groups. A partial submission can leave cases blank. When both raters submit an event, full matching uses overlap and annotated center; unmatched pairs may enter a separately reported partial comparison only with two shared timed anchors within 0.35 s. Blank or unjudgeable cases remain separate from no event. Resample player clusters with all their cases together. Partial agreement is conditional on overlapping anchors and may select easy cases.",
           "No annotations were fabricated or derived from algorithm output. G3B will require independent path review and a representative sampling frame before any local feature or mechanism K is authorized.",
           "", "## Verified 2023-only context feasibility",
           "Null counts are reused from the G2.4 read-only query, which took only 2023-drafted WR/TE and rookie REG rows with route and separation present.",
           "", "| Football context field | Recorded / eligible rows | Intended use |","|---|---:|---|"]
    for field,num in details["context_fields"].items():
        if field=="separation_at_pass_forward":
            use="Outcome, not an input"
        elif field in ("target","pass_length"):
            use="Realized pass outcome or selection; unavailable pre-snap"
        elif field.startswith("team_coverage"):
            use="Retrospective coverage context requiring timing metadata"
        else:
            use="Candidate retrospective context"
        lines.append(f"| {field} | {num}/{details['field_denominator']} | {use} |")
    lines+=["","The common outcome uses same-row route and separation at throw. G2.3's game-window screen matches player/game/play keys, exact single ball_snap and pass_forward markers, regular 10 Hz timing, and 0.6 s support. Snap-to-throw median speed is contextual pace. Neither snap nor throw locates the route bend.",
            "", "| Criterion at ≥25 records | 2023 rookie players |", "|---|---:|"]
    for label in ("overall","out","hitch","transfer","out_transfer","hitch_transfer"):
        lines.append(f"| {label} | {out[label]['25']['players']} |")
    lines+=["","OUT and HITCH are broad class candidates. Keep SPEED_OUT_LEFT separate from SPEED_OUT_ROUTE_LEFT. Verify route origin, alignment, play-direction normalization, break depth and timing, and unique anchor support on both Combine and game paths. Trace may be clipped or contain multiple bends. Require human proximal path anchors on both sides, not task labels alone. Report route family and role coverage, clean joins, phase missingness and matched players before comparing K.",
            "Use a separately approved retrospective reference model for contextual explanation with grouped player validation and game dependence. Game fields, observed opportunities, route context and K must remain outside pre-draft R/Y predictors. Do not infer body rotation or contact mechanics from planar data."]
    (run/"mechanism_transfer_feasibility.md").write_text("\n".join(lines)+"\n")
    return dict(blind_cases=23,player_clusters=len(roster),annotations=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--private-clusters",type=Path,required=True)
    parser.add_argument("left",type=Path)
    parser.add_argument("right",type=Path)
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    with a.cases.open(newline="") as f:
        rows=list(csv.DictReader(f))
    expected={r["case"]:float(r["last_frame_s"]) for r in rows}
    clusters=json.loads(a.private_clusters.read_text())
    if not expected or len(rows)!=len(expected) or set(expected)!=set(clusters) or any(not math.isfinite(v) or v<=0 for v in expected.values()):
        parser.error("Invalid case duration or cluster mapping")
    x,left=read_partial(a.left,expected)
    y,right=read_partial(a.right,expected)
    if (x is not None and x==y) or a.output.exists():
        parser.error("Use separate reviewer identities and a new output file")
    atomic_json(a.output,agreement(left,right,clusters))


if __name__=="__main__":
    main()

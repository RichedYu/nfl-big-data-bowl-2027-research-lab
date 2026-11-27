"""Compare independent path annotations without athlete identifiers."""
import argparse
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from movement_probe import atomic_json

FIELDS = ("case", "reviewer", "review_status", "event_index", "onset_s", "bend_start_s",
          "bend_end_s", "bend_center_s", "exit_s", "direction", "confidence", "reason")
POINTS = ("onset_s", "bend_start_s", "bend_end_s", "bend_center_s", "exit_s")
STATUSES = ("events", "no_event", "unjudgeable")


def load(path, expected):
    groups = defaultdict(list)
    names = set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            def fail(reason):
                raise ValueError(f"{path.name}:{line}: {reason}")
            if any(k not in row or row[k] is None for k in FIELDS):
                fail("incomplete columns")
            case = row["case"]
            status = row["review_status"]
            if case not in expected or status not in STATUSES or not row["reviewer"].strip():
                fail("unexpected case, status or blank reviewer")
            names.add(row["reviewer"].strip())
            if status != "events":
                if any(row[k].strip() for k in (*POINTS,"event_index","direction","confidence")):
                    fail("no-event or unjudgeable must not carry event measurements")
                if status=="unjudgeable" and not row["reason"].strip():
                    fail("unjudgeable requires a reason")
                groups[case].append((status,None))
                continue
            try:
                p = {k:float(row[k]) if row[k].strip() else None for k in POINTS}
                ix = int(row["event_index"])
            except ValueError:
                fail("invalid numeric value")
            a,b,c,k,d = (p[q] for q in POINTS)
            known = [x for x in (a,b,k,c,d) if x is not None]
            if not known or ix<1 or any(not math.isfinite(x) or x<0 or x>expected[case]+1e-8 for x in known):
                fail("invalid index or time outside the recording")
            if known!=sorted(known):
                fail("onset, bend start, bend center, bend end and exit must be in order")
            if a is not None and d is not None and a>=d:
                fail("onset must precede exit")
            if len(known)<5 and not row["reason"].strip():
                fail("partial event requires a reason")
            if row["direction"] not in ("left","right","indeterminate") or row["confidence"] not in ("high","medium","low"):
                fail("invalid direction or confidence")
            groups[case].append((status,p | dict(direction=row["direction"],confidence=row["confidence"],index=ix)))
    if len(names)!=1 or set(groups)!=set(expected):
        raise ValueError("Each sheet must cover every case and identify one reviewer")
    for case,rows in groups.items():
        statuses = {r[0] for r in rows}
        if len(statuses)!=1 or (rows[0][0]!="events" and len(rows)!=1):
            raise ValueError(f"{case}: inconsistent case status")
        if rows[0][0]=="events" and len({p["index"] for _,p in rows})!=len(rows):
            raise ValueError(f"{case}: duplicate event")
    return names.pop(), groups


def pair_events(left,right):
    score=np.zeros((len(left),len(right)))
    overlap=score.copy()
    for i,x in enumerate(left):
        for j,y in enumerate(right):
            if any(p[k] is None for p in (x,y) for k in ("onset_s","exit_s","bend_center_s")):
                continue
            a,b,c,d=x["onset_s"],x["exit_s"],y["onset_s"],y["exit_s"]
            iou=max(0,min(b,d)-max(a,c))/(max(b,d)-min(a,c))
            if iou>=.20 and abs(x["bend_center_s"]-y["bend_center_s"])<=.35:
                overlap[i,j]=iou
                score[i,j]=1+min(len(left),len(right))+iou
    ia,ja=linear_sum_assignment(score,maximize=True)
    return [(int(i),int(j),float(overlap[i,j])) for i,j in zip(ia,ja) if score[i,j]>0]


def compare(left,right):
    case_rows=[]
    for case in sorted(left):
        l,r=left[case],right[case]
        ls,rs=l[0][0],r[0][0]
        a,b=[[p for s,p in rows if s=="events"] for rows in (l,r)]
        comparable=ls!="unjudgeable" and rs!="unjudgeable"
        pairs=pair_events(a,b) if comparable else []
        matched=[]
        for i,j,iou in pairs:
            shift={p:b[j][p]-a[i][p] for p in POINTS if a[i][p] is not None and b[j][p] is not None}
            matched.append(dict(left=a[i]["index"],right=b[j]["index"],iou=iou,difference_s=shift,
                               direction=[a[i]["direction"],b[j]["direction"]],
                               confidence=[a[i]["confidence"],b[j]["confidence"]]))
        case_rows.append(dict(case=case,status_left=ls,status_right=rs,comparable=comparable,
                              event_counts=[len(a),len(b)],pairs=matched,
                              unmatched_left=[x["index"] for i,x in enumerate(a) if i not in {p[0] for p in pairs}],
                              unmatched_right=[x["index"] for j,x in enumerate(b) if j not in {p[1] for p in pairs}]))
    def summarize(rows):
        use=[r for r in rows if r["comparable"]]
        num=sum(len(r["pairs"]) for r in use)
        d=sum(sum(r["event_counts"]) for r in use)
        events=[p for r in use for p in r["pairs"]]
        out=dict(status_agreement=np.mean([r["status_left"]==r["status_right"] for r in rows]),
                 matched_f1=2*num/d if d else None,
                 count_agreement=np.mean([r["event_counts"][0]==r["event_counts"][1] for r in use]) if use else None,
                 mean_iou=np.mean([e["iou"] for e in events]) if events else None)
        directions=[e["direction"] for e in events if "indeterminate" not in e["direction"]]
        out["direction_agreement"]=np.mean([a==b for a,b in directions]) if directions else None
        for key in POINTS:
            vals=[e["difference_s"][key] for e in events if key in e["difference_s"]]
            out[f"{key}_signed_mean"]=np.mean(vals) if vals else None
            out[f"{key}_mae"]=np.mean(np.abs(vals)) if vals else None
        return out
    estimate=summarize(case_rows)
    draws=[summarize([case_rows[i] for i in ix])
           for ix in np.random.default_rng(24031).integers(0,len(case_rows),(2000,len(case_rows)))]
    intervals={}
    for key in estimate:
        values=[x[key] for x in draws if x[key] is not None]
        intervals[key]={"interval":np.quantile(values,[.025,.975]).tolist() if values else None,"defined_draws":len(values)}
    return dict(cases=len(case_rows),comparable_cases=sum(r["comparable"] for r in case_rows),
                status_counts=dict(Counter(r["status_left"]+" × "+r["status_right"] for r in case_rows)),
                matched_events=sum(len(r["pairs"]) for r in case_rows),
                estimates=estimate,case_bootstrap=intervals,detail=case_rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases",required=True,type=Path)
    p.add_argument("reviewer_a",type=Path)
    p.add_argument("reviewer_b",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    with a.cases.open(newline="") as f:
        source=list(csv.DictReader(f))
    expected={r["case"]:float(r["last_frame_s"]) for r in source}
    if not expected or len(source)!=len(expected) or any(not math.isfinite(v) or v<=0 for v in expected.values()):
        p.error("invalid case manifest")
    n1,x=load(a.reviewer_a,expected)
    n2,y=load(a.reviewer_b,expected)
    if n1==n2:
        p.error("sheets must identify different reviewers")
    if a.output.exists():
        p.error("output already exists")
    atomic_json(a.output,compare(x,y))


if __name__=="__main__":
    main()

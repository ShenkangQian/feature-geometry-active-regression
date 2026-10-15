"""Compare deterministic scientific outputs, excluding wall-clock timings."""
import argparse
import csv
import json
from pathlib import Path


def compare(first, second):
    first, second=Path(first),Path(second)
    with (first/"metrics.csv").open() as f: a=list(csv.DictReader(f))
    with (second/"metrics.csv").open() as f: b=list(csv.DictReader(f))
    exclude={"fit_seconds","cumulative_fit_seconds","cumulative_acquisition_seconds"}
    equal=len(a)==len(b) and all(x.get(k)==y.get(k) for x,y in zip(a,b) for k in x if k not in exclude)
    query_equal=json.loads((first/"query_logs.json").read_text())==json.loads((second/"query_logs.json").read_text())
    return {"rows_first":len(a),"rows_second":len(b),"non_timing_fields_equal":equal,
            "query_logs_equal":query_equal,"excluded_fields":sorted(exclude)}


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("first");p.add_argument("second");args=p.parse_args()
    report=compare(args.first,args.second)
    print(json.dumps(report,indent=2))
    if not report["non_timing_fields_equal"] or not report["query_logs_equal"]: raise SystemExit(1)


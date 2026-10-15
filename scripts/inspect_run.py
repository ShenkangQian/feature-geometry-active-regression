"""Validate a completed run's artifacts without changing model results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def inspect(path):
    path = Path(path)
    completion = json.loads((path/"completion.json").read_text())
    with (path/"metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    tests = set(json.loads((path/"test_manifest.json").read_text()))
    training, validation = set(), set()
    for manifest in (path/"checkpoints").glob("*/split_manifest.json"):
        values=json.loads(manifest.read_text())
        training.update(values["train"]); validation.update(values["validation"])
    disjoint=not (training&validation or training&tests or validation&tests)
    source_root=Path(__file__).resolve().parents[1]/"src/geometa"
    hashes=json.loads((path/"source_manifest.json").read_text())
    source_match=all(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==value for name,value in hashes.items())
    result={"status":completion["status"],"metric_rows":len(rows),"train_tasks":len(training),
            "validation_tasks":len(validation),"test_tasks":len(tests),
            "split_ids_disjoint":disjoint,"methods":sorted(set(r["method"] for r in rows)),
            "query_budgets":sorted(set(int(r["queries"]) for r in rows)),
            "source_matches_current":source_match,"learning_curve_exists":(path/"learning_curves.png").exists()}
    assert len(rows)==completion["metric_rows"] and len(tests)==completion["test_tasks"] and disjoint
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("run")
    args=parser.parse_args()
    print(json.dumps(inspect(args.run),indent=2))


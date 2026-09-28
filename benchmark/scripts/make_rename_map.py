"""Generate benchmark/rename_map.json: every pre-split path -> its new path.

Input is tests/fixtures/naming_oracle.json (the frozen record of what was on
disk and what every live yaml produced under the old scheme); the labels come
from scripts/legacy_naming.py, the one deterministic rule. Nothing here is
typed by hand, and tests/test_rename_map.py checks the map is a bijection
over the oracle and that every live yaml lands where its old id is sent.

Sections (paths relative to benchmark/):

    indexes   encoder dirs, engine artifacts inside them, metagraph / mmseqs
              dirs and the sweep symlink, each with the config.json to stamp
    results   <results_root>/<experiment_id> dirs -> <root>/<enc>/<idx>/<search>,
              with the per-file renames and the search config.json
    hits      every hits parquet -> <root>/<enc>/<idx>/<search>/hits/mut<r>.parquet
    models    every distinct `model` column value -> the three label columns
    yamls     every live yaml -> its labels and paths under the new types

    uv run python scripts/make_rename_map.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH / "scripts"))
import legacy_naming as ln  # noqa: E402

ORACLE = BENCH.parent / "tests" / "fixtures" / "naming_oracle.json"
OUT = BENCH / "rename_map.json"


def _root_dataset(oracle: dict) -> dict[str, str]:
    """results root (relative) -> dataset_name, from the yamls; a root the
    yamls never wrote (…_timing, …_paper_ref) inherits its base root's."""
    by_root = {}
    for y in oracle["yamls"]:
        by_root.setdefault(y["results_dir"], y["dataset_name"])
    out = {}
    for r in sorted({e["results_root"] for e in oracle["results_dirs"] + oracle["hits_dirs"]}):
        base = re.sub(r"_(topk_hits|timing|paper_ref|paper_tables)$", "", r)
        out[r] = by_root.get(r) or by_root.get(base) or Path(base).name
    return out


def _yaml_for(oracle: dict, results_root: str, exp_id: str) -> dict | None:
    for y in oracle["yamls"]:
        if y["results_dir"] == results_root and y["experiment_id"] == exp_id:
            return y
    return None


def index_entries(oracle: dict) -> list[dict]:
    out = []
    for e in oracle["index_dirs"]:
        ds = e["dataset_root"]
        if e["kind"] == "symlink":
            # indexes/<ds>/locale/<ckpt> -> ../../<other ds>/locale/<ckpt>
            old = Path(e["path"])
            target = Path(e["target"])
            ckpt = old.name
            assert target.name == ckpt and target.parts[-2] == "locale", e
            other_ds = target.parts[-3]
            out.append({
                "kind": "symlink",
                "old": str(old),
                "new": f"{ds}/locale@{ckpt}",
                "target": f"../{other_ds}/locale@{ckpt}",
            })
            continue
        if e["kind"] == "metagraph":
            m = re.fullmatch(r"metagraph/k(\d+)", e["index_suffix"])
            out.append({
                "kind": "index", "old": e["path"], "new": f"{ds}/metagraph",
                "config": {"engine": "metagraph", "k": int(m.group(1))}, "done": e["done"],
            })
            continue
        if e["kind"] == "mmseqs":
            out.append({
                "kind": "index", "old": e["path"], "new": f"{ds}/mmseqs",
                "config": {"engine": "mmseqs"}, "done": e["done"],
            })
            continue
        # dense encoder dir: index_suffix locale/<ckpt>/<step>/<tag> | dna2vec/<tag> | llmed/<tag>
        parts = e["index_suffix"].split("/")
        if parts[0] == "locale":
            enc_id = f"locale_{parts[1]}_{parts[2]}_{parts[3]}"
        else:
            enc_id = f"{parts[0]}_{parts[1]}"
        p = ln.parse(enc_id)
        enc_label = ln.encoder_label(p)
        new_enc = f"{ds}/{enc_label}"
        out.append({
            "kind": "encoder", "old": e["path"], "new": new_enc,
            "config": ln.encoder_identity(p), "done": e["done"],
            "shards": e["shards"], "leftover_dirs": e["other_dirs"],
        })
        for eng in e["engines"]:
            if eng["engine"] == "ivfpq":
                m = re.fullmatch(r"pq(\d+)x(\d+)_L(\d+)", Path(eng["path"]).name)
                (ns,) = eng["num_shards"]
                q = {**p, "engine": "ivfpq", "pq_dim": int(m.group(1)), "pq_bits": int(m.group(2)),
                     "lists_per_shard": int(m.group(3)), "num_shards": ns}
                out.append({
                    "kind": "engine", "encoder_new": new_enc,
                    "old": eng["path"], "old_rel": str(Path(eng["path"]).relative_to(e["path"])),
                    "new": f"{new_enc}/{ln.index_label(q)}",
                    "config": ln.index_identity(q),
                    "done": eng["n_shard_files"] == ns,
                })
            elif eng["engine"] == "rabitq":
                q = {**p, "engine": "rabitq"}
                out.append({
                    "kind": "engine", "encoder_new": new_enc,
                    "old": eng["path"], "old_rel": "rabitq",
                    "new": f"{new_enc}/{ln.index_label(q)}",
                    "config": ln.index_identity(q),
                    "done": eng["complete"],
                })
            else:  # ivf: one label per nlist; centroids travel with their nlist
                for key, files in eng["files"].items():
                    m = re.fullmatch(r"(\d+)x(\d+)", key)
                    if not m:
                        continue
                    nlist, nb = int(m.group(1)), int(m.group(2))
                    q = {**p, "engine": "ivfrabitq", "nlist": nlist, "nb_bits": nb}
                    cent = eng["files"].get(f"centroids_{nlist}", [])
                    out.append({
                        "kind": "engine_files", "encoder_new": new_enc,
                        "old": eng["path"], "old_rel": "ivf",
                        "new": f"{new_enc}/{ln.index_label(q)}",
                        "files": sorted(files + cent),
                        "config": ln.index_identity(q),
                        "done": all(re.search(r"_shard_\d+_of_(\d+)\.faiss$", f) for f in files)
                        and len(files) == int(re.search(r"_of_(\d+)\.faiss$", files[0]).group(1)),
                    })
                leftovers = [f for k, v in eng["files"].items() if k.startswith("centroids_")
                             and not any(k == f"centroids_{kk.split('x')[0]}" for kk in eng["files"] if "x" in kk)
                             for f in v] + eng["other_files"]
                if leftovers:
                    out.append({"kind": "leftover_files", "old": eng["path"], "files": sorted(leftovers),
                                "note": "ivf_probe.py artifacts; left in place"})
    return out


def results_entries(oracle: dict, datasets: dict) -> list[dict]:
    out = []
    for e in oracle["results_dirs"]:
        p = ln.parse(e["id"])
        labels = ln.labels(p)
        y = _yaml_for(oracle, e["results_root"], e["id"])
        nq = y["num_queries"] if y else 1000
        seed = y["random_seed"] if y else 1337
        new_dir = "/".join([e["results_root"]] + [v for v in labels.values() if v])
        out.append({
            "old": e["path"], "new": new_dir, "labels": labels,
            "dataset": datasets[e["results_root"]],
            "config": ln.search_identity(p, num_queries=nq, random_seed=seed),
            "files": {f: ln.new_results_file(f) for f in e["files"]},
        })
    return out


def hits_entries(oracle: dict, datasets: dict) -> list[dict]:
    out = []
    for e in oracle["hits_dirs"]:
        p = ln.parse(e["id"])
        root = re.sub(r"_topk_hits$", "", e["results_root"])
        for f in e["files"]:
            m = ln.HITS_FILE_RE.match(f)
            k = int(m["k"])
            labels = ln.labels(p, top_k=k)
            exp_id = ln.format_experiment_id({**p, "top_k": k})
            y = _yaml_for(oracle, root, exp_id)
            nq = y["num_queries"] if y else 1000
            seed = y["random_seed"] if y else 1337
            search_dir = "/".join([root] + [v for v in labels.values() if v])
            out.append({
                "old": f"{e['path']}/{f}", "new": f"{search_dir}/hits/{ln.new_results_file(f)}",
                "search_dir": search_dir, "labels": labels,
                "dataset": datasets[e["results_root"]],
                "config": ln.search_identity(p, top_k=k, num_queries=nq, random_seed=seed),
            })
    return out


def model_entries(oracle: dict) -> dict:
    out = {}
    for model in oracle["model_values"]:
        p = ln.parse(model)
        out[model] = ln.labels(p)
    return out


def yaml_entries(oracle: dict) -> dict:
    out = {}
    for y in oracle["yamls"]:
        p = ln.parse(y["experiment_id"])
        labels = ln.labels(p)
        entry = {
            "labels": labels,
            "index_path": "/".join([y["index_dir"], labels["encoder"] or labels["index"]]),
            "results_path": "/".join([y["results_dir"]] + [v for v in labels.values() if v]),
        }
        if labels["encoder"]:
            entry["engine_path"] = "/".join([y["index_dir"], labels["encoder"], labels["index"]])
        out[y["yaml"]] = entry
    return out


def main():
    oracle = json.loads(ORACLE.read_text())
    datasets = _root_dataset(oracle)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=BENCH).stdout.strip()
    rename_map = {
        "_meta": {
            "generated_by": "benchmark/scripts/make_rename_map.py",
            "oracle": str(ORACLE.relative_to(BENCH.parent)),
            "oracle_commit": oracle["_meta"]["git_commit"],
            "git_commit": sha,
            "note": "paths relative to benchmark/; labels from scripts/legacy_naming.py",
        },
        "indexes": index_entries(oracle),
        "results": results_entries(oracle, datasets),
        "hits": hits_entries(oracle, datasets),
        "models": model_entries(oracle),
        "yamls": yaml_entries(oracle),
        "results_archive": [e["path"] for e in oracle.get("results_archive", [])],
    }
    OUT.write_text(json.dumps(rename_map, indent=1) + "\n")
    n = {k: len(v) for k, v in rename_map.items() if k != "_meta"}
    print(f"wrote {OUT}: {n}")


if __name__ == "__main__":
    main()

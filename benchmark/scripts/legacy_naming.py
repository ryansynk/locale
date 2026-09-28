"""The pre-split naming scheme, parsed: old experiment ids -> new labels.

Before the encoder/index split (2026-09-28) every run was named by one flat
string built in DenseConfig.__post_init__::

    <encoder>[_<engine>][top<k> | _top<k>][_bothstrands]         experiment_id
    <encoder>_<engine>[_bothstrands]                             hits_id
    locale/<ckpt>/<step>/<tag>   dna2vec/<tag>   llmed/<tag>     index_suffix
    metagraph_k<k>[_p<N>]   mmseqs

with ``<encoder> = locale_<ckpt>_<step>_<tag> | dna2vec_<tag> | llmed_<tag>``,
``<tag> = maxlen<n>_pool<p>_chunkstride`` and ``<engine>`` one of exact,
rabitq1bit, cagra, ivf<nlist>rabitq<b>_np<n>_rr<r>_qb<q>[_hnsw][_fs],
ivfpq<d>x<b>_L<l>x<s>_np<n>_rr<r>[_lut32]. A bare encoder id (no engine, no
k) was the exhaustive reference protocol.

``parse`` inverts that string exactly (``format_experiment_id(parse(x)) ==
x`` for every id in tests/fixtures/naming_oracle.json) and ``labels`` applies
the one deterministic rule that names the new directories. This module is
used by make_rename_map.py and migrate_layout.py only; the live code never
sees an old id.
"""

import re

CHUNK_OVERLAP_DEFAULT = 150  # not in the old id; every live config used it
RABITQ_SAMPLE_ROWS_DEFAULT = 2_000_000
IVF_TRAIN_ROWS_DEFAULT = 6_000_000
DEFAULT_POOLING = {"locale": "mean", "llmed": "mean", "dna2vec": "max"}
DEFAULT_MAX_SEQ_LEN = 256

_ENCODER = (
    r"(?P<name>locale|dna2vec|llmed)"
    r"(?:_(?P<ckpt>[A-Za-z0-9]+)_(?P<step>\d+))?"
    r"_maxlen(?P<maxlen>\d+)_pool(?P<pool>[a-z]+)_chunkstride"
)
_ENGINE = (
    r"(?:_(?P<engine>"
    r"exact|rabitq1bit|cagra"
    r"|ivf(?P<nlist>\d+)rabitq(?P<nb>\d+)_np(?P<inp>\d+)_rr(?P<irr>\d+)_qb(?P<qb>\d+)(?P<hnsw>_hnsw)?(?P<fs>_fs)?"
    r"|ivfpq(?P<pqd>\d+)x(?P<pqb>\d+)_L(?P<lists>\d+)x(?P<shards>\d+)_np(?P<pnp>\d+)_rr(?P<prr>\d+)(?P<lut32>_lut32)?"
    r"))?"
)
_TOPK = r"(?:_?top(?P<topk>\d+))?"
_STRANDS = r"(?P<both>_bothstrands)?"
DENSE_RE = re.compile("^" + _ENCODER + _ENGINE + _TOPK + _STRANDS + "$")
METAGRAPH_RE = re.compile(r"^metagraph_k(?P<k>\d+)(?:_p(?P<p>\d+))?$")
MMSEQS_RE = re.compile(r"^mmseqs$")
HITS_FILE_RE = re.compile(r"^raw_read_mut_(?P<rate>[0-9.]+)_topk(?P<k>\d+)\.parquet$")
RESULTS_FILE_RE = re.compile(r"^raw_read_mut_(?P<rate>[0-9.]+)\.parquet$")


def parse(exp_id: str) -> dict:
    """Old experiment/hits id -> flat dict (see module docstring). Raises on
    anything the old scheme could not have produced."""
    m = METAGRAPH_RE.match(exp_id)
    if m:
        return {"method": "metagraph", "k": int(m["k"]), "server_parallel": int(m["p"] or 1)}
    if MMSEQS_RE.match(exp_id):
        return {"method": "mmseqs"}
    m = DENSE_RE.match(exp_id)
    if not m:
        raise ValueError(f"not a pre-split experiment id: {exp_id!r}")
    g = m.groupdict()
    if (g["name"] == "locale") != (g["ckpt"] is not None):
        raise ValueError(f"checkpoint/name mismatch in {exp_id!r}")
    out = {
        "method": "dense",
        "name": g["name"],
        "ckpt": g["ckpt"],
        "step": int(g["step"]) if g["step"] else None,
        "max_seq_len": int(g["maxlen"]),
        "pooling": g["pool"],
        "engine": None,
        "top_k": int(g["topk"]) if g["topk"] else None,
        "both_strands": g["both"] is not None,
    }
    e = g["engine"]
    if e is None:
        if out["top_k"] is not None:
            raise ValueError(f"top_k without an engine in {exp_id!r}")
    elif e in ("exact", "cagra"):
        out["engine"] = e
    elif e == "rabitq1bit":
        out["engine"] = "rabitq"
    elif e.startswith("ivfpq"):
        out["engine"] = "ivfpq"
        out.update(
            pq_dim=int(g["pqd"]), pq_bits=int(g["pqb"]), lists_per_shard=int(g["lists"]),
            num_shards=int(g["shards"]), nprobe=int(g["pnp"]), rerank=int(g["prr"]),
            lut="float32" if g["lut32"] else "float16",
        )
    else:
        out["engine"] = "ivfrabitq"
        out.update(
            nlist=int(g["nlist"]), nb_bits=int(g["nb"]), nprobe=int(g["inp"]), rerank=int(g["irr"]),
            qb=int(g["qb"]), quantizer="hnsw" if g["hnsw"] else "flat", fastscan=g["fs"] is not None,
        )
    return out


def format_experiment_id(p: dict, with_top_k: bool = True) -> str:
    """Inverse of parse (with_top_k=False gives the old hits_id)."""
    if p["method"] == "metagraph":
        return f"metagraph_k{p['k']}" + (f"_p{p['server_parallel']}" if p["server_parallel"] > 1 else "")
    if p["method"] == "mmseqs":
        return "mmseqs"
    enc = p["name"] + (f"_{p['ckpt']}_{p['step']}" if p["name"] == "locale" else "")
    enc += f"_maxlen{p['max_seq_len']}_pool{p['pooling']}_chunkstride"
    strands = "_bothstrands" if p["both_strands"] else ""
    e = p["engine"]
    if e is None:
        return enc + strands
    if e == "exact":
        tag, sep = "exact", "top"
    elif e == "cagra":
        tag, sep = "cagra", "_top"
    elif e == "rabitq":
        tag, sep = "rabitq1bit", "_top"
    elif e == "ivfpq":
        tag = (f"ivfpq{p['pq_dim']}x{p['pq_bits']}_L{p['lists_per_shard']}x{p['num_shards']}"
               f"_np{p['nprobe']}_rr{p['rerank']}" + ("_lut32" if p["lut"] == "float32" else ""))
        sep = "_top"
    else:
        tag = (f"ivf{p['nlist']}rabitq{p['nb_bits']}_np{p['nprobe']}_rr{p['rerank']}_qb{p['qb']}"
               + ("_hnsw" if p["quantizer"] == "hnsw" else "") + ("_fs" if p["fastscan"] else ""))
        sep = "_top"
    if with_top_k and p["top_k"] is not None:
        return f"{enc}_{tag}{sep}{p['top_k']}{strands}"
    return f"{enc}_{tag}{strands}"


def format_index_suffix(p: dict) -> str:
    if p["method"] == "metagraph":
        return f"metagraph/k{p['k']}"
    if p["method"] == "mmseqs":
        return "mmseqs"
    tag = f"maxlen{p['max_seq_len']}_pool{p['pooling']}_chunkstride"
    if p["name"] == "locale":
        return f"locale/{p['ckpt']}/{p['step']}/{tag}"
    return f"{p['name']}/{tag}"


# --------------------------------------------------------------------------- #
# the label rule
# --------------------------------------------------------------------------- #


def encoder_label(p: dict) -> str:
    base = {"locale": f"locale@{p['ckpt']}", "dna2vec": "esa", "llmed": "llmed"}[p["name"]]
    if p["max_seq_len"] != DEFAULT_MAX_SEQ_LEN:
        base += f"-maxlen{p['max_seq_len']}"
    if p["pooling"] != DEFAULT_POOLING[p["name"]]:
        base += f"-pool{p['pooling']}"
    return base


def index_label(p: dict) -> str:
    e = p["engine"]
    if e is None:
        return "exhaustive"
    if e in ("exact", "cagra", "rabitq"):
        return e
    if e == "ivfpq":
        return f"ivfpq-pq{p['pq_dim']}x{p['pq_bits']}-L{p['lists_per_shard']}x{p['num_shards']}"
    return f"ivfrabitq-L{p['nlist']}-b{p['nb_bits']}"


def search_label(p: dict, top_k: int | None = None) -> str:
    """top<k>[-engine search tokens][-fwd]; "default" for the exhaustive
    protocol. ``top_k`` overrides the id's (hits ids carry none: it comes
    from the hits file name)."""
    e = p["engine"]
    if e is None:
        return "default" + ("" if p["both_strands"] else "-fwd")
    k = p["top_k"] if top_k is None else top_k
    if k is None:
        raise ValueError("search_label needs top_k for a top-k engine")
    parts = [f"top{k}"]
    if e == "ivfpq":
        parts += [f"np{p['nprobe']}", f"rr{p['rerank']}"]
        if p["lut"] == "float32":
            parts.append("lut32")
    elif e == "ivfrabitq":
        parts += [f"np{p['nprobe']}", f"rr{p['rerank']}", f"qb{p['qb']}"]
        if p["quantizer"] == "hnsw":
            parts.append("hnsw")
        if p["fastscan"]:
            parts.append("fs")
    if not p["both_strands"]:
        parts.append("fwd")
    return "-".join(parts)


def labels(p: dict, top_k: int | None = None) -> dict:
    if p["method"] == "metagraph":
        sl = f"p{p['server_parallel']}" if p["server_parallel"] > 1 else "default"
        return {"encoder": None, "index": "metagraph", "search": sl}
    if p["method"] == "mmseqs":
        return {"encoder": None, "index": "mmseqs", "search": "default"}
    return {
        "encoder": encoder_label(p),
        "index": index_label(p),
        "search": search_label(p, top_k),
    }


# --------------------------------------------------------------------------- #
# identities (what config.json holds), synthesized from an old id
# --------------------------------------------------------------------------- #


def encoder_identity(p: dict) -> dict:
    return {
        "name": p["name"],
        "checkpoint": p["ckpt"],
        "step": p["step"],
        "pooling": p["pooling"],
        "max_seq_len": p["max_seq_len"],
        "chunk_overlap": CHUNK_OVERLAP_DEFAULT,
    }


def index_identity(p: dict) -> dict:
    if p["method"] == "metagraph":
        return {"engine": "metagraph", "k": p["k"]}
    if p["method"] == "mmseqs":
        return {"engine": "mmseqs"}
    e = p["engine"]
    if e is None:
        return {"engine": "exhaustive"}
    if e in ("exact", "cagra"):
        return {"engine": e}
    if e == "rabitq":
        return {"engine": "rabitq", "sample_rows": RABITQ_SAMPLE_ROWS_DEFAULT}
    if e == "ivfpq":
        return {"engine": "ivfpq", "pq_dim": p["pq_dim"], "pq_bits": p["pq_bits"],
                "lists_per_shard": p["lists_per_shard"], "num_shards": p["num_shards"]}
    return {"engine": "ivfrabitq", "nlist": p["nlist"], "nb_bits": p["nb_bits"],
            "train_rows": IVF_TRAIN_ROWS_DEFAULT}


def search_identity(p: dict, top_k: int | None = None, num_queries: int = 1000,
                    random_seed: int = 1337) -> dict:
    """Search config.json contents; SEARCH field order follows src/config.py."""
    if p["method"] == "metagraph":
        base = {"engine": "metagraph", "server_parallel": p["server_parallel"]}
    elif p["method"] == "mmseqs":
        base = {"engine": "mmseqs", "max_seqs": 300}
    else:
        e = p["engine"]
        k = p["top_k"] if top_k is None else top_k
        if e is None:
            base = {"engine": "exhaustive"}
        elif e in ("exact", "rabitq", "cagra"):
            base = {"engine": e, "top_k": k}
        elif e == "ivfpq":
            base = {"engine": "ivfpq", "top_k": k, "nprobe": p["nprobe"], "rerank": p["rerank"], "lut": p["lut"]}
        else:
            base = {"engine": "ivfrabitq", "top_k": k, "nprobe": p["nprobe"], "rerank": p["rerank"],
                    "qb": p["qb"], "quantizer": p["quantizer"], "fastscan": p["fastscan"]}
        base["both_strands"] = p["both_strands"]
    return {**base, "num_queries": num_queries, "random_seed": random_seed}


def new_results_file(old_name: str) -> str:
    """raw_read_mut_0.1.parquet / raw_read_mut_0.1_topk100.parquet -> mut0.10.parquet."""
    m = HITS_FILE_RE.match(old_name) or RESULTS_FILE_RE.match(old_name)
    if not m:
        raise ValueError(f"not a pre-split results file name: {old_name!r}")
    return f"mut{float(m['rate']):.2f}.parquet"

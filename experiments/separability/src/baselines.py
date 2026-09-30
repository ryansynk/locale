"""Identity-based baselines: exact k-mers, MMseqs2 search, MMseqs2 clustering.

The probe is comparative. LOCALE having a usable radius window means nothing on
its own -- the question is whether that window is wider than what identity
clustering already achieves, especially under noise. So every baseline here
produces a score on the same [0, 1] scale as cosine, over the same units, and
goes through the same ``sweep.sweep``.

Where a choice had to be made, it was made in the baseline's favour. MMseqs2's
score here is raw ``fident`` (fraction identical over the local alignment),
not identity discounted by query coverage, because a short high-identity local
hit then counts as a full match and flatters the baseline. Since "MMseqs2
matches LOCALE" is a NO-GO verdict, being generous to MMseqs2 is the
conservative direction. ``fident * qcov`` is computed alongside and reported as
the stricter reading.

Sequences MMseqs2 finds no alignment for score 0, not NaN. That is the real
behaviour of an identity index -- a miss is a miss -- and dropping those pairs
would quietly delete exactly the fragmentation failure the probe is looking for.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# The benchmark expects `mmseqs` on PATH; on this machine it lives in a scratch
# prefix (see benchmark/src/mmseqs2_index.py and the README's external
# dependency list). MMSEQS_BIN overrides both.
_MMSEQS_FALLBACK = "/pscratch/sd/r/rsynk/mmseqs/bin/mmseqs"

K = 31  # the benchmark's k-mer size; metagraph indexes 31-mers too
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def mmseqs_bin() -> str | None:
    env = os.environ.get("MMSEQS_BIN")
    if env and Path(env).is_file():
        return env
    found = shutil.which("mmseqs")
    if found:
        return found
    if Path(_MMSEQS_FALLBACK).is_file():
        return _MMSEQS_FALLBACK
    return None


# --------------------------------------------------------------------------- #
# exact k-mers
# --------------------------------------------------------------------------- #


def canonical_kmers(seq: str, k: int = K) -> set[str]:
    """Canonical (strand-collapsed) k-mer set.

    Canonical because every identity tool this is standing in for -- metagraph,
    MMseqs2's nucleotide mode -- is strand agnostic, and because the carriers
    were extracted as CDS on the coding strand while a real read is not
    oriented. k-mers containing anything but ACGT are dropped rather than
    canonicalised, which is what a DBG index does with them.
    """
    s = seq.upper()
    out: set[str] = set()
    for i in range(len(s) - k + 1):
        km = s[i : i + k]
        if not set(km) <= {"A", "C", "G", "T"}:
            continue
        rc = km.translate(_COMPLEMENT)[::-1]
        out.add(km if km <= rc else rc)
    return out


def kmer_similarity(
    query_seqs: list[str],
    target_seqs: list[str],
    k: int = K,
    mode: str = "containment",
) -> np.ndarray:
    """(n_queries, n_targets) exact-k-mer similarity.

    ``containment`` = |Kq n Kt| / |Kq| is the right analogue of the
    max-over-windows reduction used for LOCALE: it asks how much of the query
    was found, which is what a k-mer index answers, and it does not punish a
    short query matched against a long gene. ``jaccard`` is provided for the
    symmetric reading and is dominated by length ratio when the two differ.
    """
    if mode not in ("containment", "jaccard"):
        raise ValueError(f"unknown mode {mode!r}")
    qk = [canonical_kmers(s, k) for s in query_seqs]
    tk = [canonical_kmers(s, k) for s in target_seqs]

    # All-pairs set intersection as one sparse boolean product: with a few
    # hundred sequences of a few kb each the Python double loop costs minutes,
    # and this costs a second. The vocabulary is the union of both sides, so
    # only k-mers that actually occur get a column.
    from scipy import sparse

    vocab: dict[str, int] = {}
    for s in (*qk, *tk):
        for km in s:
            if km not in vocab:
                vocab[km] = len(vocab)

    def _matrix(sets: list[set[str]]) -> "sparse.csr_matrix":
        indptr, indices = [0], []
        for s in sets:
            indices.extend(vocab[km] for km in s)
            indptr.append(len(indices))
        return sparse.csr_matrix(
            (np.ones(len(indices), dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
            shape=(len(sets), max(len(vocab), 1)),
        )

    if not vocab:
        return np.zeros((len(qk), len(tk)), dtype=np.float32)

    inter = (_matrix(qk) @ _matrix(tk).T).toarray().astype(np.float32)
    qn = np.array([len(s) for s in qk], dtype=np.float32)[:, None]
    tn = np.array([len(s) for s in tk], dtype=np.float32)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "containment":
            out = np.where(qn > 0, inter / np.maximum(qn, 1), 0.0)
        else:
            union = qn + tn - inter
            out = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# MMseqs2
# --------------------------------------------------------------------------- #


def write_fasta(path: Path, ids: list[str], seqs: list[str]) -> None:
    with open(path, "w") as f:
        for i, s in zip(ids, seqs):
            f.write(f">{i}\n{s}\n")


def _safe_ids(n: int, prefix: str) -> list[str]:
    """Opaque FASTA ids for MMseqs2.

    The probe's own ids look like ``ctxm|NG_048935.1|0``. MMseqs2 treats '|' as
    a field separator inside FASTA headers, which makes it read past the end of
    its own index -- the symptom is a stream of "getDbKey: local id >= db size"
    followed by a segfault in align2clust. Feeding it positional ids and mapping
    back afterwards sidesteps the whole question.
    """
    return [f"{prefix}{i}" for i in range(n)]


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-15:])
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}):\n{tail}")


def mmseqs_search_similarity(
    query_ids: list[str],
    query_seqs: list[str],
    target_ids: list[str],
    target_seqs: list[str],
    workdir: Path,
    sensitivity: float = 7.5,
    max_seqs: int = 20000,
    evalue: float = 1e3,
) -> tuple[np.ndarray, np.ndarray]:
    """Alignment identity of every query against every target.

    Returns (fident, fident_x_qcov), both (n_queries, n_targets), zero where no
    alignment was reported. Run permissively -- high sensitivity, a loose
    E-value and a large --max-seqs -- so a missing entry means MMseqs2 genuinely
    found nothing, not that the search was truncated.
    """
    exe = mmseqs_bin()
    if exe is None:
        raise RuntimeError("mmseqs not found; set MMSEQS_BIN")
    workdir.mkdir(parents=True, exist_ok=True)
    qfa, tfa = workdir / "q.fasta", workdir / "t.fasta"
    out, tmp = workdir / "hits.m8", workdir / "tmp"
    q_safe = _safe_ids(len(query_seqs), "q")
    t_safe = _safe_ids(len(target_seqs), "t")
    write_fasta(qfa, q_safe, query_seqs)
    write_fasta(tfa, t_safe, target_seqs)
    _run([
        exe, "easy-search", str(qfa), str(tfa), str(out), str(tmp),
        "--search-type", "3",              # nucleotide vs nucleotide
        "-s", str(sensitivity),
        "-e", str(evalue),
        "--max-seqs", str(max_seqs),
        "-a",
        "--format-output", "query,target,fident,qcov,tcov,alnlen,bits",
        "-v", "1",
    ])

    qpos = {q: i for i, q in enumerate(q_safe)}
    tpos = {t: j for j, t in enumerate(t_safe)}
    fid = np.zeros((len(query_ids), len(target_ids)), dtype=np.float32)
    fcov = np.zeros_like(fid)
    with open(out) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            q, t = parts[0], parts[1]
            if q not in qpos or t not in tpos:
                continue
            i, j = qpos[q], tpos[t]
            ident, qcov = float(parts[2]), float(parts[3])
            # Several local alignments per pair: keep the best, matching the
            # max-over-windows reduction used everywhere else in this probe.
            fid[i, j] = max(fid[i, j], ident)
            fcov[i, j] = max(fcov[i, j], ident * qcov)
    shutil.rmtree(tmp, ignore_errors=True)
    return fid, fcov


def mmseqs_all_vs_all(
    ids: list[str], seqs: list[str], workdir: Path, **kw
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Self-search: (fident, qcov, tcov), each (n, n).

    ``search_type`` 3 is nucleotide (the default), 1 is protein -- the latter is
    what the UHGP-style protein catalog point uses.
    """
    exe = mmseqs_bin()
    if exe is None:
        raise RuntimeError("mmseqs not found; set MMSEQS_BIN")
    workdir.mkdir(parents=True, exist_ok=True)
    fa = workdir / "all.fasta"
    out, tmp = workdir / "all.m8", workdir / "tmp"
    safe = _safe_ids(len(seqs), "s")
    write_fasta(fa, safe, seqs)
    _run([
        exe, "easy-search", str(fa), str(fa), str(out), str(tmp),
        "--search-type", str(kw.get("search_type", 3)),
        "-s", str(kw.get("sensitivity", 7.5)),
        "-e", str(kw.get("evalue", 1e3)),
        "--max-seqs", str(kw.get("max_seqs", 20000)),
        "-a",
        "--format-output", "query,target,fident,qcov,tcov,alnlen,bits",
        "-v", "1",
    ])
    pos = {x: i for i, x in enumerate(safe)}
    n = len(seqs)
    fid = np.zeros((n, n), dtype=np.float32)
    qc = np.zeros((n, n), dtype=np.float32)
    tc = np.zeros((n, n), dtype=np.float32)
    with open(out) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            q, t = parts[0], parts[1]
            if q not in pos or t not in pos:
                continue
            i, j = pos[q], pos[t]
            if float(parts[2]) > fid[i, j]:
                fid[i, j] = float(parts[2])
                qc[i, j] = float(parts[3])
                tc[i, j] = float(parts[4])
    shutil.rmtree(tmp, ignore_errors=True)
    return fid, qc, tc


def greedy_set_cover_clusters(
    ids: list[str], adjacency: np.ndarray
) -> dict[str, str]:
    """id -> representative, by greedy set cover over an adjacency matrix.

    This is the algorithm ``mmseqs clust --cluster-mode 0`` runs: repeatedly
    take the still-unassigned sequence with the most unassigned neighbours,
    make it a representative, and absorb its neighbourhood.

    It exists here because ``easy-cluster``/``easy-linclust`` segfault in this
    MMseqs2 build (d45e0c4) on nucleotide input: ``align2clust`` reads past the
    end of the reverse-complement-doubled database ("getDbKey: local id (548)
    >= db size (335)") and dies. ``easy-search`` is unaffected, so the
    clustering is rebuilt from the all-vs-all alignments it produces. The
    identity and coverage rules are MMseqs2's; only the graph traversal is
    ours.
    """
    n = len(ids)
    adj = np.asarray(adjacency, dtype=bool).copy()
    np.fill_diagonal(adj, True)
    assigned = np.zeros(n, dtype=bool)
    mapping: dict[str, str] = {}
    while not assigned.all():
        cand = adj[:, ~assigned][~assigned]
        degrees = cand.sum(axis=1)
        local = int(np.argmax(degrees))
        rep = int(np.nonzero(~assigned)[0][local])
        members = np.nonzero(adj[rep] & ~assigned)[0]
        for m in members:
            mapping[ids[m]] = ids[rep]
        assigned[members] = True
    return mapping


def mmseqs_cluster(
    ids: list[str],
    seqs: list[str],
    min_seq_id: float,
    workdir: Path,
    coverage: float = 0.8,
    precomputed: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, str]:
    """id -> cluster representative, at one identity threshold.

    Two sequences are linked when MMseqs2 aligns them at >= ``min_seq_id``
    fractional identity with >= ``coverage`` of *both* covered -- the
    bidirectional-coverage rule (``--cov-mode 0``), which is what stops a short
    fragment from dragging two unrelated long genes into one cluster. The link
    graph is then greedily set-covered.

    ``precomputed`` is the (fident, qcov, tcov) triple from
    ``mmseqs_all_vs_all``; pass it to cluster at several thresholds off one
    alignment run, which is the only expensive part.
    """
    fid, qc, tc = precomputed if precomputed is not None else mmseqs_all_vs_all(
        ids, seqs, workdir
    )
    adj = (fid >= min_seq_id) & (qc >= coverage) & (tc >= coverage)
    adj = adj | adj.T          # symmetrise: alignment is reported one way only
    return greedy_set_cover_clusters(ids, adj)


def catalog_capture(
    mapping: dict[str, str],
    target_ids: list[str],
    assign_sim: np.ndarray,
    query_source: np.ndarray,
    min_seq_id: float,
    unit_of_target: np.ndarray,
    unit_is_carrier: np.ndarray,
) -> dict[str, float]:
    """Capture rates of a catalog-lookup workflow, averaged over queries.

    This is how an identity catalog is actually used, and it is deliberately
    not "cluster the queries in with everything else":

      1. the *clean* reference set is clustered once -- that is the catalog;
      2. each (possibly noisy) query is assigned to the catalog by its best
         alignment, and only if that alignment clears ``min_seq_id``;
      3. the query's feature column is the cluster it landed in.

    So a clustering has no radius left to sweep at eval time -- the threshold
    was spent when the catalog was built -- and the single point it yields is
    what LOCALE's whole curve has to beat.

    Counting is done in *units*, which are whatever the caller is comparing
    against: genes for the same-gene scope, accessions for the element scope.
    ``unit_of_target`` maps each reference sequence to its unit (-1 to drop it)
    and ``unit_is_carrier`` labels the units.

    assign_sim     (n_queries, n_targets) identity of the query against each
                   clean reference, from mmseqs_search_similarity
    query_source   (n_queries,) index into target_ids of the reference the
                   query was derived from, so a query never counts its own unit
    """
    members: dict[str, list[int]] = {}
    pos = {t: i for i, t in enumerate(target_ids)}
    for m, rep in mapping.items():
        if m in pos:
            members.setdefault(rep, []).append(pos[m])

    unit_of_target = np.asarray(unit_of_target)
    unit_is_carrier = np.asarray(unit_is_carrier, dtype=bool)
    n_car = int(unit_is_carrier.sum())
    n_con = int((~unit_is_carrier).sum())
    car_rates, con_rates, assigned = [], [], 0

    for qi in range(assign_sim.shape[0]):
        row = assign_sim[qi]
        src = int(query_source[qi])
        q_unit = int(unit_of_target[src]) if src >= 0 else -1
        best = int(np.argmax(row))
        if row[best] < min_seq_id:
            # Unassigned: the read falls outside the catalog entirely. This is
            # the fragmentation failure mode and it must score 0, not be
            # dropped from the average.
            car_rates.append(0.0)
            con_rates.append(0.0)
            continue
        assigned += 1
        rep = mapping.get(target_ids[best], target_ids[best])
        units = {
            int(unit_of_target[m]) for m in members.get(rep, [])
            if unit_of_target[m] >= 0
        } - {q_unit}
        q_is_car = q_unit >= 0 and unit_is_carrier[q_unit]
        denom_car = n_car - (1 if q_is_car else 0)
        car_rates.append(
            sum(1 for u in units if unit_is_carrier[u]) / denom_car
            if denom_car else 0.0
        )
        con_rates.append(
            sum(1 for u in units if not unit_is_carrier[u]) / n_con
            if n_con else 0.0
        )

    car_units = np.nonzero(unit_is_carrier)[0]
    reps_with_carriers: dict[str, set[int]] = {}
    for rep, mem in members.items():
        us = {int(unit_of_target[m]) for m in mem if unit_of_target[m] >= 0}
        cs = {u for u in us if unit_is_carrier[u]}
        if cs:
            reps_with_carriers[rep] = cs
    largest = max((len(v) for v in reps_with_carriers.values()), default=0)
    return {
        "min_seq_id": min_seq_id,
        "carrier_capture": float(np.mean(car_rates)) if car_rates else 0.0,
        "confusable_capture": float(np.mean(con_rates)) if con_rates else 0.0,
        "carrier_capture_median": float(np.median(car_rates)) if car_rates else 0.0,
        "assigned_fraction": assigned / max(assign_sim.shape[0], 1),
        "n_carrier_clusters": len(reps_with_carriers),
        "largest_carrier_cluster_share": largest / max(len(car_units), 1),
    }


def translate_cds(seq: str) -> str:
    """Naive frame-1 translation, stops stripped.

    Only used for the protein-level clustering point, which stands in for how
    UHGP is actually built (MMseqs2 linclust over predicted proteins). It is
    frame-1 and therefore meaningless once indels have shifted the frame --
    which is the point being made when the protein baseline collapses under
    the noise axis.
    """
    from Bio.Seq import Seq

    trimmed = seq[: len(seq) - len(seq) % 3]
    aa = str(Seq(trimmed).translate(to_stop=False))
    return aa.replace("*", "X")


def scratch_dir(tag: str) -> Path:
    base = Path(os.environ.get("PROBE_TMP", tempfile.gettempdir()))
    d = base / f"sep_probe_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d

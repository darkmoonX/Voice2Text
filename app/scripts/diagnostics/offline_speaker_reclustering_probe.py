"""Round 0038 Phase A — offline speaker re-clustering headroom probe.

Reproduces the DIRECT path's global diarization cheaply (pyannote-3.1 only, no ASR/align),
extracts a PER-SEGMENT ECAPA embedding for every diarization segment, then asks whether a
quality-scored offline re-clustering can do better than raw pyannote-global + the blind
`reconcile_threshold` (0.52) WITHOUT merging close real speakers.

Numpy-only analysis (no scipy/sklearn dependency):
  - cosine silhouette of any labeling over the per-segment embedding cloud
  - average-linkage agglomerative re-clustering, cut at each K -> (K, silhouette) curve
  - split detection: 2-way split of each pyannote cluster's own embeddings + temporal check
  - merge detection vs what blind reconcile@0.52 would merge

Usage (from app/src, venv python):
  ..\.venv\Scripts\python.exe ..\scripts\diagnostics\offline_speaker_reclustering_probe.py \
      --input ..\src\tests\compare_whisperx_test\input\YT_aXqBRYQSGp0 \
      --device cuda --out <report_dir>

This is a measurement tool only; it ships nothing and touches no persistent profile store.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make the product package importable when run from app/scripts/diagnostics.
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voice2text.config import RuntimeConfig  # noqa: E402
from voice2text.pipeline.direct_transcription import (  # noqa: E402
    decode_to_wav_16k_mono,
    read_wav,
)
from voice2text.stt.factory import create_stt_transcriber  # noqa: E402

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# numpy clustering / quality helpers
# --------------------------------------------------------------------------- #
def _l2norm(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return mat / norm


def cosine_distance_matrix(emb: np.ndarray) -> np.ndarray:
    unit = _l2norm(emb)
    sim = np.clip(unit @ unit.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over a precomputed distance matrix. Singletons score 0."""
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    n = len(labels)
    if len(uniq) < 2 or n < 3:
        return 0.0
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        own = labels[i]
        own_mask = labels == own
        own_mask[i] = False
        if own_mask.sum() == 0:
            scores[i] = 0.0
            continue
        a = dist[i, own_mask].mean()
        b = np.inf
        for other in uniq:
            if other == own:
                continue
            m = labels == other
            if m.sum() == 0:
                continue
            b = min(b, dist[i, m].mean())
        scores[i] = 0.0 if max(a, b) < 1e-12 else (b - a) / max(a, b)
    return float(scores.mean())


def average_linkage_labels(dist: np.ndarray, k: int) -> np.ndarray:
    """Average-linkage agglomerative clustering cut at exactly k clusters."""
    n = dist.shape[0]
    k = max(1, min(k, n))
    members: list[list[int]] = [[i] for i in range(n)]
    # cluster-cluster average distance, lazily recomputed on merge
    active = list(range(n))
    cdist: dict[tuple[int, int], float] = {}
    for ai in range(len(active)):
        for bi in range(ai + 1, len(active)):
            a, b = active[ai], active[bi]
            cdist[(a, b)] = float(dist[a, b])
    next_id = n
    while len(active) > k:
        best = None
        best_pair = None
        for ai in range(len(active)):
            for bi in range(ai + 1, len(active)):
                a, b = active[ai], active[bi]
                key = (a, b) if a < b else (b, a)
                d = cdist[key]
                if best is None or d < best:
                    best = d
                    best_pair = (a, b)
        a, b = best_pair
        new_members = members[a] + members[b] if a < n else None
        # rebuild members via dict to support synthetic ids
        ma = _members_of(members, a, n)
        mb = _members_of(members, b, n)
        merged = ma + mb
        new = next_id
        next_id += 1
        members.append(merged)
        active = [c for c in active if c not in (a, b)]
        for c in active:
            mc = _members_of(members, c, n)
            d = float(dist[np.ix_(merged, mc)].mean())
            key = (c, new) if c < new else (new, c)
            cdist[key] = d
        active.append(new)
    labels = np.full(n, -1, dtype=int)
    for lab, c in enumerate(active):
        for idx in _members_of(members, c, n):
            labels[idx] = lab
    return labels


def _members_of(members: list[list[int]], cid: int, n: int) -> list[int]:
    if cid < n:
        return [cid]
    return members[cid]


def two_way_split(dist_sub: np.ndarray) -> np.ndarray:
    return average_linkage_labels(dist_sub, 2)


# --------------------------------------------------------------------------- #
# diarization + embedding extraction
# --------------------------------------------------------------------------- #
def _build_transcriber(device: str, store_path: Path):
    cfg = RuntimeConfig()
    cfg.model_device = device
    cfg.stt_variant = "auto"
    cfg.whisperx_enable_diarization = True
    cfg.whisperx_enable_forced_alignment = False  # not needed; we never run ASR/align
    cfg.whisperx_speaker_profile_enabled = True
    cfg.whisperx_speaker_profile_backend = "pyannote"  # engine backend unused; embeddings via wespeaker below
    cfg.whisperx_diarization_device = device
    cfg.whisperx_speaker_profile_store_path = str(store_path)

    def _status(msg: str) -> None:
        print(f"[probe] {msg}", flush=True)

    return create_stt_transcriber(cfg, progress_callback=_status), cfg


def _load_wespeaker_inference(transcriber, device: str):
    """Load pyannote wespeaker embedding (the model diarization-3.1 uses; NOT HF-gated).

    speechbrain ECAPA cannot be installed alongside this whisperx/pyannote build (its
    k2_fsa lazy import breaks the diarization pipeline), and pyannote/embedding is HF-gated
    for this user. wespeaker-voxceleb-resnet34-lm is already cached as a diarization dep and
    is the same embedding the diarizer clusters on, so it is the faithful choice.
    """
    from pyannote.audio import Inference, Model  # type: ignore
    import torch  # type: ignore

    model_root = transcriber._model_root
    local_bin = model_root / "diarization_deps" / "pyannote-wespeaker-voxceleb-resnet34-lm" / "pytorch_model.bin"
    token = transcriber._resolve_hf_token() or None
    refs = []
    if local_bin.exists():
        refs.append(str(local_bin))
    refs.append("pyannote/wespeaker-voxceleb-resnet34-lm")
    model = None
    errors = []
    for ref in refs:
        for kwargs in ({"token": token}, {"use_auth_token": token}, {}):
            try:
                active = {k: v for (k, v) in kwargs.items() if v}
                model = Model.from_pretrained(ref, **active)
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{ref}: {exc}")
        if model is not None:
            break
    if model is None:
        raise RuntimeError("wespeaker embedding load failed: " + " | ".join(errors[:3]))
    dev = "cuda" if (str(device).lower().startswith("cuda") and torch.cuda.is_available()) else "cpu"
    return Inference(model, window="whole", device=torch.device(dev))


def _diarize_segments(transcriber, audio_f32: np.ndarray) -> list[dict]:
    transcriber._ensure_diarization_pipeline_loaded()
    result = transcriber._diarization_pipeline(audio_f32)
    rows: list[dict] = []
    # whisperx DiarizationPipeline returns a pandas DataFrame
    try:
        import pandas as pd  # type: ignore

        if isinstance(result, pd.DataFrame):
            for _, r in result.iterrows():
                rows.append(
                    {
                        "start": float(r.get("start")),
                        "end": float(r.get("end")),
                        "speaker": str(r.get("speaker")),
                    }
                )
            return rows
    except Exception:
        pass
    # fallback: list of dicts / pyannote Annotation
    if isinstance(result, list):
        for r in result:
            rows.append(
                {
                    "start": float(r.get("start")),
                    "end": float(r.get("end")),
                    "speaker": str(r.get("speaker") or r.get("label")),
                }
            )
    return rows


def _segment_embeddings(transcriber, audio_f32: np.ndarray, segments: list[dict], *, min_seconds: float, device: str):
    import torch  # type: ignore

    inference = _load_wespeaker_inference(transcriber, device)
    emb: list[np.ndarray] = []
    kept: list[dict] = []
    min_samples = int(min_seconds * SAMPLE_RATE)
    for seg in segments:
        s = max(0, int(round(seg["start"] * SAMPLE_RATE)))
        e = min(int(audio_f32.size), int(round(seg["end"] * SAMPLE_RATE)))
        if e - s < min_samples:
            continue
        clip = np.ascontiguousarray(audio_f32[s:e], dtype=np.float32)
        waveform = torch.from_numpy(clip).unsqueeze(0)
        vec = None
        for payload in (
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
            {"waveform": waveform, "sampling_rate": SAMPLE_RATE},
        ):
            try:
                value = inference(payload)
                vec = np.asarray(value, dtype=np.float64).reshape(-1)
                break
            except Exception:
                continue
        if vec is None or vec.size == 0 or not np.isfinite(vec).all():
            continue
        emb.append(vec)
        kept.append(seg)
    if not emb:
        return np.zeros((0, 0)), []
    return np.vstack(emb), kept


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def _speaker_centroids(emb: np.ndarray, segs: list[dict]) -> dict[str, np.ndarray]:
    by: dict[str, list[tuple[float, np.ndarray]]] = {}
    for i, seg in enumerate(segs):
        dur = max(0.0, seg["end"] - seg["start"])
        by.setdefault(seg["speaker"], []).append((dur, emb[i]))
    out: dict[str, np.ndarray] = {}
    for spk, items in by.items():
        w = np.array([d for (d, _) in items], dtype=np.float64)
        w = w / max(w.sum(), 1e-9)
        cen = np.zeros_like(items[0][1])
        for (wi, (_, v)) in zip(w, items):
            cen = cen + wi * v
        out[spk] = cen
    return out


def _blind_reconcile_merges(centroids: dict[str, np.ndarray], threshold: float) -> list[dict]:
    """Replay reconcile_similar_profiles: greedily merge the most-similar pair >= threshold."""
    ids = list(centroids.keys())
    cents = {k: v.copy() for (k, v) in centroids.items()}
    weights = {k: 1.0 for k in ids}
    merges: list[dict] = []
    active = set(ids)
    changed = True
    while changed:
        changed = False
        best = None
        for a in active:
            ua = cents[a] / max(np.linalg.norm(cents[a]), 1e-12)
            for b in active:
                if b <= a:
                    continue
                ub = cents[b] / max(np.linalg.norm(cents[b]), 1e-12)
                sim = float(np.dot(ua, ub))
                if sim < threshold:
                    continue
                if best is None or sim > best[2]:
                    best = (a, b, sim)
        if best is None:
            break
        a, b, sim = best
        merges.append({"from": b, "to": a, "similarity": round(sim, 4)})
        cents[a] = cents[a] * weights[a] + cents[b] * weights[b]
        weights[a] += weights[b]
        active.discard(b)
        changed = True
    return merges


def analyze(emb: np.ndarray, segs: list[dict], reconcile_threshold: float) -> dict:
    n = emb.shape[0]
    dist = cosine_distance_matrix(emb)
    pa_speakers = sorted({s["speaker"] for s in segs})
    pa_index = {spk: i for (i, spk) in enumerate(pa_speakers)}
    pa_labels = np.array([pa_index[s["speaker"]] for s in segs])
    k_pa = len(pa_speakers)
    pa_sil = silhouette(dist, pa_labels)

    # re-cluster K sweep
    sweep = []
    for k in range(2, min(n, max(k_pa + 3, 6)) + 1):
        labels = average_linkage_labels(dist, k)
        sweep.append({"k": k, "silhouette": round(silhouette(dist, labels), 4)})
    best = max(sweep, key=lambda r: r["silhouette"]) if sweep else {"k": k_pa, "silhouette": pa_sil}

    # split detection per pyannote speaker
    splits = []
    for spk in pa_speakers:
        idx = [i for i, s in enumerate(segs) if s["speaker"] == spk]
        if len(idx) < 4:
            continue
        sub = dist[np.ix_(idx, idx)]
        sub_labels = two_way_split(sub)
        sub_sil = silhouette(sub, sub_labels)
        # temporal distinctness of the two sub-clusters
        mids = np.array([0.5 * (segs[i]["start"] + segs[i]["end"]) for i in idx])
        t0 = mids[sub_labels == 0]
        t1 = mids[sub_labels == 1]
        if len(t0) and len(t1):
            overlap = not (t0.max() < t1.min() or t1.max() < t0.min())
        else:
            overlap = True
        splits.append(
            {
                "speaker": spk,
                "n_segments": len(idx),
                "sub_silhouette": round(sub_sil, 4),
                "sub_sizes": [int((sub_labels == 0).sum()), int((sub_labels == 1).sum())],
                "temporally_overlapping": bool(overlap),
                "split_candidate": bool(sub_sil >= 0.25 and min((sub_labels == 0).sum(), (sub_labels == 1).sum()) >= 2),
            }
        )

    # blind reconcile@threshold vs quality
    centroids = _speaker_centroids(emb, segs)
    blind_merges = _blind_reconcile_merges(centroids, reconcile_threshold)
    # would those merges raise or lower silhouette?
    blind_sil = pa_sil
    if blind_merges:
        remap = {}
        for m in blind_merges:
            remap[m["from"]] = m["to"]
        # resolve chains
        def root(x):
            while x in remap:
                x = remap[x]
            return x

        merged_labels = np.array([pa_index_root(root(s["speaker"]), pa_index) for s in segs])
        blind_sil = silhouette(dist, merged_labels)

    return {
        "n_segments": int(n),
        "pyannote_k": int(k_pa),
        "pyannote_silhouette": round(pa_sil, 4),
        "recluster_sweep": sweep,
        "recluster_best": best,
        "split_candidates": splits,
        "blind_reconcile_threshold": float(reconcile_threshold),
        "blind_reconcile_merges": blind_merges,
        "blind_reconcile_silhouette": round(blind_sil, 4),
        "pyannote_speakers": {spk: int(sum(1 for s in segs if s["speaker"] == spk)) for spk in pa_speakers},
    }


def pa_index_root(spk: str, pa_index: dict[str, int]) -> int:
    # merged labels: collapse to the kept speaker's index; unseen -> stable hash bucket
    if spk in pa_index:
        return pa_index[spk]
    return abs(hash(spk)) % 100000


def main() -> int:
    ap = argparse.ArgumentParser(description="Round 0038 Phase A offline re-clustering probe")
    ap.add_argument("--input", required=True, help="case folder containing voice.m4a (or a wav/m4a file)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-seconds", type=float, default=0.5, help="min diarization segment length to embed")
    ap.add_argument("--reconcile-threshold", type=float, default=0.52)
    ap.add_argument("--out", default="", help="report dir (default: alongside input)")
    args = ap.parse_args()

    in_path = Path(args.input)
    audio_path = in_path
    if in_path.is_dir():
        for name in ("voice.m4a", "voice.wav", "audio.wav"):
            if (in_path / name).exists():
                audio_path = in_path / name
                break
    print(f"[probe] audio = {audio_path}", flush=True)

    store_path = Path(args.out or ".") / "_probe_speaker_profiles.json"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    transcriber, cfg = _build_transcriber(args.device, store_path)

    wav = decode_to_wav_16k_mono(audio_path, ffmpeg_dir=cfg.ffmpeg_dll_dir)
    chunk = read_wav(Path(wav))
    audio_f32 = np.frombuffer(chunk.pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if chunk.channels > 1:
        audio_f32 = audio_f32.reshape(-1, chunk.channels).mean(axis=1)
    print(f"[probe] audio seconds = {audio_f32.size / SAMPLE_RATE:.1f}", flush=True)

    segments = _diarize_segments(transcriber, audio_f32)
    print(f"[probe] pyannote segments = {len(segments)}", flush=True)
    emb, kept = _segment_embeddings(transcriber, audio_f32, segments, min_seconds=args.min_seconds, device=args.device)
    print(f"[probe] embedded segments = {len(kept)}", flush=True)
    if emb.shape[0] < 3:
        print("[probe] too few embedded segments to cluster", flush=True)
        return 1

    report = analyze(emb, kept, args.reconcile_threshold)
    report["case"] = in_path.name
    report["audio_seconds"] = round(audio_f32.size / SAMPLE_RATE, 1)

    out_dir = Path(args.out or in_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "offline_recluster_probe.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n==== Round 0038 Phase A — offline re-clustering probe ====", flush=True)
    print(f"case={report['case']} audio={report['audio_seconds']}s segments={report['n_segments']}", flush=True)
    print(f"pyannote: K={report['pyannote_k']} silhouette={report['pyannote_silhouette']}", flush=True)
    print(f"  speakers={report['pyannote_speakers']}", flush=True)
    print(f"recluster best: K={report['recluster_best']['k']} silhouette={report['recluster_best']['silhouette']}", flush=True)
    print(f"  sweep={report['recluster_sweep']}", flush=True)
    print(f"blind reconcile@{report['blind_reconcile_threshold']}: merges={report['blind_reconcile_merges']} -> silhouette={report['blind_reconcile_silhouette']}", flush=True)
    print("split candidates:", flush=True)
    for sc in report["split_candidates"]:
        print(f"  {sc}", flush=True)
    print(f"\n[probe] report -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

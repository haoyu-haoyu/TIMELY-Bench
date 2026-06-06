#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier


TASK_SPECS = {
    "AKI-T1": {"condition": "aki"},
    "AKI-S1": {"condition": "aki"},
    "DEL-T1": {"condition": "delirium"},
    "DEL-S1": {"condition": "delirium"},
    "SEP-T1": {"condition": "sepsis"},
    "SEP-S1": {"condition": "sepsis"},
    "S-T1": {"condition": "stroke"},
}
SEQ_MODELS = ("bilstm_attention", "temporal_transformer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6.5A baselines for CRES v3")
    p.add_argument("--mode", choices=["xgb", "seq", "merge"], required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--output-dir", default="results/cres_v3/phase65a_baselines")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--task-filter", nargs="*", default=sorted(TASK_SPECS))
    p.add_argument("--seq-model", choices=["all", *SEQ_MODELS], default="all")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=96)
    p.add_argument("--transformer-lr", type=float, default=1e-4)
    p.add_argument("--transformer-hidden-size", type=int, default=64)
    p.add_argument("--transformer-epochs", type=int, default=5)
    p.add_argument("--transformer-warmup-ratio", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=168)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0:
        return 0.5, 0.0
    if np.unique(y_true).size < 2:
        return 0.5, float(np.mean(y_true))
    return float(roc_auc_score(y_true, y_prob)), float(average_precision_score(y_true, y_prob))


def filtered_read(path: Path, columns: List[str], task_id: str | None = None) -> pd.DataFrame:
    kwargs = {"columns": columns}
    if task_id is not None:
        kwargs["filters"] = [("task_id", "==", task_id)]
    return pd.read_parquet(path, **kwargs)


def resolve_manifest_path(root: Path, raw_path: str | Path) -> Path:
    raw = Path(str(raw_path))
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        if "TIMELY-Bench_Final" in raw.parts:
            idx = raw.parts.index("TIMELY-Bench_Final")
            candidates.append(root.joinpath(*raw.parts[idx + 1 :]))
    else:
        candidates.append(root / raw)

    expanded: List[Path] = []
    for candidate in candidates:
        expanded.append(candidate)
        if str(candidate).endswith(".parquet"):
            expanded.append(Path(str(candidate) + ".parts"))
        if str(candidate).endswith(".parquet.parts"):
            expanded.append(Path(str(candidate)[: -len(".parts")]))

    seen = set()
    unique_candidates = []
    for candidate in expanded:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(str(x) for x in unique_candidates)
    raise FileNotFoundError(f"Could not resolve manifest path from {raw_path!r}. Tried:\n{tried}")


def load_branch_manifest(root: Path, branch: str, task_id: str) -> pd.DataFrame:
    path = root / "data" / "processed" / "v3" / "cres" / f"cres_{branch}_manifest.parquet"
    cols = [
        "instance_id",
        "condition",
        "task_id",
        "task_mode",
        "stay_id",
        "subject_id",
        "anchor_hour",
        "primary_label_binary",
        "representation_profile",
    ]
    if branch == "A":
        cols += ["representation_table_path"]
    else:
        cols += ["b1_anchor_index_path", "b1_bank_path"]
    df = filtered_read(path, cols, task_id=task_id)
    df = df.dropna(subset=["primary_label_binary"]).copy()
    df["primary_label_binary"] = df["primary_label_binary"].astype(np.int8)
    df["stay_id"] = df["stay_id"].astype(np.int64)
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").fillna(-1).astype(np.int64)
    df["anchor_hour"] = pd.to_numeric(df["anchor_hour"], errors="coerce").fillna(0).astype(np.int16)
    return df


def load_a_table(root: Path, rel_path: str) -> pd.DataFrame:
    path = resolve_manifest_path(root, rel_path)
    df = pd.read_parquet(path)
    df["stay_id"] = df["stay_id"].astype(np.int64)
    df["anchor_hour"] = pd.to_numeric(df["anchor_hour_requested"], errors="coerce").fillna(0).astype(np.int16)
    return df


def build_a_dataset(root: Path, task_id: str) -> pd.DataFrame:
    manifest = load_branch_manifest(root, "A", task_id)
    rep_path = manifest["representation_table_path"].iloc[0]
    features = load_a_table(root, rep_path)
    data = manifest.merge(features, on=["stay_id", "anchor_hour"], how="inner", validate="many_to_one")
    return data


def pick_a_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "instance_id",
        "condition_x",
        "condition_y",
        "task_id",
        "task_mode",
        "stay_id",
        "subject_id",
        "anchor_hour",
        "primary_label_binary",
        "representation_profile",
        "representation_table_path",
        "anchor_hour_requested",
        "source_task_ids",
        "condition",
        "representation_id",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_fold_assignments(df: pd.DataFrame, n_folds: int, random_state: int) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    X_dummy = np.zeros(len(df), dtype=np.float32)
    y = df["primary_label_binary"].astype(np.int8).to_numpy()
    groups = df["subject_id"].astype(np.int64).to_numpy()
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_dummy, y, groups), start=1):
        yield fold, tr_idx, va_idx


def run_xgb(root: Path, args: argparse.Namespace, output_dir: Path) -> None:
    rows = []
    for task_id in args.task_filter:
        data = build_a_dataset(root, task_id)
        feature_cols = pick_a_feature_columns(data)
        X = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).to_numpy()
        y = data["primary_label_binary"].astype(np.int8).to_numpy()
        for fold, tr_idx, va_idx in build_fold_assignments(data, args.n_folds, args.random_state):
            y_tr = y[tr_idx]
            pos = float((y_tr == 1).sum())
            neg = float((y_tr == 0).sum())
            scale_pos_weight = (neg / pos) if pos > 0 else 1.0
            model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=args.random_state,
                n_jobs=max(1, args.n_jobs),
                scale_pos_weight=scale_pos_weight,
                verbosity=0,
            )
            model.fit(X[tr_idx], y_tr)
            y_prob = model.predict_proba(X[va_idx])[:, 1]
            auroc, auprc = safe_metrics(y[va_idx], y_prob)
            rows.append(
                {
                    "task_id": task_id,
                    "condition": TASK_SPECS[task_id]["condition"],
                    "branch": "A",
                    "model": "xgboost",
                    "fold": fold,
                    "train_rows": int(len(tr_idx)),
                    "val_rows": int(len(va_idx)),
                    "val_positive_rate": float(y[va_idx].mean()),
                    "auroc": auroc,
                    "auprc": auprc,
                    "n_features": len(feature_cols),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "xgb_fold_metrics.csv", index=False)


@dataclass
class SequenceStore:
    feature_cols: List[str]
    data: Dict[int, Tuple[np.ndarray, np.ndarray]]


def load_sequence_store(root: Path, bank_rel_path: str) -> SequenceStore:
    bank = pd.read_parquet(resolve_manifest_path(root, bank_rel_path))
    bank = bank.sort_values(["stay_id", "sequence_hour"], kind="mergesort")
    feature_cols = []
    for c in bank.columns:
        if c in {"condition", "stay_id", "sequence_hour", "hadm_id"}:
            continue
        if pd.api.types.is_numeric_dtype(bank[c]):
            feature_cols.append(c)
    bank[feature_cols] = bank[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for stay_id, g in bank.groupby("stay_id", sort=False):
        hours = g["sequence_hour"].astype(np.int16).to_numpy()
        arr = g[feature_cols].to_numpy(dtype=np.float32, copy=True)
        data[int(stay_id)] = (hours, arr)
    return SequenceStore(feature_cols=feature_cols, data=data)


def build_b1_dataset(root: Path, task_id: str) -> Tuple[pd.DataFrame, str, str]:
    manifest = load_branch_manifest(root, "B1", task_id)
    anchor_rel = manifest["b1_anchor_index_path"].iloc[0]
    bank_rel = manifest["b1_bank_path"].iloc[0]
    anchor = pd.read_parquet(resolve_manifest_path(root, anchor_rel))
    anchor["stay_id"] = anchor["stay_id"].astype(np.int64)
    anchor["anchor_hour"] = pd.to_numeric(anchor["anchor_hour_requested"], errors="coerce").fillna(0).astype(np.int16)
    anchor["history_end_hour"] = pd.to_numeric(anchor["history_end_hour"], errors="coerce").fillna(0).astype(np.int16)
    data = manifest.merge(anchor[["stay_id", "anchor_hour", "history_end_hour"]], on=["stay_id", "anchor_hour"], how="inner", validate="many_to_one")
    return data, anchor_rel, bank_rel


def get_torch():
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset

    # Avoid exhausting file descriptors when many workers share tensors.
    torch.multiprocessing.set_sharing_strategy("file_system")

    class SeqDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, store: SequenceStore, max_seq_len: int):
            self.frame = frame.reset_index(drop=True)
            self.store = store
            self.max_seq_len = max_seq_len

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, idx: int):
            row = self.frame.iloc[idx]
            hours, arr = self.store.data[int(row.stay_id)]
            pos = np.searchsorted(hours, int(row.history_end_hour), side="right")
            seq = arr[:pos]
            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len :]
            if len(seq) == 0:
                seq = np.zeros((1, arr.shape[1]), dtype=np.float32)
            return (
                torch.from_numpy(seq.astype(np.float32)),
                torch.tensor(float(row.primary_label_binary), dtype=torch.float32),
            )

    def collate(batch):
        seqs, labels = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        padded = pad_sequence(seqs, batch_first=True)
        labels_t = torch.stack(labels)
        return padded, lengths, labels_t

    class BiLSTMAttention(nn.Module):
        def __init__(self, input_dim: int, hidden_size: int):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True, bidirectional=True)
            self.attn = nn.Linear(hidden_size * 2, 1)
            self.out = nn.Linear(hidden_size * 2, 1)

        def forward(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            max_len = out.size(1)
            mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
            scores = self.attn(out).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(out * weights.unsqueeze(-1), dim=1)
            return self.out(pooled).squeeze(-1)

    class TemporalTransformer(nn.Module):
        def __init__(self, input_dim: int, hidden_size: int, max_len: int):
            super().__init__()
            self.proj = nn.Linear(input_dim, hidden_size)
            self.pos = nn.Embedding(max_len, hidden_size)
            enc = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc, num_layers=2)
            self.out = nn.Linear(hidden_size, 1)

        def forward(self, x, lengths):
            bsz, seq_len, _ = x.shape
            pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
            h = self.proj(x) + self.pos(pos_idx)
            pad_mask = torch.arange(seq_len, device=x.device)[None, :] >= lengths[:, None]
            h = self.encoder(h, src_key_padding_mask=pad_mask)
            mask = (~pad_mask).unsqueeze(-1)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return self.out(pooled).squeeze(-1)

    return torch, nn, DataLoader, SeqDataset, collate, BiLSTMAttention, TemporalTransformer


def train_seq_model(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, store: SequenceStore, args: argparse.Namespace) -> Tuple[float, float]:
    torch, nn, DataLoader, SeqDataset, collate, BiLSTMAttention, TemporalTransformer = get_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SeqDataset(train_df, store, args.max_seq_len)
    val_ds = SeqDataset(val_df, store, args.max_seq_len)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    input_dim = len(store.feature_cols)
    if model_name == "bilstm_attention":
        model = BiLSTMAttention(input_dim, args.hidden_size)
        epochs = args.epochs
        lr = args.lr
    else:
        model = TemporalTransformer(input_dim, args.transformer_hidden_size, args.max_seq_len)
        epochs = args.transformer_epochs
        lr = args.transformer_lr
    model.to(device)

    pos = float(train_df["primary_label_binary"].sum())
    neg = float(len(train_df) - pos)
    pos_weight = torch.tensor([(neg / pos) if pos > 0 else 1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = max(1, int(total_steps * args.transformer_warmup_ratio)) if model_name == "temporal_transformer" else 0

    def lr_lambda(current_step: int) -> float:
        if model_name != "temporal_transformer":
            return 1.0
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - current_step) / float(remaining))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best = (-1.0, -1.0)
    best_auroc = -1.0
    patience = 1
    bad_epochs = 0
    for _epoch in range(epochs):
        model.train()
        for x, lengths, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        probs = []
        ys = []
        with torch.no_grad():
            for x, lengths, y in val_loader:
                x = x.to(device)
                lengths = lengths.to(device)
                logits = model(x, lengths)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(y.numpy())
        y_true = np.concatenate(ys)
        y_prob = np.nan_to_num(np.concatenate(probs), nan=0.5, posinf=1.0, neginf=0.0)
        auroc, auprc = safe_metrics(y_true, y_prob)
        if auroc > best_auroc:
            best = (auroc, auprc)
            best_auroc = auroc
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                break
    return best


def run_seq(root: Path, args: argparse.Namespace, output_dir: Path) -> None:
    rows = []
    task_groups: Dict[str, List[str]] = {}
    for task_id in args.task_filter:
        task_groups.setdefault(TASK_SPECS[task_id]["condition"], []).append(task_id)

    seq_models = SEQ_MODELS if args.seq_model == "all" else (args.seq_model,)
    for condition, task_ids in task_groups.items():
        # Use the first task to discover shared B1 bank for the condition.
        _, _anchor_rel, bank_rel = build_b1_dataset(root, task_ids[0])
        store = load_sequence_store(root, bank_rel)
        for task_id in task_ids:
            data, _anchor_rel, _bank_rel = build_b1_dataset(root, task_id)
            for fold, tr_idx, va_idx in build_fold_assignments(data, args.n_folds, args.random_state):
                train_df = data.iloc[tr_idx].copy()
                val_df = data.iloc[va_idx].copy()
                for model_name in seq_models:
                    auroc, auprc = train_seq_model(model_name, train_df, val_df, store, args)
                    rows.append(
                        {
                            "task_id": task_id,
                            "condition": condition,
                            "branch": "B1",
                            "model": model_name,
                            "fold": fold,
                            "train_rows": int(len(train_df)),
                            "val_rows": int(len(val_df)),
                            "val_positive_rate": float(val_df["primary_label_binary"].mean()),
                            "auroc": auroc,
                            "auprc": auprc,
                            "n_features": len(store.feature_cols),
                        }
                    )
    out_path = output_dir / "seq_fold_metrics.csv"
    new_df = pd.DataFrame(rows)
    if out_path.exists() and args.seq_model != "all":
        existing = pd.read_csv(out_path)
        existing = existing[~existing["model"].isin(seq_models)]
        new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df = new_df.sort_values(["condition", "task_id", "model", "fold"]).reset_index(drop=True)
    new_df.to_csv(out_path, index=False)


def merge_outputs(output_dir: Path) -> None:
    dfs = []
    for name in ["xgb_fold_metrics.csv", "seq_fold_metrics.csv"]:
        p = output_dir / name
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError("No baseline metric files found to merge.")
    fold_df = pd.concat(dfs, ignore_index=True)
    fold_df.to_csv(output_dir / "phase65a_fold_metrics.csv", index=False)
    task_df = (
        fold_df.groupby(["task_id", "condition", "branch", "model"], as_index=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            folds=("fold", "nunique"),
            rows_mean=("val_rows", "mean"),
        )
        .sort_values(["condition", "task_id", "branch", "model"])
    )
    task_df.to_csv(output_dir / "phase65a_per_task_metrics.csv", index=False)
    summary = {
        "tasks": sorted(fold_df["task_id"].unique().tolist()),
        "models": sorted(fold_df["model"].unique().tolist()),
        "branches": sorted(fold_df["branch"].unique().tolist()),
        "n_fold_rows": int(len(fold_df)),
        "n_task_rows": int(len(task_df)),
        "outputs": {
            "fold_metrics": str((output_dir / "phase65a_fold_metrics.csv").as_posix()),
            "per_task_metrics": str((output_dir / "phase65a_per_task_metrics.csv").as_posix()),
        },
    }
    (output_dir / "phase65a_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    ensure_dir(output_dir)
    if args.mode == "xgb":
        run_xgb(root, args, output_dir)
    elif args.mode == "seq":
        run_seq(root, args, output_dir)
    else:
        merge_outputs(output_dir)


if __name__ == "__main__":
    main()

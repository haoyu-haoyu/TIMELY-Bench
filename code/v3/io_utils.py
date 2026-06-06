from __future__ import annotations

from pathlib import Path

import pandas as pd


def fallback_csv_path(path: Path) -> Path:
    return path.with_suffix(".csv")


def chunk_dir_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve(strict=False)
    return path.with_name(f"{path.name}.parts")


def portable_path(path: str | Path, root: str | Path | None = None) -> str:
    path = Path(path).expanduser().resolve(strict=False)
    if root is None:
        return path.as_posix()
    root_path = Path(root).expanduser().resolve(strict=False)
    try:
        return path.relative_to(root_path).as_posix()
    except Exception:
        root_name = root_path.name
        if root_name and root_name in path.parts:
            idx = path.parts.index(root_name)
            rel_parts = path.parts[idx + 1 :]
            if rel_parts:
                return Path(*rel_parts).as_posix()
            return "."
        return path.as_posix()


def relativize_value(value, root: str | Path | None = None):
    if isinstance(value, dict):
        return {k: relativize_value(v, root=root) for k, v in value.items()}
    if isinstance(value, list):
        return [relativize_value(v, root=root) for v in value]
    if isinstance(value, (str, Path)):
        text = str(value)
        if "/" in text or text.startswith("."):
            return portable_path(text, root=root)
    return value


def table_exists(path: str | Path) -> bool:
    path = Path(path).expanduser().resolve(strict=False)
    return path.exists() or fallback_csv_path(path).exists() or chunk_dir_path(path).exists()


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path).expanduser().resolve(strict=False)
    if path.exists():
        if path.suffix == ".parquet":
            try:
                return pd.read_parquet(path, **kwargs)
            except Exception:
                csv_path = fallback_csv_path(path)
                if csv_path.exists():
                    csv_kwargs = {"low_memory": False}
                    csv_kwargs.update(kwargs)
                    return pd.read_csv(csv_path, **csv_kwargs)
                raise
        csv_kwargs = {"low_memory": False}
        csv_kwargs.update(kwargs)
        return pd.read_csv(path, **csv_kwargs)
    if path.suffix == ".parquet":
        csv_path = fallback_csv_path(path)
        if csv_path.exists():
            csv_kwargs = {"low_memory": False}
            csv_kwargs.update(kwargs)
            return pd.read_csv(csv_path, **csv_kwargs)
    parts_dir = chunk_dir_path(path)
    if parts_dir.exists():
        frames = list(iter_table_chunks(path, **kwargs))
        if frames:
            return pd.concat(frames, ignore_index=True)
    raise FileNotFoundError(path)


def iter_table_chunks(path: str | Path, **kwargs):
    path = Path(path).expanduser().resolve(strict=False)
    if path.exists() or fallback_csv_path(path).exists():
        yield read_table(path, **kwargs)
        return
    parts_dir = chunk_dir_path(path)
    if not parts_dir.exists():
        raise FileNotFoundError(path)
    part_files = sorted(
        [
            p
            for p in parts_dir.iterdir()
            if p.is_file() and p.suffix in {".csv", ".parquet"}
        ]
    )
    if not part_files:
        raise FileNotFoundError(parts_dir)
    for part in part_files:
        yield read_table(part, **kwargs)


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=index)
            return path
        except Exception:
            csv_path = fallback_csv_path(path)
            df.to_csv(csv_path, index=index)
            return csv_path
    df.to_csv(path, index=index)
    return path

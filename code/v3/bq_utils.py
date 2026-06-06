from __future__ import annotations

from typing import Any

try:
    import google.auth
except Exception:  # pragma: no cover
    google = None  # type: ignore
else:  # pragma: no cover
    google = google.auth

try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover
    bigquery = None


def make_bq_client(project: str):
    """Create a BigQuery client with an explicit quota project when possible."""

    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery is not installed in this environment.")

    if google is not None:
        try:
            credentials, detected_project = google.default(quota_project_id=project)
            return bigquery.Client(project=project or detected_project, credentials=credentials)
        except Exception:
            pass

    return bigquery.Client(project=project)


def quota_project_of(client: Any) -> str | None:
    creds = getattr(client, "_credentials", None)
    return getattr(creds, "quota_project_id", None)

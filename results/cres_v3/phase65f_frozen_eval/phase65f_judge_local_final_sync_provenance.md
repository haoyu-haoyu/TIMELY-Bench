# Phase 6.5F Judge Local Final Sync Provenance

Sync date: 2026-05-12

This directory contains the finalized Phase 6.5F LLM-as-Judge artifacts copied
from the local synchronized analysis workspace back to CREATE for archival
clarity.

Important provenance:

- CREATE constructed the frozen scoring artifacts and the judge packet.
- The original CREATE-side Claude judge attempt stopped early because the API
  provider returned Cloudflare 403 / Error 1010 browser_signature_banned.
- The original CREATE-side Claude output had 225 http_error rows and 0 ok rows.
- GPT-5.4 and Gemini 3.1 Pro cross-check judge outputs were not produced in the
  original CREATE-side judge run.
- The finalized judge execution, repair, merge, and aggregation were completed
  in the local synchronized analysis workspace.
- Finalized judge coverage is 2000/2000 ok rows for each judge:
  Claude Opus 4.6, GPT-5.4, and Gemini 3.1 Pro.

Recommended paper wording:

The judge packet was constructed from frozen CREATE artifacts; external judge
execution and final judge aggregation were completed in the synchronized
analysis workspace.

Do not treat this sync directory as evidence that all judge API calls succeeded
inside CREATE Slurm. It is an archival copy of the local finalized judge
artifacts placed on CREATE to avoid future path/provenance ambiguity.

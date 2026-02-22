#!/usr/bin/env python3
"""
Deprecated legacy audit script.

This script previously used an older episode schema assumption
(top-level subject_id / clinical_notes) and can produce misleading outputs
for the current canonical release.

Use the canonical release QA flow instead:
1) python3 code/data_processing/run_final_qa.py --no-skip
2) python3 code/data_processing/build_final_release_bundle.py
"""

import sys


def main() -> int:
    msg = (
        "[DEPRECATED] scripts/final_delivery_audit.py is disabled to avoid stale audit outputs.\n"
        "Use canonical commands:\n"
        "  python3 code/data_processing/run_final_qa.py --no-skip\n"
        "  python3 code/data_processing/build_final_release_bundle.py\n"
    )
    sys.stderr.write(msg)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

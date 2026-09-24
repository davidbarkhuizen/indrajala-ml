"""
Capture the benchmark machine's profile as JSON, or compare a profile against a reference (see
indrajala_ml/machine_profile.py and docs/optimizations/method.md, "This machine").

    python scripts/machine_profile.py profile [--out FILE]
    python scripts/machine_profile.py compare docs/machine_profiles/ryzen7-3700u.json [CURRENT]

compare exits 0 if the identities match and 1 if they don't, printing each difference as
`path: reference -> current`.
"""

import sys

from indrajala_ml.machine_profile import main

if __name__ == "__main__":
    sys.exit(main())

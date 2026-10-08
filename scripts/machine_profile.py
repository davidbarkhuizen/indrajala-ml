"""
Capture the benchmark machine's profile as JSON, or compare a profile against a reference (see
indrajala_ml/measurement/machine_profile.py and docs/measurement.md, "Preparing the machine").

    python scripts/machine_profile.py profile [--out FILE]
    python scripts/machine_profile.py compare docs/machine_profiles/i7-9700k.json [CURRENT]

compare exits 0 if the identities match and 1 if they don't, printing each difference as
`path: reference -> current`.
"""

import sys

from indrajala_ml.measurement.machine_profile import main

if __name__ == "__main__":
    sys.exit(main())

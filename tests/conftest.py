import os
import tempfile

# Isolate test runs from the real data/ dir; must be set before backend.config is imported.
os.environ.setdefault("FORENSICS_DATA_DIR", tempfile.mkdtemp(prefix="forensics_test_"))

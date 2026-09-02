"""Shared test environment setup."""

import os


os.environ.setdefault("DEBUG", "True")
os.environ.setdefault("INTERNAL_API_KEY", "ci-test-internal-key")

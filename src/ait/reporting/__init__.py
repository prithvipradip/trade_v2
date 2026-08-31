"""Reporting authorities shared by every operator-facing surface.

Deliberately import-light: `status.py` (a root script run by hand and by the
web dashboard) imports from here, so this package must never pull the broker,
the scheduler, or the ML stack in at import time.
"""

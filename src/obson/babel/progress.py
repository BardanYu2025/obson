"""Flush progress to stderr so CLI JSON output remains machine-readable."""

import os
import sys
from datetime import datetime


def progress(message):
    print(f"[{datetime.now():%H:%M:%S}] pid={os.getpid()} {message}", file=sys.stderr, flush=True)

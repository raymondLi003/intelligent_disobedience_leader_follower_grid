"""One-shot LLM-proxy quota check (exit 0 = quota available, exit 1 = still exhausted).

Usage:
    python check_quota.py                 # single probe
    while ! python check_quota.py; do sleep 1800; done   # poll until it refreshes
"""
import datetime
import sys

from llmproxy import LLMProxy

r = LLMProxy().model_info()
now = datetime.datetime.now().astimezone()
utc = datetime.datetime.now(datetime.timezone.utc)
stamp = f"[{now:%H:%M %Z} / {utc:%H:%M} UTC]"
if isinstance(r, dict) and "error" not in r:
    print(f"{stamp} QUOTA OK — proxy is answering again")
    sys.exit(0)
print(f"{stamp} still exhausted: {r.get('error', r) if isinstance(r, dict) else r}")
sys.exit(1)

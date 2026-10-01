# shared-logs

Each trading box's own bot events, one gzip JSONL per ET trading day: `<owner>_logs/YYYY-MM-DD.jsonl.gz`.
Written by `src/ait/orchestration/shared_logs.py`; read with `scripts/read_shared_logs.py`. Orphan branch — never merge into code.

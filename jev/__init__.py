"""Frozen decision head for a frozen backbone.

Entry points:
  python -m jev.identity    --out identity/runtime-identity.json   (on the serving host)
  python -m jev.predict     --identity ... --state ... --question ... --options ...
  python -m jev.cascade     --identity ... --items items.jsonl
  python -m jev.eval_benchmark --mode criterion_as_option --out ...
  python -m jev.selfcheck                                            (offline, no server)
"""
VERSION = '1.0.0'

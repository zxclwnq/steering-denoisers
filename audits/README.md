# Audits

One-shot analysis scripts kept **verbatim** as the executable record behind an
audit note in `docs/`. They are archived evidence, not maintained source.

* They live outside `src/`, `scripts/`, and `configs/` so that running or adding
  one cannot change any release's `source_revision` identity.
* They are excluded from lint for the same reason: reformatting them would break
  the correspondence with the numbers already published in the audit note.

| script | note |
|---|---|
| `degeneracy_rule_recovery_2026_08_14.py` | `docs/DEGENERACY_RULE_AUDIT_2026-08-14.md` §2 |
| `degeneracy_rule_materiality_2026_08_14.py` | `docs/DEGENERACY_RULE_AUDIT_2026-08-14.md` §5 |

---
applyTo: "**"
---

## PR Review Focus for MLOps Project

You are reviewing code for an MLOps data pipeline (raw → bronze → silver → gold → train). Focus on finding bugs, logic errors, and dead code. Keep reviews simple and actionable.

### 1. Data Pipeline Bugs
- Schema drift: field order must match `schemas.BRONZE_SCHEMA` / `SILVER_SCHEMA` / `GOLD_SCHEMA`. Never build column order from sets or dict keys.
- DType issues: IDs and categorical fields must stay string-like before `pa.Table.from_pandas()` or `pa.Table.from_pydict()`.
- Timestamp contracts: raw layer uses `event_time`; all downstream layers must use `source_event_time`.
- Null handling: check that null prices are allowed but `price <= 0` is rejected.

### 2. Memory & Performance
- **NEVER** load full datasets into RAM. Always use chunked processing (`ParquetFile.iter_batches`, `pl.scan_csv`, `collect_batches`).
- Bronze writes must use `ParquetWriter` append, not single-shot materialization.
- Silver dedup/sort must use external sort + k-way merge, not `df.unique()` on full data.
- Gold snapshot generation must stream sessions, not load entire silver into memory.

### 3. Logic Errors to Catch
- Temporal bugs: train/val/test splits must be session-based (not row-based) to prevent data leakage.
- Feature computation: snapshot features must reflect state BEFORE the current event, not after.
- Label generation: `label` must predict purchase within `LABEL_HORIZON` (10 min) from current event time.
- Window selection: bronze must respect `--window-profile` and only read matching raw files.

### 4. Dead Code & Redundancy
- Remove unused imports, wrapper functions that just call another function, and no-op calls.
- If a function parameter is never used, question why it exists.
- If `normalize_category_code(df, policy="keep")` is called, it's a no-op — remove it.

### 5. ML-Specific Checks
- Data leakage: features computed from future events relative to prediction point.
- Deterministic ordering: sort keys must be stable and reproducible across runs.
- DVC sync: if code changes, check `dvc.yaml` deps/outs are updated too.

### 6. Testing Requirements
- Bug fixes must include regression tests.
- New logic must have unit tests, especially edge cases (empty input, single row, all nulls).
- Do not approve PRs that break existing tests.

### 7. Quick Commands to Verify
Ask the author if they ran:
```bash
ruff check .
pytest training/tests -q
dvc dag
```

Keep reviews focused. If you find a bug, explain the bug and suggest a fix. If you find dead code, say "Remove — unused." Be direct.

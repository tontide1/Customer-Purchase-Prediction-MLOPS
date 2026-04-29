# Skill Optimization Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this spec task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep a minimal, MLOps-focused skills set for Python/FastAPI/ML/Ops, removing non-essential project skills.

**Architecture:** Compare project skills against Superpowers defaults to detect duplicates, keep the four selected MLOps skills (preserving any customizations), and delete three non-MLOps skills. Ensure Superpowers continues to work after pruning.

**Tech Stack:** OpenCode skills, filesystem, diff.

---

## Scope

- Keep a minimal, MLOps-focused skill set aligned with the current repo (Week 1 data foundation) and roadmap (serving, monitoring, ops).
- Preserve local customizations in retained skills.
- Remove non-essential skills for MLOps.

## Retained Skills (4)

- `senior-ml-engineer` — core MLOps, serving, monitoring, drift, inference ops.
- `senior-data-engineer` — data pipeline and ETL/ELT practices for bronze/silver layers.
- `docker-development` — containerization, compose orchestration, and infra hygiene.
- `data-quality-auditor` — data validation and quality checks.

## Removed Skills (3)

- `senior-frontend`
- `senior-fullstack`
- `api-test-suite-builder`

## Implementation Notes

- If a retained skill is customized locally, keep it as-is.
- If a removed skill has customizations, confirm before deletion.
- Use `diff -qr` against the Superpowers cache to detect identical skills.

## Success Criteria

- `/.opencode/skills/` contains only the 4 retained skills.
- No customizations are lost in retained skills.
- Superpowers skills remain discoverable and functional.

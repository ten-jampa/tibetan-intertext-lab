# Domain Docs

How engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- Root `CONTEXT.md`
- Root `docs/adr/` for architectural decisions relevant to the area being changed

If those files do not exist for a future area, proceed silently. `CONTEXT.md` and ADRs grow only when terms or decisions become worth recording.

## File structure

This repo is single-context:

```text
/
├── CONTEXT.md
├── docs/adr/
└── tibetan_pipeline/
```

## Use glossary vocabulary

When naming domain concepts in plans, issues, tests, or refactor proposals, use the terms defined in `CONTEXT.md`. Do not drift to synonyms the glossary marks as avoided.

## Flag ADR conflicts

If a proposal contradicts an ADR, surface that explicitly instead of silently overriding it.

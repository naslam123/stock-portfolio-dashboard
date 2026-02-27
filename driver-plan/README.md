# DRIVER Project

This project follows the DRIVER methodology for finance/quant tool development.

## Project Structure

```
driver-plan/
├── README.md                 ← You are here
├── research.md               ← Created by /research or /define
├── product-overview.md       ← Created by /define (your PRD)
├── roadmap.md                ← Created by /represent-roadmap
├── spec-[section].md         ← Created by /represent-section
├── data-model.md             ← Created by /represent-datamodel
├── validation.md             ← Created by /validate
├── reflect.md                ← Created by /reflect
├── design/                   ← Web apps only
│   ├── tokens.json
│   └── shell.md
└── build/                    ← Implementation artifacts
    └── [section-id]/
        ├── data.json
        └── types.ts
```

## Workflow

1. `/finance-driver:define` — Establish vision, research what exists (开题调研)
2. `/finance-driver:represent-roadmap` — Break into 3-5 buildable sections
3. `/finance-driver:implement-screen` — Build and run, iterate on feedback
4. `/finance-driver:validate` — Cross-check: known answers, reasonableness, edges, AI risks
5. `/finance-driver:evolve` — Generate final export package
6. `/finance-driver:reflect` — Capture lessons learned

## Philosophy

**Cognition Mate (认知伙伴):** 互帮互助，因缘合和，互相成就

- AI brings: patterns, research, heavy lifting on code
- You bring: vision, domain expertise, judgment
- Neither creates alone. Meaning emerges from interaction.

## Next Step

Run `/finance-driver:define` to begin.

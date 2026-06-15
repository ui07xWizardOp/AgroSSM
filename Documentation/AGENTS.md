# AGENTS.md — Documentation Subdirectory

## Purpose

This directory contains all research specifications, architecture documents, validation reports, and planning documents for the AgroEarthFM / AgroSSM project. **No source code lives here.**

## Document Authority Chain (Read in This Order)

An agent working on this project MUST read documents in this priority order:

1. **`AgroSSM_Agent_Handoff.md`** — The step-by-step implementation instructions. Start here when writing code.
2. **`AgroSSM_Implementation_Plan.md`** — The layman-friendly overview of the entire project. Read to understand the "why".
3. **`AgroEarthFM_Architecture_v3_AgroSSM.md`** — The active architecture specification. This defines WHAT to build.
4. **`AgroEarthFM_MVP_Spec_v2.3.md`** — The frozen master specification. This defines the scientific claims, evaluation protocol, and boundaries. DO NOT MODIFY.

## Document Status

| File | Status | Purpose |
|------|--------|---------|
| `AgroEarthFM_MVP_Spec_v2.3.md` | **FROZEN** | Master spec: claims, metrics, splits, baselines, terminology |
| `AgroEarthFM_Architecture_v3_AgroSSM.md` | **ACTIVE** | v3.0 architecture (supersedes v2.3 architecture only) |
| `AgroSSM_Implementation_Plan.md` | **ACTIVE** | Detailed layman-friendly implementation plan |
| `AgroSSM_Agent_Handoff.md` | **ACTIVE** | Agent-executable implementation instructions |
| `AgroEarthFM_Complete_Finalized_Specification.md` | Superseded by v2.3 | Historical reference only |
| `AgroEarthFM_Final_Architecture_Spec.md` | Superseded by v3.0 | Historical reference only |
| `AgroEarthFM_MVP_Spec_v2.2.md` | Superseded by v2.3 | Historical reference only |
| `AgroEarthFM_Final_Spec_v2.1.md` | Superseded by v2.2 | Historical reference only |
| `AgroEarthFM_Validation_Report.md` | Historical | Original IV&V report |
| `AgroEarthFM_Full_Context_File_Coherence_Validation.md` | Historical | Coherence audit (gaps resolved in v2.3) |
| `AgroEarthFM_Issue_Resolution_and_VV_Draft.md` | Historical | Issue tracking document |
| `AgroEarthFM_External_Feedback_Review.md` | Historical | External feedback integration |
| `AgroEarthFM_File_Review_Validation_Response.md` | Historical | File review reconciliation |

## Rules

- **Never modify FROZEN documents.** If a frozen document needs updating, create a new version.
- **Always update ACTIVE documents** when implementation decisions change.
- `.docx`, `.pdf`, and `_text.txt` files are rendered/extracted copies of their `.md` counterparts. Do not edit them directly.

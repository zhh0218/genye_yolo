# DepthLightFPN Draw.io Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native, fully editable diagrams.net file that reproduces the approved DepthLightFPN structure and tensor dimensions.

**Architecture:** Store the diagram as an uncompressed `mxGraphModel` inside an `mxfile`. Use native vertices for groups, tensors, operators, concatenation diamonds, and annotations; use native orthogonal edges for computation and identity paths.

**Tech Stack:** diagrams.net XML (`mxGraphModel`), PowerShell XML parser for validation.

## Global Constraints

- Create `DepthLightFPN.drawio` in the project root.
- Use a left-to-right landscape layout.
- Keep every shape, label, and connector natively editable.
- Represent feature tensors in blue, operators in gray, concatenations as diamonds, and identity as a dashed connector.
- Preserve every tensor shape and channel transformation specified in the approved design.

---

### Task 1: Create and validate the editable DepthLightFPN diagram

**Files:**

- Create: `DepthLightFPN.drawio`
- Reference: `docs/superpowers/specs/2026-08-13-depthlightfpn-drawio-design.md`

**Interfaces:**

- Consumes: the approved node labels, dimensions, operator sequence, and visual language from the design specification.
- Produces: one diagrams.net document whose diagram contains native `mxCell` vertices and edges.

- [ ] **Step 1: Define the expected validation assertions**

Validate that XML parsing succeeds, the document contains one `mxGraphModel`, all required labels are present (`D3`, `D4`, `D5`, `D3F`, `D4F`, `D5F`, two `Nearest Upsample`, two `Channel Concat`, and two `CBS 1x1`), and the graph contains exactly eleven directed connectors.

- [ ] **Step 2: Run validation before creation**

Run: `Test-Path .\DepthLightFPN.drawio`

Expected: `False` before the file is created.

- [ ] **Step 3: Create the native draw.io XML**

Create an `mxfile` with a single `diagram` and uncompressed `mxGraphModel`. Add two grouped regions, twelve native content nodes, one explanatory annotation, and eleven orthogonal connectors. Route the `D5 -> D5F` connector as dashed and label it `Identity`.

- [ ] **Step 4: Parse and verify the completed document**

Run a PowerShell XML validation that loads `DepthLightFPN.drawio`, asserts the required labels, counts eleven edge cells, confirms the identity edge contains `dashed=1`, and confirms every edge has both `source` and `target` attributes.

Expected: all assertions pass and the command prints the vertex count, edge count, and output path.

- [ ] **Step 5: Inspect repository status**

Run: `git status --short -- DepthLightFPN.drawio docs/superpowers/plans/2026-08-13-depthlightfpn-drawio.md`

Expected: only the new diagram and implementation-plan file appear for this task.

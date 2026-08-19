# DepthLightFPN Draw.io Diagram Design

## Deliverable

Create `DepthLightFPN.drawio` in the project root as a native, fully editable diagrams.net document.

## Layout

- Use a left-to-right landscape layout suitable for a paper figure.
- Place the three Depth Backbone outputs (`D3`, `D4`, `D5`) in a grouped area on the left.
- Place the internal `DepthLightFPN` operations in a grouped area on the right.
- Show the `D5 -> Upsample -> Concat(D4) -> CBS -> D4F -> Upsample -> Concat(D3) -> CBS -> D3F` top-down path.
- Show `D5 -> D5F` as a dashed identity path.

## Labels and dimensions

- `D3`: `256 x 80 x 80`
- `D4`: `256 x 40 x 40`
- `D5`: `512 x 20 x 20`
- First upsample output: `512 x 40 x 40`
- First concat output: `768 x 40 x 40`
- First CBS: `1x1, 768 -> 256`
- `D4F`: `256 x 40 x 40`
- Second upsample output: `256 x 80 x 80`
- Second concat output: `512 x 80 x 80`
- Second CBS: `1x1, 512 -> 256`
- `D3F`: `256 x 80 x 80`
- `D5F = D5`: `512 x 20 x 20`

`CBS 1x1` means `Conv 1x1 + BatchNorm + SiLU`.

## Visual language

- Blue rounded rectangles: Depth feature tensors.
- Neutral gray rounded rectangles: Upsample and CBS operators.
- Diamonds: channel concatenation.
- Solid orthogonal arrows: computation paths.
- Dashed orthogonal arrow: identity mapping from `D5` to `D5F`.
- Keep all elements as native draw.io nodes and connectors so labels, colors, positions, and routes remain editable.

## Validation

- Parse the output as XML.
- Confirm all required nodes and connectors exist.
- Confirm the file opens as an uncompressed `mxGraphModel` diagrams.net document.

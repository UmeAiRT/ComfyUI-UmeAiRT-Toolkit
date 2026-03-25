# ⬡ Label

> Visual annotation node for organizing and documenting workflows.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `title` | `STRING` | ✅ | Label Title | Header text |
| `text` | `STRING` | ✅ | Description or Notes | Body content (multiline) |
| `color` | `COMBO` | ✅ | white | Text color (red, green, blue, yellow, cyan, magenta, white, black) |
| `font_size` | `INT` | ✅ | 20 | Text size in pixels (10–100) |

## Outputs

None — this is a purely visual node with no functional outputs.

!!! tip "Use cases"
    - Mark workflow sections (e.g. "LOADING", "POST-PROCESS")
    - Document settings choices for future reference
    - Leave notes for collaborators

<!-- TODO: Screenshot — Multiple Label nodes used to annotate a workflow -->
<!-- PLACEHOLDER: Show 2-3 Label nodes with different colors placed around a workflow as section headers -->

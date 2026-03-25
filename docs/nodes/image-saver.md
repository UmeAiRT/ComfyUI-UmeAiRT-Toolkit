# ⬡ Image Saver

> Save generated images with automatic metadata embedding, configurable naming, and folder organization.

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `gen_pipe` | `UME_PIPELINE` | ✅ | Pipeline containing the generated image to save |

## Outputs

This is an **output node** — it saves files to disk but produces no outputs.

## Features

- **Automatic metadata** — embeds generation parameters (model, seed, steps, prompts) in PNG metadata
- **Configurable naming** — supports templates with date, model name, seed, counter
- **Folder organization** — saves to ComfyUI's `output/` directory

<!-- TODO: Screenshot — Image Saver node with a generated image preview -->
<!-- PLACEHOLDER: Show Image Saver at the end of a workflow, with the preview showing a saved image and the filename template visible -->

import { app } from "../../scripts/app.js";

// UmeAiRT Node Colors
// Dark background colors for nodes, lighter for connections
const UME_NODE_COLORS = {
    // === BLOCK NODES ===

    // Settings Block - Amber/Bronze
    "UmeAiRT_GenerationSettings": {
        color: "#935116",
        bgcolor: "#4A290B"
    },

    // Model Loader Blocks - Blue
    "UmeAiRT_FilesSettings_Checkpoint": {
        color: "#154360",
        bgcolor: "#0A2130"
    },
    "UmeAiRT_FilesSettings_FLUX": {
        color: "#154360",
        bgcolor: "#0A2130"
    },
    "UmeAiRT_FilesSettings_Fragmented": {
        color: "#154360",
        bgcolor: "#0A2130"
    },
    "UmeAiRT_FilesSettings_ZIMG": {
        color: "#154360",
        bgcolor: "#0A2130"
    },

    // LoRA Blocks - Violet
    "UmeAiRT_LoraBlock_1": {
        color: "#4A235A",
        bgcolor: "#25122D"
    },
    "UmeAiRT_LoraBlock_3": {
        color: "#4A235A",
        bgcolor: "#25122D"
    },
    "UmeAiRT_LoraBlock_5": {
        color: "#4A235A",
        bgcolor: "#25122D"
    },
    "UmeAiRT_LoraBlock_10": {
        color: "#4A235A",
        bgcolor: "#25122D"
    },

    // Block Sampler - Slate Gray (neutral main processor)
    "UmeAiRT_BlockSampler": {
        color: "#2C3E50",
        bgcolor: "#1A252F"
    },



    // === PIPELINE NODES ===

    // Pipeline Upscale - Pale Blue
    "UmeAiRT_PipelineUltimateUpscale": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    "UmeAiRT_PipelineUltimateUpscale_Advanced": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    "UmeAiRT_PipelineSeedVR2Upscale": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    "UmeAiRT_PipelineSeedVR2Upscale_Advanced": {
        color: "#2471A3",
        bgcolor: "#123851"
    },

    // Detailer Daemon - Pale Blue
    "UmeAiRT_Detailer_Daemon_Simple": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    "UmeAiRT_Detailer_Daemon_Advanced": {
        color: "#2471A3",
        bgcolor: "#123851"
    },

    // Pipeline SubjectDetailer - Pale Blue
    "UmeAiRT_PipelineSubjectDetailer": {
        color: "#2471A3",
        bgcolor: "#123851"
    },

    // Pipeline Inpaint Composite - Pale Blue
    "UmeAiRT_PipelineInpaintComposite": {
        color: "#2471A3",
        bgcolor: "#123851"
    },

    // Pipeline Image Saver - Blue-Teal
    "UmeAiRT_PipelineImageSaver": {
        color: "#1A5653",
        bgcolor: "#0D2B29"
    },

    // Pipeline Image Loader - Rust Red
    "UmeAiRT_PipelineImageLoader": {
        color: "#6B2D1A",
        bgcolor: "#35160D"
    },

    // Pipeline Image Process - Pale Blue
    "UmeAiRT_PipelineImageProcess": {
        color: "#2471A3",
        bgcolor: "#123851"
    },

    // Multi-LoRA Loader - Violet
    "UmeAiRT_MultiLoraLoader": {
        color: "#4A235A",
        bgcolor: "#25122D"
    },

    // === UTILITY NODES ===

    // Bbox Detector Loader - Pale Blue
    "UmeAiRT_BboxDetectorLoader": {
        color: "#2471A3",
        bgcolor: "#123851"
    },

    // Source Image Output - Rust Red
    "UmeAiRT_SourceImage_Output": {
        color: "#6B2D1A",
        bgcolor: "#35160D"
    },


    // Block Image Loaders - Rust Red
    "UmeAiRT_BlockImageLoader": {
        color: "#6B2D1A",
        bgcolor: "#35160D"
    },
    "UmeAiRT_BlockImageLoader_Advanced": {
        color: "#6B2D1A",
        bgcolor: "#35160D"
    },

    // Block Image Process - Amber/Bronze (Settings family)
    "UmeAiRT_BlockImageProcess": {
        color: "#935116",
        bgcolor: "#4A290B"
    },

    // ControlNet - Amber/Bronze
    "UmeAiRT_ControlNetImageApply_Simple": {
        color: "#935116",
        bgcolor: "#4A290B"
    },
    "UmeAiRT_ControlNetImageApply_Advanced": {
        color: "#935116",
        bgcolor: "#4A290B"
    },
    "UmeAiRT_ControlNetImageProcess": {
        color: "#935116",
        bgcolor: "#4A290B"
    },

    // Bundle Downloader - Light Grey/Dark Text
    "UmeAiRT_Bundle_Downloader": {
        color: "#333333",
        bgcolor: "#D5D8DC"
    },

    // Bundle Auto-Loader - Blue
    "UmeAiRT_BundleLoader": {
        color: "#154360",
        bgcolor: "#0A2130"
    },

    // Label - Dark Gray
    "UmeAiRT_Label": {
        color: "#34495E",
        bgcolor: "#1A252F"
    },

    // Prompt Inputs - Green
    "UmeAiRT_Positive_Input": {
        color: "#145A32",
        bgcolor: "#0A2D19"
    },
    "UmeAiRT_Negative_Input": {
        color: "#641E16",
        bgcolor: "#3B100C"
    },

    // Unpack Nodes - Amber
    "UmeAiRT_Faces_Unpack_Node": {
        color: "#935116",
        bgcolor: "#4A290B"
    },
    "UmeAiRT_Tags_Unpack_Node": {
        color: "#935116",
        bgcolor: "#4A290B"
    },
    "UmeAiRT_Pipe_Unpack_Node": {
        color: "#935116",
        bgcolor: "#4A290B"
    },
    "UmeAiRT_Unpack_SettingsBundle": {
        color: "#935116",
        bgcolor: "#4A290B"
    },
    "UmeAiRT_Unpack_PromptsBundle": {
        color: "#935116",
        bgcolor: "#4A290B"
    },
    "UmeAiRT_Unpack_FilesBundle": {
        color: "#935116",
        bgcolor: "#4A290B"
    },
    "UmeAiRT_Unpack_ImageBundle": {
        color: "#935116",
        bgcolor: "#4A290B"
    },

    // Pack/Unpack Pipeline - Teal
    "UmeAiRT_Unpack_Pipeline": {
        color: "#17A589",
        bgcolor: "#0B5345"
    },
    "UmeAiRT_Pack_Bundle": {
        color: "#154360",
        bgcolor: "#0A2130"
    },

    // Log Viewer - Dark Grey
    "UmeAiRT_Log_Viewer": {
        color: "#34495E",
        bgcolor: "#1A252F"
    }
};

// Connection slot colors - Softer, harmonious palette
const UME_SLOT_COLORS = {
    "UME_BUNDLE": "#3498DB",     // Bright Blue (model bundle)
    "UME_SETTINGS": "#CD8B62",   // Amber/Copper (matches node)
    "UME_PROMPTS": "#52BE80",    // Soft Green  
    "POSITIVE": "#52BE80",       // Soft Green for Positive
    "NEGATIVE": "#E74C3C",       // Vibrant Red for Negative
    "UME_LORA_STACK": "#9B59B6", // Purple
    "UME_IMAGE": "#DC7633",      // Orange/Brown
    "UME_PIPELINE": "#1ABC9C"    // Teal (generation context)
};

// Enforce minimum sizes was removed as Nodes 2.0 fixed the shrinking native bug.

app.registerExtension({
    name: "UmeAiRT.NodeColors",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const colors = UME_NODE_COLORS[nodeData.name];

        if (colors) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                this.color = colors.color;
                this.bgcolor = colors.bgcolor;
            };
        }

        // --- VUE 2.0 BUG FIX: Ghost Spacing for Advanced Inputs ---
        // ComfyUI Vue reads computeSize to set min-height. By default, Litegraph adds 
        // the height of ALL widgets (even visually hidden "advanced" ones).
        // This causes a massive empty space at the bottom of nodes.
        // We strip advanced widgets from the LiteGraph math so Vue can use DOM flex dynamically.
        if (nodeData.name.startsWith("UmeAiRT_")) {
            const defaultComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function (out) {
                let size = [0, 0];
                if (defaultComputeSize) {
                    size = defaultComputeSize.apply(this, arguments);
                }
                
                if (this.widgets && this.widgets.length > 0) {
                    let advancedCount = 0;
                    for (let w of this.widgets) {
                        // Mark advanced widgets to be omitted from absolute height calculation
                        if (w.advanced || (w.options && w.options.advanced)) {
                            advancedCount++;
                        }
                    }
                    if (advancedCount > 0) {
                        // Standard ComfyUI widget height calculation is approx 24px per line + 4px padding
                        size[1] -= (advancedCount * 28);
                    }
                }
                return size;
            };
        }
    },

    async setup() {
        // === Modern Vue ComfyUI: inject CSS custom properties ===
        const cssRules = Object.entries(UME_SLOT_COLORS)
            .map(([type, color]) => `--color-datatype-${type}: ${color};`)
            .join('\n            ');
        const style = document.createElement('style');
        style.id = 'umeairt-slot-colors';
        style.textContent = `:root {\n            ${cssRules}\n        }`;
        document.head.appendChild(style);

        // === Legacy LiteGraph fallback (older ComfyUI versions) ===
        if (app.canvas && app.canvas.default_connection_color_byType) {
            Object.assign(app.canvas.default_connection_color_byType, UME_SLOT_COLORS);
        }
    }
});

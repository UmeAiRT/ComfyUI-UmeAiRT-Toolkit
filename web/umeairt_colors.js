import { app } from "../../scripts/app.js";

// UmeAiRT Block Node Colors
// Dark background colors for nodes, lighter for connections
const UME_NODE_COLORS = {
    // === BLOCK NODES ===

    // Settings Block - Amber/Bronze (more muted than bright yellow)
    "UmeAiRT_GenerationSettings": {
        color: "#935116",
        bgcolor: "#4A290B"
    },

    // Model Loader Blocks - Blue
    "UmeAiRT_FilesSettings_Checkpoint": {
        color: "#154360",
        bgcolor: "#0A2130"
    },
    "UmeAiRT_FilesSettings_Checkpoint_Advanced": {
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

    // Prompt Block - Green
    "UmeAiRT_PromptBlock": {
        color: "#145A32",
        bgcolor: "#0A2D19"
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

    // Block Upscale/Detailer - Pale Blue
    "UmeAiRT_BlockUltimateSDUpscale": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    "UmeAiRT_BlockFaceDetailer": {
        color: "#2471A3",
        bgcolor: "#123851"
    },

    // === WIRELESS NODES ===

    // Wireless KSampler - Slate Gray (same as Block Sampler)
    "UmeAiRT_WirelessKSampler": {
        color: "#2C3E50",
        bgcolor: "#1A252F"
    },

    // Wireless Upscale - Pale Blue
    "UmeAiRT_WirelessUltimateUpscale": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    "UmeAiRT_WirelessUltimateUpscale_Advanced": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    "UmeAiRT_WirelessSeedVR2Upscale": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    "UmeAiRT_WirelessSeedVR2Upscale_Advanced": {
        color: "#2471A3",
        bgcolor: "#123851"
    },

    // Detailer Daemon - Pale Blue (Same as SeedVR2)
    "UmeAiRT_Detailer_Daemon_Simple": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    "UmeAiRT_Detailer_Daemon_Advanced": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    // Wireless FaceDetailer - Pale Blue
    "UmeAiRT_WirelessFaceDetailer_Simple": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    "UmeAiRT_WirelessFaceDetailer_Advanced": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    // Wireless Inpaint Composite - Pale Blue (Post-Process family)
    "UmeAiRT_WirelessInpaintComposite": {
        color: "#2471A3",
        bgcolor: "#123851"
    },

    // Wireless Image Saver - Blue-Teal (output, distinct from green prompts)
    "UmeAiRT_WirelessImageSaver": {
        color: "#1A5653",
        bgcolor: "#0D2B29"
    },

    // Wireless Image Loader - Rust Red (input, distinct from amber settings)
    "UmeAiRT_WirelessImageLoader": {
        color: "#6B2D1A",
        bgcolor: "#35160D"
    },

    // Wireless Checkpoint Loader - Blue
    "UmeAiRT_WirelessCheckpointLoader": {
        color: "#154360",
        bgcolor: "#0A2130"
    },

    // Multi-LoRA Loader - Violet
    "UmeAiRT_MultiLoraLoader": {
        color: "#4A235A",
        bgcolor: "#25122D"
    },

    // === UTILITY NODES ===

    // Debug - Dark Gray
    "UmeAiRT_Wireless_Debug": {
        color: "#34495E",
        bgcolor: "#1A252F"
    },

    // Bbox Detector Loader - Pale Blue (same as upscale/detailer family)
    "UmeAiRT_BboxDetectorLoader": {
        color: "#2471A3",
        bgcolor: "#123851"
    },

    // Source Image Output - Rust Red (image family)
    "UmeAiRT_SourceImage_Output": {
        color: "#6B2D1A",
        bgcolor: "#35160D"
    },

    // Block Image Loader Block - Rust Red
    "UmeAiRT_BlockImageLoader": {
        color: "#6B2D1A",
        bgcolor: "#35160D"
    },
    "UmeAiRT_BlockImageLoader_Advanced": {
        color: "#6B2D1A",
        bgcolor: "#35160D"
    },

    // Wireless Image Process - Pale Blue
    "UmeAiRT_WirelessImageProcess": {
        color: "#2471A3",
        bgcolor: "#123851"
    },
    // Block Image Process - Amber/Bronze (Settings family)
    "UmeAiRT_BlockImageProcess": {
        color: "#935116",
        bgcolor: "#4A290B"
    },

    // ControlNet - Amber/Bronze (Same as Image Process)
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

    // Tools - Light Grey/Dark Text
    "UmeAiRT_Bundle_Downloader": {
        color: "#333333",
        bgcolor: "#D5D8DC"
    },

    // Bundle Auto-Loader - Blue (Loader family)
    "UmeAiRT_BundleLoader": {
        color: "#154360",
        bgcolor: "#0A2130"
    },

    // Label - Dark Gray (utility)
    "UmeAiRT_Label": {
        color: "#34495E",
        bgcolor: "#1A252F"
    },

    // === INPUT/OUTPUT NODES (Raw Wireless) - Subtle Gray ===

    // Settings-related I/O - Amber tint
    "UmeAiRT_Guidance_Input": { color: "#6B4423", bgcolor: "#35220F" },
    "UmeAiRT_Guidance_Output": { color: "#6B4423", bgcolor: "#35220F" },
    "UmeAiRT_Steps_Input": { color: "#6B4423", bgcolor: "#35220F" },
    "UmeAiRT_Steps_Output": { color: "#6B4423", bgcolor: "#35220F" },
    "UmeAiRT_Denoise_Input": { color: "#6B4423", bgcolor: "#35220F" },
    "UmeAiRT_Denoise_Output": { color: "#6B4423", bgcolor: "#35220F" },
    "UmeAiRT_Seed_Input": { color: "#6B4423", bgcolor: "#35220F" },
    "UmeAiRT_Seed_Output": { color: "#6B4423", bgcolor: "#35220F" },
    "UmeAiRT_ImageSize_Input": { color: "#6B4423", bgcolor: "#35220F" },
    "UmeAiRT_ImageSize_Output": { color: "#6B4423", bgcolor: "#35220F" },
    "UmeAiRT_FPS_Input": { color: "#6B4423", bgcolor: "#35220F" },
    "UmeAiRT_FPS_Output": { color: "#6B4423", bgcolor: "#35220F" },

    // Sampler/Scheduler I/O - Gray
    "UmeAiRT_Scheduler_Input": { color: "#2C3E50", bgcolor: "#1A252F" },
    "UmeAiRT_Scheduler_Output": { color: "#2C3E50", bgcolor: "#1A252F" },
    "UmeAiRT_Sampler_Input": { color: "#2C3E50", bgcolor: "#1A252F" },
    "UmeAiRT_Sampler_Output": { color: "#2C3E50", bgcolor: "#1A252F" },
    "UmeAiRT_SamplerScheduler_Input": { color: "#2C3E50", bgcolor: "#1A252F" },

    // Prompt I/O - Green
    "UmeAiRT_Positive_Input": { color: "#145A32", bgcolor: "#0A2D19" },
    "UmeAiRT_Positive_Output": { color: "#145A32", bgcolor: "#0A2D19" },
    "UmeAiRT_Negative_Input": { color: "#145A32", bgcolor: "#0A2D19" },
    "UmeAiRT_Negative_Output": { color: "#145A32", bgcolor: "#0A2D19" },

    // Model/VAE/CLIP I/O - Blue
    "UmeAiRT_Model_Input": { color: "#154360", bgcolor: "#0A2130" },
    "UmeAiRT_Model_Output": { color: "#154360", bgcolor: "#0A2130" },
    "UmeAiRT_VAE_Input": { color: "#154360", bgcolor: "#0A2130" },
    "UmeAiRT_VAE_Output": { color: "#154360", bgcolor: "#0A2130" },
    "UmeAiRT_CLIP_Input": { color: "#154360", bgcolor: "#0A2130" },
    "UmeAiRT_CLIP_Output": { color: "#154360", bgcolor: "#0A2130" },

    // Latent I/O - Gray (sampler family)
    "UmeAiRT_Latent_Input": { color: "#2C3E50", bgcolor: "#1A252F" },
    "UmeAiRT_Latent_Output": { color: "#2C3E50", bgcolor: "#1A252F" },

    // === RESTORED NODES ===

    // Wireless Model Loader - Blue (Model Family)
    "UmeAiRT_WirelessModelLoader": {
        color: "#154360",
        bgcolor: "#0A2130"
    },

    // Seed Nodes - Green (Prompt Family)
    "UmeAiRT_Seed_Node": {
        color: "#145A32",
        bgcolor: "#0A2D19"
    },
    "UmeAiRT_CR_Seed_Node": {
        color: "#145A32",
        bgcolor: "#0A2D19"
    },
    "UmeAiRT_GlobalSeed": {
        color: "#145A32",
        bgcolor: "#0A2D19"
    },

    // Unpack Nodes - Amber (Settings/Utility Family)
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
    "UmeAiRT_Unpack_Files": {
        color: "#935116",
        bgcolor: "#4A290B"
    },
    "UmeAiRT_Unpack_Settings": {
        color: "#935116",
        bgcolor: "#4A290B"
    },
    "UmeAiRT_Unpack_Prompt": {
        color: "#935116",
        bgcolor: "#4A290B"
    },

    // Log Viewer - Dark Grey (Utility)
    "UmeAiRT_Log_Viewer": {
        color: "#34495E",
        bgcolor: "#1A252F"
    },

    // Wireless Inputs - Amber (Settings Family)
    "UmeAiRT_Resolution": {
        color: "#935116",
        bgcolor: "#4A290B"
    },
    "UmeAiRT_Positive_Input": {
        color: "#145A32", // Green for Positive
        bgcolor: "#0A2D19"
    },
    "UmeAiRT_Negative_Input": {
        color: "#641E16", // Deep Red for Negative
        bgcolor: "#3B100C"
    },
    "UmeAiRT_SpeedMode": {
        color: "#935116",
        bgcolor: "#4A290B"
    }
};

// Connection slot colors - Softer, harmonious palette
const UME_SLOT_COLORS = {
    "UME_FILES": "#5499C7",      // Soft Blue
    "UME_SETTINGS": "#CD8B62",   // Amber/Copper (matches node)
    "UME_PROMPTS": "#52BE80",    // Soft Green  
    "POSITIVE": "#52BE80",       // Soft Green for Positive
    "NEGATIVE": "#E74C3C",       // Vibrant Red for Negative
    "UME_LORA_STACK": "#9B59B6", // Purple
    "UME_IMAGE": "#DC7633"       // Orange/Brown
};

// Enforce minimum sizes for specific nodes (fixes Nodes 2.0 shrinking issues)
const UME_NODE_SIZES = {
    "UmeAiRT_Positive_Input": [600, 240],
    "UmeAiRT_Negative_Input": [600, 160],
    "UmeAiRT_Prompt": [600, 300],
    "UmeAiRT_Signature": [250, 80]
};

app.registerExtension({
    name: "UmeAiRT.NodeColors",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Apply node colors
        if (UME_NODE_COLORS[nodeData.name]) {
            const colors = UME_NODE_COLORS[nodeData.name];

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                this.color = colors.color;
                this.bgcolor = colors.bgcolor;
            };
        }

        // Apply custom minimum node sizes (Aggressive override for Nodes 2.0)
        if (UME_NODE_SIZES[nodeData.name]) {
            const minSize = UME_NODE_SIZES[nodeData.name];

            // 1. Force size firmly on creation
            const onNodeCreated_sizing = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated_sizing) {
                    onNodeCreated_sizing.apply(this, arguments);
                }
                setTimeout(() => {
                    this.size[0] = Math.max(this.size[0] || 0, minSize[0]);
                    this.size[1] = Math.max(this.size[1] || 0, minSize[1]);
                    this.setDirtyCanvas(true, true);
                }, 100); // Give the DOM Vue engine a moment, then override
            };

            // 2. Override computeSize (LiteGraph standard)
            const computeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function (out) {
                let size = [0, 0];
                if (computeSize) {
                    size = computeSize.apply(this, arguments);
                }
                size[0] = Math.max(size[0] || 0, minSize[0]);
                size[1] = Math.max(size[1] || 0, minSize[1]);
                return size;
            };

            // 3. Override onResize (LiteGraph standard)
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function (size) {
                if (onResize) {
                    onResize.apply(this, arguments);
                }
                if (this.size[0] < minSize[0]) this.size[0] = minSize[0];
                if (this.size[1] < minSize[1]) this.size[1] = minSize[1];
            };

            // 4. Ultimate Defense against Vue 2.0 background-tab crushing
            // If the size is wrong when rendering, snap it back to reality!
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }
                if (this.size[0] < minSize[0] || this.size[1] < minSize[1]) {
                    this.size[0] = Math.max(this.size[0], minSize[0]);
                    this.size[1] = Math.max(this.size[1], minSize[1]);
                    this.setDirtyCanvas(true, true);
                }
            };
        }
    },

    async setup() {
        // Inject UME slot colors into the active theme's connection color map.
        // ComfyUI's theme manager may overwrite this map AFTER setup() runs,
        // so we re-inject periodically until the colors stick.
        function injectSlotColors() {
            if (app.canvas && app.canvas.default_connection_color_byType) {
                const map = app.canvas.default_connection_color_byType;
                for (const [type, color] of Object.entries(UME_SLOT_COLORS)) {
                    // Only inject if the color is missing or was cleared by the theme manager
                    if (!map[type]) {
                        map[type] = color;
                    }
                }
            }
        }

        injectSlotColors();

        // Re-inject for 5 seconds to survive theme application after setup
        let attempts = 0;
        const interval = setInterval(() => {
            injectSlotColors();
            if (++attempts >= 10) clearInterval(interval);
        }, 500);
    }
});

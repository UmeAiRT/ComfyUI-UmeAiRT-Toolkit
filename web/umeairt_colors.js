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
    "UmeAiRT_Latent_Output": { color: "#2C3E50", bgcolor: "#1A252F" }
};

// Connection slot colors - Softer, harmonious palette
const UME_SLOT_COLORS = {
    "UME_FILES": "#5499C7",      // Soft Blue
    "UME_SETTINGS": "#CD8B62",   // Amber/Copper (matches node)
    "UME_PROMPTS": "#52BE80",    // Soft Green  
    "UME_LORA_STACK": "#9B59B6", // Purple
    "UME_IMAGE": "#DC7633"       // Orange/Brown
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
    },

    async setup() {
        // Register custom slot colors
        if (app.canvas && app.canvas.default_connection_color_byType) {
            Object.assign(app.canvas.default_connection_color_byType, UME_SLOT_COLORS);
        }

        // Also try LiteGraph if available
        if (typeof LiteGraph !== "undefined" && LiteGraph.slot_types_default_out) {
            Object.keys(UME_SLOT_COLORS).forEach(type => {
                LiteGraph.slot_types_default_out[type] = UME_SLOT_COLORS[type];
                LiteGraph.slot_types_default_in[type] = UME_SLOT_COLORS[type];
            });
        }
    }
});

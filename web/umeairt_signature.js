import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Fetch the image from our new backend Python route
const SIGNATURE_URL = api.apiURL("/umeairt/signature");

app.registerExtension({
    name: "UmeAiRT.Signature",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UmeAiRT_Signature") {

            // Remove title bar entirely
            nodeType.title_mode = LiteGraph.NO_TITLE;

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Allow resizing to set scale
                this.resizable = true;

                // Keep track of the loaded image
                this.signatureImage = new Image();
                this.imageLoaded = false;

                // Track if we've initialized the size to prevent overriding custom sizing on reload
                this.sizeInitialized = false;

                this.signatureImage.onload = () => {
                    this.imageLoaded = true;
                    // Auto-adjust node size to image aspect ratio natively FIRST time only
                    if (!this.sizeInitialized && this.size[0] <= LiteGraph.NODE_WIDTH && this.size[1] <= 60) {
                        this.size[0] = Math.min(800, this.signatureImage.width);
                        this.size[1] = this.size[0] * (this.signatureImage.height / this.signatureImage.width);
                        this.sizeInitialized = true;
                    }
                    app.graph.setDirtyCanvas(true);
                };

                this.signatureImage.onerror = () => {
                    console.warn(`[UmeAiRT] Signature image not found at ${SIGNATURE_URL}.`);
                };

                // The browser will load this from the web server directory
                this.signatureImage.src = SIGNATURE_URL;

                return r;
            };
        }
    },
    async setup() {
        // Hook into the main LiteGraph canvas draw function to bypass default node rendering
        // specifically for our UmeAiRT Signature node
        if (typeof LGraphCanvas !== 'undefined' && LGraphCanvas.prototype.drawNode) {
            const originalDrawNode = LGraphCanvas.prototype.drawNode;

            LGraphCanvas.prototype.drawNode = function (node, ctx) {
                if (node.type === "UmeAiRT_Signature") {

                    // Force complete transparency for default rendering properties
                    node.bgcolor = "transparent";
                    node.color = "transparent";
                    const originalBoxColor = node.boxcolor;
                    node.boxcolor = "transparent";

                    // Run the original drawNode to handle selections, etc (but invisible background)
                    const result = originalDrawNode.apply(this, arguments);

                    // Now manually draw our image over everything, bypassing the constraints
                    if (node.imageLoaded && node.signatureImage) {
                        ctx.save();
                        ctx.beginPath();
                        ctx.rect(0, 0, node.size[0], node.size[1]);
                        ctx.clip(); // Ensure it doesn't bleed outside the resize box
                        ctx.drawImage(node.signatureImage, 0, 0, node.size[0], node.size[1]);
                        ctx.restore();
                    } else if (!node.imageLoaded) {
                        // Outline when empty so user knows it's there
                        ctx.save();
                        ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
                        ctx.setLineDash([5, 5]);
                        ctx.strokeRect(0, 0, node.size[0], node.size[1]);
                        ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
                        ctx.font = "14px Arial";
                        ctx.fillText("Replace assets/signature.png", 10, node.size[1] / 2);
                        ctx.restore();
                    }

                    // Restore properties
                    node.boxcolor = originalBoxColor;
                    return result;

                } else {
                    // Normal node behavior
                    return originalDrawNode.apply(this, arguments);
                }
            };
        }
    }
});

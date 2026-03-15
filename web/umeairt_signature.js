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

            // Per-node drawing: replaces the global LGraphCanvas.prototype.drawNode patch
            // This avoids a per-render type check for every node on the canvas.
            nodeType.prototype.onDrawBackground = function (ctx) {
                // Force complete transparency — skip default background rendering
                this.bgcolor = "transparent";
                this.color = "transparent";
            };

            nodeType.prototype.onDrawForeground = function (ctx) {
                if (this.imageLoaded && this.signatureImage) {
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(0, 0, this.size[0], this.size[1]);
                    ctx.clip();
                    ctx.drawImage(this.signatureImage, 0, 0, this.size[0], this.size[1]);
                    ctx.restore();
                } else if (!this.imageLoaded) {
                    // Outline when empty so user knows it's there
                    ctx.save();
                    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
                    ctx.setLineDash([5, 5]);
                    ctx.strokeRect(0, 0, this.size[0], this.size[1]);
                    ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
                    ctx.font = "14px Arial";
                    ctx.fillText("Replace assets/signature.png", 10, this.size[1] / 2);
                    ctx.restore();
                }
            };
        }
    },
});

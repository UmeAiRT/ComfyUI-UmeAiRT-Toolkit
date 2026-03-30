import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Fetch the image from our backend Python route
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

                this.resizable = true;
                this._signatureReady = false;

                // Delay widget creation so mouse click for node placement is not blocked
                const self = this;
                setTimeout(() => {
                    if (self._signatureReady) return;
                    self._signatureReady = true;

                    const container = document.createElement("div");
                    container.style.width = "100%";
                    container.style.height = "100%";
                    container.style.display = "flex";
                    container.style.alignItems = "center";
                    container.style.justifyContent = "center";
                    container.style.overflow = "hidden";
                    container.style.background = "transparent";
                    container.style.pointerEvents = "none";

                    const img = document.createElement("img");
                    img.src = SIGNATURE_URL;
                    img.style.width = "100%";
                    img.style.height = "auto";
                    img.style.objectFit = "contain";
                    img.style.pointerEvents = "none";
                    img.style.userSelect = "none";

                    container.appendChild(img);

                    self.addDOMWidget("signature_display", "custom", container, {
                        hideOnZoom: false,
                        serialize: false,
                    });

                    img.onload = () => {
                        if (self.size[0] <= LiteGraph.NODE_WIDTH && self.size[1] <= 60) {
                            const w = Math.min(800, img.naturalWidth);
                            const h = w * (img.naturalHeight / img.naturalWidth) + 10;
                            self.size[0] = w;
                            self.size[1] = h;
                        }
                        app.graph.setDirtyCanvas(true);
                    };

                    img.onerror = () => {
                        console.warn(`[UmeAiRT] Signature image not found at ${SIGNATURE_URL}.`);
                        container.textContent = "Replace assets/signature.png";
                        container.style.color = "rgba(255, 255, 255, 0.4)";
                        container.style.fontSize = "14px";
                    };
                }, 500);

                return r;
            };

            // Transparent background for both legacy Canvas and Vue renderers
            nodeType.prototype.onDrawBackground = function (ctx) {
                this.bgcolor = "transparent";
                this.color = "transparent";
            };
        }
    },
});

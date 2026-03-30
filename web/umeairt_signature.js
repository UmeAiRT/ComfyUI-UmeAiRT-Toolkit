import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Fetch the image from our backend Python route
const SIGNATURE_URL = api.apiURL("/umeairt/signature");

function setupSignatureWidget(node) {
    if (node._signatureReady) return;
    node._signatureReady = true;

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

    node.addDOMWidget("signature_display", "custom", container, {
        hideOnZoom: false,
        serialize: false,
    });

    img.onload = () => {
        if (node.size[0] <= LiteGraph.NODE_WIDTH && node.size[1] <= 60) {
            const w = Math.min(800, img.naturalWidth);
            const h = w * (img.naturalHeight / img.naturalWidth) + 10;
            node.size[0] = w;
            node.size[1] = h;
        }
        app.graph.setDirtyCanvas(true);
    };

    img.onerror = () => {
        console.warn(`[UmeAiRT] Signature image not found at ${SIGNATURE_URL}.`);
        container.textContent = "Replace assets/signature.png";
        container.style.color = "rgba(255, 255, 255, 0.4)";
        container.style.fontSize = "14px";
    };
}

app.registerExtension({
    name: "UmeAiRT.Signature",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UmeAiRT_Signature") {

            // Remove title bar entirely
            nodeType.title_mode = LiteGraph.NO_TITLE;

            // --- First creation (from search menu) ---
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.resizable = true;
                this._signatureReady = false;
                // Delay to allow placement click to register first
                const self = this;
                setTimeout(() => setupSignatureWidget(self), 500);
                return r;
            };

            // --- Restore from workflow load / tab switch ---
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                if (onConfigure) onConfigure.apply(this, arguments);
                this.resizable = true;
                this._signatureReady = false;
                setupSignatureWidget(this);
            };

            // Transparent background for both legacy Canvas and Vue renderers
            nodeType.prototype.onDrawBackground = function (ctx) {
                this.bgcolor = "transparent";
                this.color = "transparent";
            };
        }
    },
});

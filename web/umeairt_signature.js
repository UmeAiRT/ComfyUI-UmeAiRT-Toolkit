import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Fetch the image from our backend Python route
const SIGNATURE_URL = api.apiURL("/umeairt/signature");

function setupSignatureWidget(node) {
    // Check if widget already exists and its DOM element is still connected
    const existing = node.widgets?.find(w => w.name === "signature_display");
    if (existing && existing.element && existing.element.isConnected) {
        return; // Widget is alive, nothing to do
    }

    // Remove stale widget reference if DOM was destroyed by Vue re-render
    if (existing) {
        const idx = node.widgets.indexOf(existing);
        if (idx >= 0) node.widgets.splice(idx, 1);
    }

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
        if (node.size[0] <= LiteGraph.NODE_WIDTH && node.size[1] <= 80) {
            const w = Math.min(800, img.naturalWidth);
            const h = w * (img.naturalHeight / img.naturalWidth) + 10;
            node.size[0] = w;
            node.size[1] = h;
        }
        app.graph.setDirtyCanvas(true);
    };

    img.onerror = () => {
        console.warn(`[UmeAiRT] Signature image not found at ${SIGNATURE_URL}.`);
    };
}

app.registerExtension({
    name: "UmeAiRT.Signature",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UmeAiRT_Signature") {

            nodeType.title_mode = LiteGraph.NO_TITLE;

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.resizable = true;
                // Delay widget creation so the placement click is not blocked
                const self = this;
                setTimeout(() => setupSignatureWidget(self), 500);
                return r;
            };

            // Transparent background
            nodeType.prototype.onDrawBackground = function (ctx) {
                this.bgcolor = "transparent";
                this.color = "transparent";
            };
        }
    },

    // Periodic watchdog: re-creates destroyed widgets (tab switch, workflow load)
    async setup() {
        setInterval(() => {
            if (!app.graph?._nodes) return;
            for (const node of app.graph._nodes) {
                if (node.comfyClass === "UmeAiRT_Signature" || node.type === "UmeAiRT_Signature") {
                    setupSignatureWidget(node);
                }
            }
        }, 1500);
    }
});

import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "UmeAiRT.LogViewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UmeAiRT_Log_Viewer") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                // message is returned via standard ComfyUI "user interface" protocol
                if (message && message.text && message.text.length > 0) {
                    const text = message.text[0];

                    // Try to find existing widget
                    let w = this.widgets?.find((w) => w.name === "log_output");

                    if (!w) {
                        // Create it using the standard ComfyWidget factory if not found
                        // This handles DOM element creation, multiline setup, etc. automatically
                        const widget = ComfyWidgets.STRING(this, "log_output", ["STRING", { multiline: true }], app).widget;
                        w = widget;
                        w.inputEl.readOnly = true;
                        w.inputEl.style.opacity = 0.8;
                        w.inputEl.style.fontSize = "12px";
                    }

                    if (w) {
                        w.value = text;
                        // Also update DOM directly if possible
                        if (w.inputEl) {
                            w.inputEl.value = text;
                        }

                        this.onResize?.(this.size);
                        app.graph.setDirtyCanvas(true, true);
                    }
                }
            };
        }
    },
});

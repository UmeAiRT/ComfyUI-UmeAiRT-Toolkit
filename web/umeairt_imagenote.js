import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "UmeAiRT.ImageNote",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UmeAiRT_ImageNote") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Find the image widget
                const imageWidget = this.widgets.find(w => w.name === "image");

                if (imageWidget) {
                    // Force the node to update its image display when the widget value changes
                    const callback = imageWidget.callback;
                    imageWidget.callback = function () {
                        if (callback) {
                            callback.apply(this, arguments);
                        }

                        // Tell ComfyUI to preview this image
                        const node = this;
                        // For nodes without prompt execution, we can manually populate node.imgs
                        if (imageWidget.value) {
                            const imgUrl = api.apiURL(`/view?filename=${encodeURIComponent(imageWidget.value)}&type=input&subfolder=`);

                            if (node.imgs && node.imgs.length > 0) {
                                node.imgs[0].src = imgUrl;
                            } else {
                                const img = new Image();
                                img.onload = () => {
                                    node.imgs = [img];
                                    app.graph.setDirtyCanvas(true);
                                };
                                img.src = imgUrl;
                            }
                        }
                    }.bind(this); // Bind to the node instance
                }

                // Initial load
                setTimeout(() => {
                    if (imageWidget && imageWidget.value) {
                        imageWidget.callback(imageWidget.value);
                    }
                }, 100);

                return r;
            };
        }
    }
});

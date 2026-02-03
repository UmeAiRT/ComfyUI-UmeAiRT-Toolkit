# 🌌 ComfyUI UmeAiRT Toolkit

**A Wireless, Nodes 2.0 Ready, and Aesthetic Toolkit for ComfyUI.**

Stop fighting with "noodle soup"! The UmeAiRT Toolkit provides a suite of "Wireless" nodes that share a global state, allowing you to build clean, professional, and readable workflows.

![Banner](https://github.com/user-attachments/assets/placeholder.png)

## ✨ Key Features

### 📡 Wireless Architecture
- **Setters & Getters**: Define your inputs (Model, VAE, CLIP, Steps, CFG, etc.) in one place using "Setter" nodes. Retrieve them anywhere using "Getter" nodes or autonomous Processors.
- **Global State**: All parameters are synchronized globally. No need to drag long wires across your canvas.

### 🧠 Smart Automation
- **Autonomous KSampler**: Automatically detects your workflow mode:
    - **Txt2Img**: If normal generation.
    - **Img2Img**: Automatically switches if `denoise < 1.0`, fetching the source image and encoding it without manual wiring.
- **Smart Image Saver**: Automatically resolves efficient paths (`SDXL/Date/Time_Model_Seed`) and fetches generation metadata (Steps, CFG, Model Name, LoRAs) without connecting distinct inputs.

### 🎨 Clean & Future-Proof UI
- **Nodes 2.0 Ready**: Built using standard ComfyUI widgets (Python-based) to ensure 100% compatibility with future ComfyUI updates (Frontend V2).
- **Minimalist Design**: Custom "Label" nodes and hidden widget labels create a sleek, professional look.

## 📦 Nodes Overview

| Category | Node | Description |
| :--- | :--- | :--- |
| **Variables** | `Wireless Checkpoint Loader` | Loads Model/CLIP/VAE and broadcasts them wirelessly. |
| **Loaders** | `Wireless Image Loader` | Loads a source image for Img2Img/ControlNet. |
| **Variables** | `Global Params` | Nodes to set Steps, CFG, Sampler, Scheduler, ISO, etc. |
| **Generation** | `Wireless KSampler` | The magic node. Auto-generates based on global state. |
| | `Wireless UltimateUpscale` | Wrapper for USDU using wireless inputs. |
| | `Wireless FaceDetailer` | Wrapper for FaceDetailer using wireless inputs. |
| **Tools** | `Label` | Aesthetic minimal sticky notes for documentation. |
| | `Wireless Image Saver` | Saves images with auto-generated paths and metadata. |
| | `Wireless Debug` | Inspect the current state of your global variables. |

## 🚀 Installation

### Option A: ComfyUI Manager (Recommended)
1.  Open **ComfyUI Manager**.
2.  Search for `UmeAiRT Toolkit`.
3.  Click **Install**.

### Option B: Manual Installation
1.  Navigate to your ComfyUI `custom_nodes` directory.
2.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/ComfyUI-UmeAiRT-Toolkit.git
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Restart ComfyUI.

## 🛠️ Usage Guide

1.  **Setup Phase**: Place "Setter" nodes (Checkpoint Loader, Steps, Image Size, etc.) on the left side of your workflow.
2.  **Logic Phase**: Place a `Wireless KSampler` on the right.
    *   *No wires needed between them!*
3.  **Img2Img**: To switch to Img2Img, simply add a `Wireless Image Loader` and lower the `Denoise` slider below 1.0. The Sampler handles the rest.

## ❤️ Credits

Developed by **UmeAiRT Team**.
Designed to simplify complex workflows while maintaining maximum power and flexibility.

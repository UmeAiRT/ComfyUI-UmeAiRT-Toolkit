"""Progress tracking module for SeedVR2 upscaling operations."""

try:
    from logger import log_node
except ImportError:
    def log_node(msg, **kwargs):
        print(f"[UmeAiRT-Toolkit] {msg}")

try:
    from server import PromptServer
    from comfy_execution.utils import get_executing_context
    from comfy_execution.progress import get_progress_state
    WEBSOCKET_PROGRESS_AVAILABLE = True
except ImportError:
    WEBSOCKET_PROGRESS_AVAILABLE = False

try:
    import comfy.utils
    LEGACY_PROGRESS_AVAILABLE = True
except ImportError:
    LEGACY_PROGRESS_AVAILABLE = False


class Progress:
    def __init__(self, total_steps, sub_steps_per_tile=3):
        self.total_steps = total_steps
        self.sub_steps_per_tile = sub_steps_per_tile
        self.total_sub_steps = total_steps * sub_steps_per_tile
        self.current_step = 0
        self.current_sub_step = 0

        self.context = None
        self.node_id = None
        self.prompt_id = None

        if WEBSOCKET_PROGRESS_AVAILABLE:
            try:
                self.context = get_executing_context()
                if self.context:
                    self.node_id = self.context.node_id
                    self.prompt_id = self.context.prompt_id
            except Exception as e:
                log_node(f"Could not get execution context: {e}", color="YELLOW")

        log_node(f"SeedVR2 Upscale: Starting - {total_steps} tiles to process", color="CYAN")

    def _send_websocket_progress(self):
        if not WEBSOCKET_PROGRESS_AVAILABLE or not self.node_id:
            return

        try:
            progress_state = get_progress_state()
            if progress_state:
                progress_state.update_progress(
                    self.node_id,
                    self.current_sub_step,
                    self.total_sub_steps
                )

            if hasattr(PromptServer, 'instance') and PromptServer.instance:
                progress_text = f"Tile {self.current_step}/{self.total_steps} • Step {self.current_sub_step}/{self.total_sub_steps}"
                PromptServer.instance.send_progress_text(progress_text, self.node_id)

        except Exception as e:
            log_node(f"WebSocket progress failed: {e}", color="YELLOW")

    def update(self, sub_progress_step=None):
        if sub_progress_step is not None:
            self.current_sub_step = self.current_step * self.sub_steps_per_tile + sub_progress_step
        else:
            self.current_step += 1
            self.current_sub_step = self.current_step * self.sub_steps_per_tile

        main_percentage = (self.current_step / self.total_steps) * 100
        progress_bar = "█" * int(main_percentage // 5) + "░" * (20 - int(main_percentage // 5))
        log_node(f"SeedVR2 Tile {self.current_step}/{self.total_steps} [{progress_bar}] {main_percentage:.1f}%", color="CYAN")

        self._send_websocket_progress()

        if LEGACY_PROGRESS_AVAILABLE and not WEBSOCKET_PROGRESS_AVAILABLE:
            try:
                comfy.utils.report_progress(self.current_sub_step, self.total_sub_steps)
            except Exception:
                pass

        if self.current_step == self.total_steps:
            log_node(f"SeedVR2 Upscale completed! Processed {self.total_steps} tiles", color="GREEN")

    def update_sub_progress(self, step_name, step_number):
        self.current_sub_step = self.current_step * self.sub_steps_per_tile + step_number

        if WEBSOCKET_PROGRESS_AVAILABLE and self.node_id:
            try:
                progress_state = get_progress_state()
                if progress_state:
                    progress_state.update_progress(
                        self.node_id,
                        self.current_sub_step,
                        self.total_sub_steps
                    )

                if hasattr(PromptServer, 'instance') and PromptServer.instance:
                    detailed_text = f"Tile {self.current_step}/{self.total_steps} • {step_name} ({step_number}/{self.sub_steps_per_tile})"
                    PromptServer.instance.send_progress_text(detailed_text, self.node_id)

            except Exception:
                pass

        elif LEGACY_PROGRESS_AVAILABLE:
            try:
                comfy.utils.report_progress(self.current_sub_step, self.total_sub_steps)
            except Exception:
                pass

    def initialize_websocket_progress(self):
        if WEBSOCKET_PROGRESS_AVAILABLE:
            try:
                context = get_executing_context()
                if context:
                    node_id = context.node_id

                    progress_state = get_progress_state()
                    if progress_state:
                        progress_state.update_progress(node_id, 0, 100)

                    if hasattr(PromptServer, 'instance') and PromptServer.instance:
                        PromptServer.instance.send_progress_text("Initializing SeedVR2 upscale...", node_id)

            except Exception as e:
                log_node(f"Could not initialize WebSocket progress: {e}", color="YELLOW")

    def finalize_websocket_progress(self):
        if WEBSOCKET_PROGRESS_AVAILABLE and self.node_id:
            try:
                progress_state = get_progress_state()
                if progress_state:
                    progress_state.update_progress(self.node_id, self.total_sub_steps, self.total_sub_steps)

                if hasattr(PromptServer, 'instance') and PromptServer.instance:
                    PromptServer.instance.send_progress_text("SeedVR2 Upscale completed!", self.node_id)

            except Exception as e:
                log_node(f"Could not finalize WebSocket progress: {e}", color="YELLOW")

        elif LEGACY_PROGRESS_AVAILABLE:
            try:
                comfy.utils.report_progress(self.total_sub_steps, self.total_sub_steps)
            except Exception:
                pass

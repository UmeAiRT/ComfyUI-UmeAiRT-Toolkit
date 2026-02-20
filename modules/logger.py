"""
UmeAiRT Toolkit - Centralized Logger
------------------------------------
Provides a standardized logging interface with color support using colorama.
"""

try:
    import colorama
    from colorama import Fore, Style
    colorama.init(convert=True, autoreset=True)
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    RED = Fore.LIGHTRED_EX
    MAGENTA = Fore.LIGHTMAGENTA_EX
    YELLOW = Fore.LIGHTYELLOW_EX
    ORANGE = "\033[38;5;208m" # True ANSI Orange (208) for modern terminals, avoiding yellow confusion
    RESET = Style.RESET_ALL
except ImportError:
    CYAN = GREEN = RED = MAGENTA = YELLOW = ORANGE = RESET = ""

class UmeAiRT_Logger:
    def __init__(self, max_buffer=100):
        self.logs = []
        self.max_buffer = max_buffer
    
    def log(self, msg, color=None, prefix="UmeAiRT-Toolkit"):
        c = ""
        if color == "GREEN": c = GREEN
        elif color == "CYAN": c = MAGENTA # Kept for legacy compatibility
        elif color == "RED": c = RED
        elif color == "YELLOW": c = YELLOW
        elif color == "ORANGE": c = ORANGE
        
        # Auto-parse Node Name (before first colon) to color it Yellow
        if ":" in msg:
            parts = msg.split(":", 1)
            node_name = f"{YELLOW}{parts[0]}:{RESET}"
            rest_of_msg = f"{c}{parts[1]}{RESET}" if c else f"{parts[1]}"
            formatted_msg = f"[{CYAN}{prefix}{RESET}] {node_name}{rest_of_msg}"
        else:
            formatted_msg = f"[{CYAN}{prefix}{RESET}] {c}{msg}{RESET}" if c else f"[{CYAN}{prefix}{RESET}] {msg}"
            
        print(formatted_msg)
        
        # Store clean message for viewing
        clean_msg = f"[{prefix}] {msg}"
        self.logs.append(clean_msg)
        if len(self.logs) > self.max_buffer:
            self.logs.pop(0)

    def get_logs(self, limit=20):
        return self.logs[-limit:]

# Global Instance
logger = UmeAiRT_Logger()

def log_node(msg, color=None, prefix="UmeAiRT-Toolkit"):
    """Standardized logger for UmeAiRT nodes. Redirects to global logger instance."""
    logger.log(msg, color, prefix)


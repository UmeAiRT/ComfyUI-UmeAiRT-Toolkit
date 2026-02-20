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
    RESET = Style.RESET_ALL
except ImportError:
    CYAN = GREEN = RED = MAGENTA = YELLOW = RESET = ""

class UmeAiRT_Logger:
    def __init__(self, max_buffer=100):
        self.logs = []
        self.max_buffer = max_buffer
    
    def log(self, msg, color=None, prefix="UmeAiRT-Toolkit"):
        c = ""
        if color == "GREEN": c = GREEN
        elif color == "CYAN": c = MAGENTA
        elif color == "RED": c = RED
        elif color == "YELLOW": c = YELLOW
        
        formatted_msg = f"[{CYAN}{prefix}{RESET}] {c}{msg}{RESET}"
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


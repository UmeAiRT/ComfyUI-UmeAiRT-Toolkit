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
    """Centralized logger class to handle stylized output and store history.

    Attributes:
        logs (list): A list storing the most recent formatted log messages.
        max_buffer (int): The maximum number of log messages retained in memory.
    """

    def __init__(self, max_buffer=100):
        """Initializes the UmeAiRT_Logger with a specific buffer size.

        Args:
            max_buffer (int, optional): The limit of log entries to keep. Defaults to 100.
        """
        self.logs = []
        self.max_buffer = max_buffer
    
    def log(self, msg, color=None, prefix="UmeAiRT-Toolkit"):
        """Prints and stores a colored log message to the console.

        Automatically formats node names (text before the first colon) in yellow
        for better readability in the terminal.

        Args:
            msg (str): The main message content to log.
            color (str, optional): The desired color for the message (e.g., "GREEN", "RED"). Defaults to None.
            prefix (str, optional): The prefix tag displayed before the message. Defaults to "UmeAiRT-Toolkit".
        """
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
        """Retrieves the most recent log entries.

        Args:
            limit (int, optional): The number of recent logs to return. Defaults to 20.

        Returns:
            list: A list of string containing the most recent logs.
        """
        return self.logs[-limit:]

# Global Instance
logger = UmeAiRT_Logger()

def log_node(msg, color=None, prefix="UmeAiRT-Toolkit"):
    """Standardized logger wrapper for UmeAiRT nodes. 
    
    Redirects to the global logger instance method to ensure consistent formatting.

    Args:
        msg (str): The main message content to log.
        color (str, optional): The desired color for the message (e.g., "GREEN", "RED"). Defaults to None.
        prefix (str, optional): The prefix tag displayed before the message. Defaults to "UmeAiRT-Toolkit".
    """
    logger.log(msg, color, prefix)


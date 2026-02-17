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

def log_node(msg, color=None, prefix="UmeAiRT-Toolkit"):
    """Standardized logger for UmeAiRT nodes."""
    c = ""
    if color == "GREEN": c = GREEN
    elif color == "CYAN": c = MAGENTA # User requested change from Cyan to avoid matching prefix
    elif color == "RED": c = RED
    elif color == "YELLOW": c = YELLOW
    
    print(f"[{CYAN}{prefix}{RESET}] {c}{msg}{RESET}")

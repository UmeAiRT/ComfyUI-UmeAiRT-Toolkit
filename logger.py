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
    RED = Fore.RED
    RESET = Style.RESET_ALL
except ImportError:
    CYAN = GREEN = RED = RESET = ""

def log_node(msg, color=None, prefix="UmeAiRT"):
    """Standardized logger for UmeAiRT nodes."""
    c = ""
    if color == "GREEN": c = GREEN
    elif color == "CYAN": c = CYAN
    elif color == "RED": c = RED
    elif color == "YELLOW": c = Fore.YELLOW if 'colorama' in globals() else ""
    
    print(f"[{CYAN}{prefix}{RESET}] {c}{msg}{RESET}")

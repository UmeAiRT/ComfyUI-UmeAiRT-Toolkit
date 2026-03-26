"""Download utilities for the UmeAiRT Toolkit.

Provides file download functionality with aria2c acceleration, urllib fallback,
SHA256 verification, and HuggingFace token management.
"""
import os
import hashlib
import urllib.request
import comfy.utils
from .common import log_node
from .logger import log_progress


def get_hf_token() -> str:
    """Retrieve HuggingFace token from environment or cache file.

    Checks in order:
    1. HF_TOKEN environment variable
    2. ~/.cache/huggingface/token file

    Returns:
        str: The token string, or empty string if not found.
    """
    # 1. Environment variable
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token

    # 2. HuggingFace cache file
    hf_token_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")
    if os.path.isfile(hf_token_path):
        try:
            with open(hf_token_path, 'r', encoding='utf-8') as f:
                token = f.read().strip()
            if token:
                return token
        except Exception as e:
            log_node(f"Bundle Loader: Could not read HF token from cache: {e}", color="YELLOW")

    return ""


def verify_file_hash(path, expected_sha256):
    """Verify a downloaded file's SHA256 hash against an expected value.

    Args:
        path (str): Path to the file to verify.
        expected_sha256 (str): Expected SHA256 hex digest. If empty/None, skip.

    Returns:
        bool: True if hash matches or verification was skipped, False on mismatch.
    """
    if not expected_sha256:
        return True
    filename = os.path.basename(path)
    log_node(f"Bundle Loader: Verifying SHA256 for '{filename}'...")
    sha256 = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()
        if actual == expected_sha256:
            log_node(f"Bundle Loader: ✅ '{filename}' hash verified.", color="GREEN")
            return True
        else:
            log_node(
                f"Bundle Loader: ⚠️ '{filename}' hash MISMATCH!\n"
                f"  Expected: {expected_sha256}\n"
                f"  Actual:   {actual}",
                color="RED"
            )
            return False
    except Exception as e:
        log_node(f"Bundle Loader: Hash verification error for '{filename}': {e}", color="YELLOW")
        return False


def _find_aria2c():
    """Find aria2c executable, searching PATH and common Windows install locations.

    ComfyUI's embedded Python often doesn't inherit the user's system PATH,
    so we also search common installation directories.

    Returns:
        str or None: Full path to aria2c executable, or None if not found.
    """
    import shutil
    import subprocess

    # 0. Check for vendored binary bundled with the toolkit
    vendor_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor", "aria2")
    vendor_exe = os.path.join(vendor_dir, "aria2c.exe") if os.name == "nt" else os.path.join(vendor_dir, "aria2c")
    if os.path.isfile(vendor_exe):
        try:
            result = subprocess.run([vendor_exe, "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return vendor_exe
        except Exception as e:
            log_node(f"Bundle Loader: Vendored aria2c check failed: {e}", color="YELLOW")

    # 1. Try shutil.which (works if aria2c is on the current PATH)
    path = shutil.which("aria2c")
    if path:
        return path

    # 2. Search common Windows install locations
    if os.name == "nt":
        candidates = []
        home = os.path.expanduser("~")
        localappdata = os.environ.get("LOCALAPPDATA", os.path.join(home, "AppData", "Local"))
        programfiles = os.environ.get("ProgramFiles", r"C:\Program Files")
        programfilesx86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

        # Scoop
        candidates.append(os.path.join(home, "scoop", "shims", "aria2c.exe"))
        candidates.append(os.path.join(home, "scoop", "apps", "aria2", "current", "aria2c.exe"))
        # Chocolatey
        candidates.append(r"C:\ProgramData\chocolatey\bin\aria2c.exe")
        # Standalone / manual install
        candidates.append(os.path.join(localappdata, "aria2", "aria2c.exe"))
        candidates.append(os.path.join(programfiles, "aria2", "aria2c.exe"))
        candidates.append(os.path.join(programfilesx86, "aria2", "aria2c.exe"))
        candidates.append(r"C:\aria2\aria2c.exe")
        # winget (typically goes into LOCALAPPDATA\Microsoft\WinGet\...)
        winget_dir = os.path.join(localappdata, "Microsoft", "WinGet", "Packages")
        if os.path.isdir(winget_dir):
            for d in os.listdir(winget_dir):
                if "aria2" in d.lower():
                    candidate = os.path.join(winget_dir, d, "aria2c.exe")
                    candidates.append(candidate)

        for candidate in candidates:
            if os.path.isfile(candidate):
                # Verify it actually runs
                try:
                    result = subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
                    if result.returncode == 0:
                        return candidate
                except Exception as e:
                    log_node(f"Bundle Loader: aria2c candidate '{candidate}' failed: {e}", color="YELLOW")

    # 3. Last resort: just try running it (maybe it's on a PATH we missed)
    try:
        result = subprocess.run(["aria2c", "--version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            return "aria2c"
    except Exception as e:
        log_node(f"Bundle Loader: aria2c not found on PATH: {e}", color="YELLOW")

    return None


# Cache: None = not checked yet, False = not available, str = path to aria2c
_ARIA2_PATH = None


def _download_with_aria2(url, dest_path, connections=8, hf_token=""):
    """Download a file using aria2c for multi-connection acceleration.

    Uses Popen to run aria2c in the background while polling file size
    to update ComfyUI's progress bar and log periodic progress.

    Args:
        url (str): The full URL to download.
        dest_path (str): The local path to save to.
        connections (int): Number of parallel connections (default: 8).
        hf_token (str): Optional HuggingFace token for authentication.

    Returns:
        bool: True if download succeeded, False otherwise.
    """
    import subprocess
    import time
    filename = os.path.basename(dest_path)
    dest_dir = os.path.dirname(dest_path)

    # Get total file size via HEAD request for progress tracking
    total_size = 0
    try:
        headers = {"User-Agent": "ComfyUI-UmeAiRT-Toolkit"}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
    except Exception as e:
        pass  # Non-critical: aria2c handles its own progress

    size_mb = f" ({total_size / 1024 / 1024:.0f} MB)" if total_size else ""
    log_node(f"Bundle Loader: Downloading '{filename}'{size_mb} via aria2c ({connections} connections)...")

    try:
        aria2_exe = _ARIA2_PATH
        cmd = [
            aria2_exe,
            "--dir=" + dest_dir,
            "--out=" + filename,
            "--split=" + str(connections),
            "--max-connection-per-server=" + str(connections),
            "--min-split-size=1M",
            "--continue=true",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--console-log-level=notice",
            "--summary-interval=1",
            "--human-readable=true",
        ]
        if hf_token:
            cmd.extend(["--header", "Authorization: Bearer " + hf_token])
        cmd.append(url)

        import re
        # Regex to parse aria2c progress output: [#id 50MiB/100MiB(50%)]
        pct_re = re.compile(r'\((\d+)%\)')

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pbar = comfy.utils.ProgressBar(100)

        # Read aria2c output line by line in real-time
        for raw_line in iter(process.stdout.readline, b''):
            line = raw_line.decode(errors="ignore").strip()
            if not line:
                continue
            # Parse percentage from lines containing (XX%)
            match = pct_re.search(line)
            if match:
                pct = int(match.group(1))
                pbar.update_absolute(pct)
                log_progress(filename, pct)

        process.wait()
        returncode = process.returncode

        # Finalize progress bar line and UI bar
        log_progress(filename, 100, done=True)
        pbar.update_absolute(100)

        if returncode == 0 and os.path.exists(dest_path):
            log_node(f"Bundle Loader: '{filename}' downloaded via aria2c.", color="GREEN")
            return True
        else:
            stderr = process.stderr.read().decode(errors="ignore").strip() if process.stderr else "no details"
            log_node(f"Bundle Loader: aria2c failed (code {returncode}: {stderr}), falling back to urllib.", color="YELLOW")
            return False

    except Exception as e:
        log_node(f"Bundle Loader: aria2c error: {e}, falling back to urllib.", color="YELLOW")
        return False


def _download_with_urllib(url, dest_path, hf_token="", timeout=300):
    """Download a file with urllib and a ComfyUI progress bar (fallback).

    Args:
        url (str): The full URL to download.
        dest_path (str): The local path to save to.
        hf_token (str): Optional HuggingFace token for authentication.
        timeout (int): Socket timeout in seconds (default: 300). Increase for very
            large files on slow connections.
    """
    filename = os.path.basename(dest_path)
    temp_path = dest_path + ".download"
    log_node(f"Bundle Loader: Downloading '{filename}' via urllib...")

    headers = {"User-Agent": "ComfyUI-UmeAiRT-Toolkit"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        pbar = comfy.utils.ProgressBar(total_size) if total_size > 0 else None

        with open(temp_path, "wb") as f:
            while True:
                chunk = response.read(8192 * 1024)  # 8MB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if pbar:
                    pbar.update_absolute(downloaded)

    # Rename temp file to final destination
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)
    log_node(f"Bundle Loader: '{filename}' downloaded successfully.", color="GREEN")


def download_file(url, dest_path, hf_token="", expected_sha256=""):
    """Download a file, preferring aria2c for speed with urllib as fallback.

    After a successful download, verifies the file's SHA256 hash if provided.

    Args:
        url (str): The full URL to download.
        dest_path (str): The local path to save to.
        hf_token (str): Optional HuggingFace token for authentication.
        expected_sha256 (str): Optional SHA256 hash to verify after download.

    Raises:
        RuntimeError: If all download methods fail.
    """
    global _ARIA2_PATH
    filename = os.path.basename(dest_path)

    try:
        # Try aria2c first (much faster for large model files)
        if _ARIA2_PATH is None:
            _ARIA2_PATH = _find_aria2c() or False
            if _ARIA2_PATH:
                log_node("Bundle Loader: aria2c detected — using accelerated downloads.", color="GREEN")
            else:
                hint = "Run: apt install aria2 (Linux) or bundled in vendor/aria2/ (Windows)" if os.name != "nt" else "bundled binary not found in vendor/aria2/"
                log_node(f"Bundle Loader: aria2c not found — using urllib. {hint}", color="YELLOW")

        if _ARIA2_PATH:
            if _download_with_aria2(url, dest_path, hf_token=hf_token):
                verify_file_hash(dest_path, expected_sha256)
                return

        # Fallback to urllib
        _download_with_urllib(url, dest_path, hf_token=hf_token)
        verify_file_hash(dest_path, expected_sha256)

    except Exception as e:
        # Clean up partial downloads
        for cleanup in [dest_path + ".download", dest_path + ".aria2"]:
            if os.path.exists(cleanup):
                try:
                    os.remove(cleanup)
                except Exception as e:
                    log_node(f"Bundle Loader: Cleanup of '{cleanup}' failed: {e}", color="YELLOW")
        raise RuntimeError(f"Bundle Loader: Failed to download '{filename}': {e}")

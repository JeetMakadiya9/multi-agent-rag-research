"""Start the Research Studio backend and the dashboard on this computer, and stop both together.

    python -m dashboard.launch [--studio-port 8765] [--dashboard-port 8501] [--no-browser] [--check]

``start_studio.bat`` runs this with the project's virtual environment. Both services listen on 127.0.0.1 only. The
launcher installs nothing: missing packages, a missing model or a busy port are reported with what to do.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = {"streamlit": "streamlit", "requests": "requests", "pypdf": "pypdf", "numpy": "numpy", "faiss": "faiss-cpu",
            "rank_bm25": "rank-bm25", "sentence_transformers": "sentence-transformers",
            "langchain_text_splitters": "langchain-text-splitters"}


def say(message: str) -> None:
    print(f"[studio] {message}", flush=True)


def missing_packages() -> List[str]:
    return [package for module, package in REQUIRED.items() if importlib.util.find_spec(module) is None]


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def studio_status(port: int, timeout: float = 3.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=timeout) as response:
            data = json.loads(response.read())
        return data if data.get("app") == "Multi-Agent Research Studio" else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def dashboard_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=3) as response:
            return response.read().strip() == b"ok"
    except (urllib.error.URLError, OSError):
        return False


def wait_for(check, seconds: float, process: subprocess.Popen) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        if process.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def stop(processes: List[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="start_studio", description=__doc__.split("\n\n")[0])
    parser.add_argument("--studio-port", type=int, default=8765)
    parser.add_argument("--dashboard-port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true", help="do not open the dashboard in a browser")
    parser.add_argument("--check", action="store_true", help="only check the prerequisites, start nothing")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    if sys.version_info[:2] != (3, 12):
        say(f"Python {sys.version.split()[0]} found; the project is developed and tested on Python 3.12.")
    missing = missing_packages()
    if missing:
        say("Missing packages: " + ", ".join(missing))
        say(r"Install them with:  .venv\Scripts\python.exe -m pip install -r requirements.txt")
        return 1

    from webrag.model_service import ModelConfig, allow_model_downloads, load_env_file
    names = load_env_file()
    say(".env read (" + ", ".join(names) + ")" if names else "No .env file; using environment settings or the defaults.")
    config = ModelConfig.from_env()
    problems = config.problems()
    if problems:
        for problem in problems:
            say("Configuration problem: " + problem)
        say("Fix .env (see .env.example) and start again.")
        return 1
    if allow_model_downloads():
        say("The retrieval models are not in the Hugging Face cache yet; they will be downloaded on the first run "
            "(about 180 MB).")

    try:
        info = config.describe() if config.remote else config.client().describe()
    except Exception as exc:                               # a model check must never stop the launcher itself
        info = {"error": f"{type(exc).__name__}: {exc}"}
    if config.remote:
        ok = info.get("service_reachable") and info.get("model_available")
        say(f"Model: {config.model} on {config.provider} service {config.endpoint}: "
            + ("ready." if ok else f"NOT READY ({info.get('error') or 'not offered'}). Research will stop at its model check."))
    else:
        if not info.get("ollama_version"):
            say("Ollama is not reachable at 127.0.0.1:11434. Start the Ollama app to run research; saved reports "
                "still work.")
        elif not info.get("model_digest"):
            say(f"Ollama {info['ollama_version']} is running but {config.model} is not installed: "
                f"ollama pull {config.model}")
        else:
            say(f"Model: {config.model} on local Ollama {info['ollama_version']}: ready.")

    reuse_studio = False
    if not port_free(args.studio_port):
        if studio_status(args.studio_port):
            reuse_studio = True
            say(f"A Research Studio is already running on port {args.studio_port}; the dashboard will use it.")
        else:
            say(f"Port {args.studio_port} is used by another program. Close it or start with --studio-port <port>.")
            return 1
    if not port_free(args.dashboard_port):
        say(f"Port {args.dashboard_port} is in use. Close the program using it or start with --dashboard-port <port>.")
        return 1
    if args.check:
        say("Checks passed. Nothing was started (--check).")
        return 0

    env = dict(os.environ, WEBRAG_STUDIO_URL=f"http://127.0.0.1:{args.studio_port}", PYTHONUNBUFFERED="1")
    processes: List[subprocess.Popen] = []
    try:
        if not reuse_studio:
            say(f"Starting the Research Studio backend on http://127.0.0.1:{args.studio_port} ...")
            studio = subprocess.Popen([sys.executable, "-m", "webrag.server", "--port", str(args.studio_port)],
                                      cwd=ROOT, env=env)
            processes.append(studio)
            if not wait_for(lambda: studio_status(args.studio_port) is not None, 60, studio):
                say("The Studio backend did not start; see the messages above.")
                return 1
        say(f"Starting the dashboard on http://127.0.0.1:{args.dashboard_port} ...")
        dashboard = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
                                      "--server.address", "127.0.0.1", "--server.port", str(args.dashboard_port),
                                      "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                                     cwd=ROOT, env=env)
        processes.append(dashboard)
        if not wait_for(lambda: dashboard_up(args.dashboard_port), 60, dashboard):
            say("The dashboard did not start; see the messages above.")
            return 1
        url = f"http://127.0.0.1:{args.dashboard_port}"
        say(f"Dashboard:        {url}")
        say(f"Research Studio:  http://127.0.0.1:{args.studio_port}")
        say("Press Ctrl+C in this window to stop both.")
        if not args.no_browser:
            webbrowser.open(url)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        say("A service stopped; stopping the other one.")
        return 1
    except KeyboardInterrupt:
        say("Stopping ...")
        return 0
    finally:
        stop(processes)


if __name__ == "__main__":
    if hasattr(signal, "SIGBREAK"):                        # Ctrl+Break in a Windows console stops cleanly too
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    raise SystemExit(main())

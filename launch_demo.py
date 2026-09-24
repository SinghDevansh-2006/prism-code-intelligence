"""Portable local launcher; uses the same interpreter for server and checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get('PRISM_PORT', '8000'))
BASE = f'http://127.0.0.1:{PORT}'

def health():
    try:
        with urllib.request.urlopen(BASE + '/health', timeout=2) as response:
            data = json.load(response)
        if data.get('service') != 'PRISM Agentic Code Intelligence':
            raise RuntimeError(f'Port {PORT} is occupied by another service')
        return data
    except OSError:
        return None

def main():
    state = health()
    process = None
    if state is None:
        log_path = Path(tempfile.gettempdir()) / f'prism_api_{PORT}.log'
        with log_path.open('a') as log:
            process = subprocess.Popen(
                [sys.executable, '-m', 'uvicorn', 'api:app', '--host', '127.0.0.1', '--port', str(PORT)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError(f'Backend exited; see {log_path}')
            state = health()
            if state:
                break
            time.sleep(1)
        if state is None:
            process.terminate()
            raise RuntimeError(f'Backend startup timed out; see {log_path}')
    expected = os.environ.get('PRISM_DEVICE', 'cpu').lower()
    if state.get('device') != expected:
        raise RuntimeError(f'Existing server reports device {state.get("device", "unknown")}; stop it and relaunch for {expected}')
    if not state['models_loaded']['semantic']:
        print('Warming semantic models; first use may download models.', flush=True)
        request = urllib.request.Request(BASE + '/search', data=json.dumps({
            'query': 'Find code that validates a tic-tac-toe board state', 'top_k': 1,
        }).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        if not result.get('results'):
            raise RuntimeError('Semantic warmup returned no results')
    print(f'Demo ready on {expected}: {BASE}/', flush=True)
    if os.environ.get('PRISM_OPEN_BROWSER', '1') != '0':
        webbrowser.open(BASE + '/')

if __name__ == '__main__':
    main()

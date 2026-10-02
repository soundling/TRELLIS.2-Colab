"""Stream subprocess output through notebook Python rather than inherited OS stdout."""
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path


def ensure_cuda_toolkit(env, toolkit_root=Path('/usr/local/cuda-12.6')):
    """Select CUDA 12.6 for the cu126 wheels, installing the toolkit if needed."""
    def compiler_version(candidate):
        if not candidate or not Path(candidate).is_file():
            return None
        try:
            output = subprocess.check_output([str(candidate), '--version'],
                                             env=env, text=True, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError):
            return None
        match = re.search(r'release (\d+\.\d+)(?:,|\s)', output)
        return match.group(1) if match else None

    pinned_nvcc = toolkit_root / 'bin/nvcc'
    candidates = [pinned_nvcc]
    if env.get('CUDA_HOME'):
        candidates.append(Path(env['CUDA_HOME']) / 'bin/nvcc')
    candidates.append(shutil.which('nvcc', path=env.get('PATH', '')))
    nvcc = next((path for path in candidates if compiler_version(path) == '12.6'), None)
    if nvcc is None:
        print('Installing CUDA 12.6 toolkit for PyTorch cu126 (toolkit only).', flush=True)
        try:
            run(['apt-get', 'install', '-y', '--no-install-recommends',
                 'cuda-toolkit-12-6'], env=env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('Could not install cuda-toolkit-12-6 from the Colab APT repositories. '
                               'See install.log for the package-manager error.') from exc
        nvcc = pinned_nvcc
        if compiler_version(nvcc) != '12.6':
            raise RuntimeError(f'CUDA installation did not provide a working 12.6 compiler at {nvcc}. '
                               'See install.log before continuing.')

    nvcc = Path(nvcc).resolve()
    cuda_home = nvcc.parent.parent
    env.update(CUDA_HOME=str(cuda_home), CUDA_PATH=str(cuda_home),
               CUDACXX=str(nvcc), CUDAToolkit_ROOT=str(cuda_home))
    for variable, prefix in [('PATH', str(cuda_home / 'bin')),
                             ('LD_LIBRARY_PATH', str(cuda_home / 'lib64'))]:
        entries = [entry for entry in env.get(variable, '').split(os.pathsep)
                   if entry and entry != prefix]
        env[variable] = os.pathsep.join([prefix, *entries])
    print(f'Using CUDA 12.6 compiler: {nvcc}', flush=True)
    return cuda_home


def run(args, cwd=None, env=None, heartbeat_seconds=20):
    if env is None:
        env = ENV
    args = [str(a) for a in args]
    private = [env.get(k) for k in ('HF_TOKEN', 'TRELLIS_API_KEY', 'NGROK_AUTHTOKEN') if env.get(k)]

    def clean(value):
        value = str(value)
        for token in private:
            value = value.replace(token, '[REDACTED]')
        return value

    label = clean(shlex.join(args))
    started = time.monotonic()
    log_path = ROOT / 'install.log'
    status_path = ROOT / 'install-status.json'
    messages = queue.Queue()
    process = None
    with log_path.open('a', encoding='utf-8', buffering=1) as log:
        def emit(text):
            text = clean(text)
            print(text, flush=True)
            log.write(text+'\n')

        def status(state, **extra):
            status_path.write_text(json.dumps({
                'command': label, 'state': state,
                'pid': process.pid if process else None,
                'elapsed_seconds': round(time.monotonic()-started, 1),
                'updated_at': time.time(), **extra,
            }, indent=2))

        def stop_owned_process():
            if process is None:
                return
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            if os.name == 'posix':
                # The direct child may exit before a compiler grandchild does.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        emit('\n>>> '+label)
        try:
            process = subprocess.Popen(args, cwd=cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1,
                start_new_session=(os.name == 'posix'))
            status('running')

            def reader():
                try:
                    for line in process.stdout:
                        messages.put(line.rstrip('\r\n'))
                finally:
                    messages.put(None)

            threading.Thread(target=reader, daemon=True).start()
            next_heartbeat = time.monotonic()+heartbeat_seconds
            while True:
                try:
                    line = messages.get(timeout=max(.01, next_heartbeat-time.monotonic()))
                except queue.Empty:
                    line = ''
                if line is None:
                    break
                if line:
                    emit(line)
                if time.monotonic() >= next_heartbeat:
                    status('running')
                    emit(f'[still running: {time.monotonic()-started:.0f}s; PID {process.pid}]')
                    next_heartbeat = time.monotonic()+heartbeat_seconds
            returncode = process.wait()
            status('succeeded' if returncode == 0 else 'failed', returncode=returncode)
            if returncode:
                emit(f'FAILED (exit {returncode}). Full log: {log_path}')
                raise subprocess.CalledProcessError(returncode, args)
            emit(f'Completed in {time.monotonic()-started:.1f}s')
        except KeyboardInterrupt:
            stop_owned_process()
            status('interrupted')
            emit('Interrupted; stopped this command and its compiler subprocesses. Cached downloads remain.')
            raise
        except BaseException:
            if process is not None and process.poll() is None:
                stop_owned_process()
            status('failed', returncode=process.poll() if process else None)
            raise
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()

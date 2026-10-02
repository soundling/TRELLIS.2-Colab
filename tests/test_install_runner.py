import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import install_runner


class InstallerTests(unittest.TestCase):
    def test_streams_both_outputs_heartbeat_and_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            install_runner.ROOT = Path(directory)
            install_runner.ENV = {**os.environ, 'HF_TOKEN': 'synthetic-secret-for-test'}
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                install_runner.run([sys.executable, '-u', '-c',
                    "import os,sys,time; print('stdout marker'); print('stderr marker',file=sys.stderr); "
                    "print(os.environ['HF_TOKEN']); time.sleep(.3)"], heartbeat_seconds=.05)
            output = buffer.getvalue()
            self.assertIn('stdout marker', output)
            self.assertIn('stderr marker', output)
            self.assertIn('still running', output)
            self.assertIn('[REDACTED]', output)
            self.assertNotIn('synthetic-secret-for-test', output)
            log = (Path(directory)/'install.log').read_text()
            self.assertNotIn('synthetic-secret-for-test', log)
            state = json.loads((Path(directory)/'install-status.json').read_text())
            self.assertEqual(state['state'], 'succeeded')

    def test_reports_failure_and_preserves_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            install_runner.ROOT = Path(directory)
            install_runner.ENV = os.environ.copy()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                install_runner.run([sys.executable, '-c', 'raise SystemExit(7)'])
            state = json.loads((Path(directory)/'install-status.json').read_text())
            self.assertEqual(state['state'], 'failed')
            self.assertEqual(state['returncode'], 7)


class CudaToolkitTests(unittest.TestCase):
    def test_reuses_pinned_toolkit_even_when_default_is_cuda_13(self):
        with tempfile.TemporaryDirectory() as directory:
            toolkit = Path(directory).resolve()
            nvcc = toolkit / 'bin/nvcc'
            nvcc.parent.mkdir()
            nvcc.touch()
            env = {'PATH': '/default/cuda-13/bin', 'CUDA_HOME': '/default/cuda-13',
                   'LD_LIBRARY_PATH': '/driver/lib64'}
            with mock.patch.object(install_runner.shutil, 'which', return_value=None), \
                 mock.patch.object(install_runner.subprocess, 'check_output',
                                   return_value='Cuda compilation tools, release 12.6, V12.6.131'), \
                 mock.patch.object(install_runner, 'run') as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                install_runner.ensure_cuda_toolkit(env, toolkit)
                first_env = env.copy()
                install_runner.ensure_cuda_toolkit(env, toolkit)
            run.assert_not_called()
            self.assertEqual(env, first_env)  # Rerunning must not accumulate path entries.
            self.assertEqual(env['CUDA_HOME'], str(toolkit))
            self.assertEqual(env['CUDA_PATH'], str(toolkit))
            self.assertEqual(env['CUDACXX'], str(nvcc))
            self.assertEqual(env['CUDAToolkit_ROOT'], str(toolkit))
            self.assertEqual(env['PATH'].split(os.pathsep)[0], str(nvcc.parent))
            self.assertEqual(env['LD_LIBRARY_PATH'].split(os.pathsep),
                             [str(toolkit / 'lib64'), '/driver/lib64'])

    def test_installs_matching_toolkit_when_default_is_incompatible_or_missing(self):
        for version in ('13.0', '12.8', '12.4', None):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                toolkit = root / 'cuda-12.6'
                default_nvcc = root / 'default/bin/nvcc'
                default_nvcc.parent.mkdir(parents=True)
                if version:
                    default_nvcc.touch()
                pinned_nvcc = toolkit / 'bin/nvcc'
                env = {'PATH': str(default_nvcc.parent)}

                def install(args, **kwargs):
                    self.assertEqual(args, ['apt-get', 'install', '-y', '--no-install-recommends',
                                            'cuda-toolkit-12-6'])
                    self.assertIs(kwargs['env'], env)
                    pinned_nvcc.parent.mkdir(parents=True)
                    pinned_nvcc.touch()

                def compiler(args, **kwargs):
                    release = '12.6' if Path(args[0]) == pinned_nvcc else version
                    return f'Cuda compilation tools, release {release}, V{release}.131'

                with mock.patch.object(install_runner.shutil, 'which',
                                       return_value=str(default_nvcc) if version else None), \
                     mock.patch.object(install_runner.subprocess, 'check_output', side_effect=compiler), \
                     mock.patch.object(install_runner, 'run', side_effect=install) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(install_runner.ensure_cuda_toolkit(env, toolkit), toolkit)
                run.assert_called_once()
                self.assertEqual(env['CUDACXX'], str(pinned_nvcc))

    def test_failed_or_incomplete_install_does_not_select_incompatible_compiler(self):
        for fails in (True, False):
            with self.subTest(apt_fails=fails), tempfile.TemporaryDirectory() as directory:
                env = {'PATH': '/default/cuda-13/bin', 'CUDA_HOME': '/default/cuda-13'}
                original = env.copy()
                error = subprocess.CalledProcessError(100, ['apt-get']) if fails else None
                with mock.patch.object(install_runner.shutil, 'which', return_value=None), \
                     mock.patch.object(install_runner, 'run', side_effect=error), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     self.assertRaisesRegex(RuntimeError, 'install.log'):
                    install_runner.ensure_cuda_toolkit(env, Path(directory))
                self.assertEqual(env, original)


if __name__ == '__main__':
    unittest.main(verbosity=2)

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
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


if __name__ == '__main__':
    unittest.main(verbosity=2)

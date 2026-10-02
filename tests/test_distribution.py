"""Validate the shareable notebook without installing CUDA or opening a tunnel."""
import ast
import json
from pathlib import Path
import re
import unittest
import nbformat
import install_runner
from tools.sync_notebook import extract

ROOT = Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def test_schema_clean_metadata_and_sources(self):
        path = ROOT / 'trellis2_l4_ngrok_api.ipynb'
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        self.assertEqual(len(notebook.cells), 25)
        for cell in notebook.cells:
            self.assertNotIn('outputId', cell.metadata)
            self.assertNotIn('colab', cell.metadata)
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
        text = path.read_text(encoding='utf-8')
        self.assertIsNone(re.search(r'https://[\w-]+\.ngrok(?:-free)?\.(?:app|io)', text))
        self.assertIsNone(re.search(r'\bhf_[A-Za-z0-9]{20,}\b', text))

    def test_extracted_sources_match_notebook(self):
        notebook = json.loads((ROOT / 'trellis2_l4_ngrok_api.ipynb').read_text(encoding='utf-8'))
        for name, source in extract(notebook).items():
            self.assertEqual((ROOT / name).read_text(encoding='utf-8'), source, name)

    def test_torch_wheels_and_constraints_match_selected_toolkit(self):
        notebook = json.loads((ROOT / 'trellis2_l4_ngrok_api.ipynb').read_text(encoding='utf-8'))
        installer = ast.parse(''.join(notebook['cells'][4]['source']))
        version = install_runner.ensure_cuda_toolkit.__defaults__[0].name.removeprefix('cuda-')
        wheel_tag = 'cu' + version.replace('.', '')
        calls = [node.value for node in installer.body if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call)]
        torch_install = next(call for call in calls if isinstance(call.func, ast.Name)
                             and call.func.id == 'pip' and call.args
                             and isinstance(call.args[0], ast.Constant)
                             and call.args[0].value.startswith('torch=='))
        args = [ast.literal_eval(arg) for arg in torch_install.args]
        for package in args[:2]:
            self.assertTrue(package.endswith('+' + wheel_tag), package)
        self.assertEqual(args[-1], 'https://download.pytorch.org/whl/' + wheel_tag)
        constraints = next(ast.literal_eval(call.args[0]) for call in calls
                           if isinstance(call.func, ast.Attribute)
                           and isinstance(call.func.value, ast.Name)
                           and call.func.value.id == 'constraints'
                           and call.func.attr == 'write_text')
        for package in args[:2]:
            self.assertIn(package, constraints.splitlines())

    def test_credentials_are_requested_not_embedded(self):
        notebook = json.loads((ROOT / 'trellis2_l4_ngrok_api.ipynb').read_text(encoding='utf-8'))
        config = ast.parse(''.join(notebook['cells'][2]['source']))
        assignments = {target.id: node.value for node in config.body if isinstance(node, ast.Assign)
                       for target in node.targets if isinstance(target, ast.Name)}
        for variable, secret in [('NGROK_AUTHTOKEN', 'NGROK_AUTHTOKEN'), ('HF_TOKEN', 'HF_TOKEN'), ('API_KEY', 'TRELLIS_API_KEY')]:
            call = assignments[variable]
            self.assertIsInstance(call, ast.Call)
            self.assertEqual(call.func.id, 'secret')
            self.assertEqual(ast.literal_eval(call.args[0]), secret)

"""Validate the shareable notebook without installing CUDA or opening a tunnel."""
import ast
import json
from pathlib import Path
import re
import unittest
import nbformat
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

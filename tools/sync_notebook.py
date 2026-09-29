"""Extract reviewable Python from the canonical notebook, without executing it."""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'trellis2_l4_ngrok_api.ipynb'


def extract(notebook):
    sources = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        tree = ast.parse(source)
        if source.startswith('# Embedded '):
            name = source.splitlines()[0].removeprefix('# Embedded ').strip()
            if name not in ('backend.py', 'api_server.py') or name in sources:
                raise ValueError('Unexpected or duplicate embedded module: ' + name)
            call = tree.body[0].value
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != 'write_text':
                raise ValueError('Expected a literal embedded write_text call')
            sources[name] = ast.literal_eval(call.args[0])
        elif any(isinstance(node, ast.FunctionDef) and node.name == 'run' for node in tree.body):
            run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
            sources['install_runner.py'] = '\n'.join(source.splitlines()[:run.end_lineno]) + '\n'
        elif any(isinstance(node, ast.FunctionDef) and node.name == 'download_job' for node in tree.body):
            end = next(node.end_lineno for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'download_job')
            sources['client.py'] = '\n'.join(source.splitlines()[:end]) + '\n'
    if set(sources) != {'backend.py', 'api_server.py', 'install_runner.py', 'client.py'}:
        raise ValueError('Notebook must contain the backend, API, installer runner and client')
    for name, source in sources.items():
        ast.parse(source, filename=name)
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail if extracted modules differ; write nothing')
    args = parser.parse_args()
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    for name, source in extract(notebook).items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text(encoding='utf-8') != source:
                raise SystemExit('Out of sync: ' + name + '. Run python tools/sync_notebook.py')
        else:
            path.write_text(source, encoding='utf-8', newline='\n')
    print('Notebook and four extracted modules are in sync.')


if __name__ == '__main__':
    main()

"""Local contract tests; no model downloads, GPU use or public tunnel."""
import ast
import io
import json
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from api_server import Service, create_app, glb_info, MAX_BODY

KEY = 'test-only-' + 'x'*40
HEADERS = {'Authorization': 'Bearer '+KEY}


def reference():
    result = io.BytesIO()
    Image.new('RGB', (64, 64), 'green').save(result, 'PNG')
    return result.getvalue()


def make_glb(path, extensions=None):
    obj = {'asset': {'version': '2.0'}, 'meshes': [{'primitives': [{'indices': 0}]}],
           'accessors': [{'count': 3}], 'materials': [{}]}
    if extensions:
        obj['extensionsRequired'] = extensions
    raw = json.dumps(obj).encode()
    raw += b' '*((-len(raw)) % 4)
    path.write_bytes(struct.pack('<4sIIII', b'glTF', 2, 20+len(raw), len(raw), 0x4E4F534A)+raw)


class FakeBackend:
    def __init__(self, gate=None, fail=False):
        self.gate, self.fail = gate, fail
        self.active, self.peak = 0, 0

    def generate(self, source, output, settings, stage):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.gate:
                self.gate.wait(10)
            stage('fake_generation')
            if self.fail:
                raise RuntimeError('CUDA out of memory')
            make_glb(output)
            return {'simulated': True}
        finally:
            self.active -= 1


def wait_until(predicate):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError('Timed out')


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def post(self, client, **settings):
        return client.post('/jobs', headers=HEADERS,
                           files={'image': ('input.png', reference(), 'image/png')}, data=settings)

    def test_auth_job_download_settings_and_delete(self):
        backend = FakeBackend()
        service = Service(self.root, lambda: backend)
        with TestClient(create_app(service, KEY)) as client:
            wait_until(lambda: service.ready)
            self.assertEqual(client.get('/health').status_code, 401)
            self.assertEqual(client.get('/openapi.json').status_code, 401)
            self.assertEqual(client.post('/jobs', content=b'bad').status_code, 401)
            self.assertEqual(client.get('/health', headers={'Authorization':'Bearer wrong'}).status_code,401)
            self.assertEqual(client.get('/openapi.json', headers=HEADERS).status_code,200)
            self.assertEqual(self.post(client, pipeline='1024_cascade').status_code,422)
            self.assertEqual(self.post(client, texture_size=123).status_code,422)
            self.assertEqual(self.post(client, seed=-1).status_code,422)
            self.assertEqual(client.post('/jobs',headers=HEADERS,files={'image':('bad.png',b'bad')}).status_code,422)
            created = self.post(client, seed=14, remove_background='false')
            self.assertEqual(created.status_code,202,created.text)
            ident = created.json()['id']
            wait_until(lambda: service.get(ident)['status']=='succeeded')
            job = client.get('/jobs/'+ident, headers=HEADERS).json()
            self.assertEqual(job['settings']['seed'],14)
            self.assertFalse(job['settings']['remove_background'])
            self.assertEqual(job['result']['triangles'],1)
            self.assertEqual(client.get('/jobs/'+ident+'/model.glb',headers=HEADERS).content[:4],b'glTF')
            self.assertEqual(client.get('/jobs/'+ident+'/manifest.json',headers=HEADERS).status_code,200)
            self.assertEqual(client.delete('/jobs/'+ident,headers=HEADERS).status_code,200)
            self.assertFalse((self.root/ident).exists())
            self.assertEqual(client.get('/jobs/'+ident,headers=HEADERS).status_code,404)
            self.assertEqual(client.get('/jobs/not-a-uuid',headers=HEADERS).status_code,422)
        service.thread.join(2)

    def test_queue_bound_single_worker_and_active_delete(self):
        gate = threading.Event()
        backend = FakeBackend(gate)
        service = Service(self.root, lambda: backend)
        try:
            with TestClient(create_app(service, KEY)) as client:
                wait_until(lambda: service.ready)
                ids = [self.post(client).json()['id'] for _ in range(4)]
                self.assertEqual(self.post(client).status_code,429)
                self.assertEqual(client.delete('/jobs/'+ids[0],headers=HEADERS).status_code,409)
                self.assertEqual(client.get('/jobs/'+ids[0]+'/model.glb',headers=HEADERS).status_code,409)
                gate.set()
                wait_until(lambda: all(service.get(x)['status']=='succeeded' for x in ids))
                self.assertEqual(backend.peak,1)
                self.assertEqual(self.post(client).status_code,202)
                wait_until(lambda: service.pending.unfinished_tasks==0)
        finally:
            gate.set()
            service.stop.set()
            service.thread.join(2)

    def test_body_limit_and_failed_job_releases_slot(self):
        service = Service(self.root, lambda: FakeBackend(fail=True), max_pending=1)
        with TestClient(create_app(service, KEY)) as client:
            wait_until(lambda: service.ready)
            self.assertEqual(client.post('/jobs',headers=HEADERS,content=b'a'*(MAX_BODY+1)).status_code,413)
            ident = self.post(client).json()['id']
            wait_until(lambda: service.get(ident)['status']=='failed')
            self.assertIn('memory',service.get(ident)['error'])
            self.assertEqual(self.post(client).status_code,202)
            wait_until(lambda: service.pending.unfinished_tasks==0)
        service.thread.join(2)

    def test_restart_marks_interrupted_jobs_and_keeps_results(self):
        import uuid
        service = Service(self.root, lambda: FakeBackend())
        pending_id, done_id = str(uuid.uuid4()), str(uuid.uuid4())
        service.save({'id':pending_id,'status':'running'})
        service.save({'id':done_id,'status':'succeeded'})
        make_glb(self.root/done_id/'model.glb')
        restored = Service(self.root, lambda: FakeBackend())
        self.assertEqual(restored.get(pending_id)['status'],'failed')
        self.assertEqual(restored.get(done_id)['status'],'succeeded')

    def test_glb_validation(self):
        path = self.root/'model.glb'
        path.write_bytes(b'bad')
        with self.assertRaises(ValueError): glb_info(path)
        make_glb(path,['EXT_texture_webp'])
        with self.assertRaises(ValueError): glb_info(path)
        make_glb(path)
        self.assertEqual(glb_info(path)['triangles'],1)

    def test_notebook_clean_and_every_python_cell_compiles(self):
        notebook = json.loads((Path(__file__).resolve().parents[1]/'trellis2_l4_ngrok_api.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(notebook['nbformat'],4)
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                source = ''.join(cell['source'])
                ast.parse(source)
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'],[])
        self.assertNotIn(KEY,json.dumps(notebook))


if __name__ == '__main__':
    unittest.main(verbosity=2)

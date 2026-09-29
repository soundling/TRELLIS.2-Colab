"""Authenticated, bounded, single-worker API. Embedded in the notebook."""
import asyncio
import hashlib
import io
import json
import logging
import os
import queue
import secrets
import shutil
import struct
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageOps, UnidentifiedImageError

LOG = logging.getLogger('trellis-api')
MAX_BODY = 16 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


def glb_info(path):
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError('GLB export is truncated.')
    magic, version, length, chunk_length, chunk_type = struct.unpack('<4sIIII', data[:20])
    if magic != b'glTF' or version != 2 or length != len(data) or chunk_type != 0x4E4F534A:
        raise ValueError('Invalid GLB export.')
    document = json.loads(data[20:20+chunk_length])
    forbidden = {'EXT_texture_webp', 'EXT_meshopt_compression', 'KHR_draco_mesh_compression', 'KHR_texture_basisu'}
    if forbidden.intersection(document.get('extensionsRequired', [])):
        raise ValueError('Export requires unsupported compressed textures or geometry.')
    if any('uri' in b for b in document.get('buffers', [])):
        raise ValueError('GLB has an external buffer.')
    if any('uri' in i for i in document.get('images', [])):
        raise ValueError('GLB has external images.')
    triangles = 0
    for mesh in document.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            if primitive.get('mode', 4) == 4:
                accessor = primitive.get('indices', primitive.get('attributes', {}).get('POSITION'))
                if accessor is not None:
                    triangles += document['accessors'][accessor]['count'] // 3
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'triangles': triangles, 'materials': len(document.get('materials', [])),
            'images': len(document.get('images', []))}


class RequestGate:
    """Check authentication and bound actual body bytes before multipart parsing."""
    def __init__(self, app, key):
        self.app, self.key = app, key

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        supplied = headers.get(b'authorization', b'')
        if not secrets.compare_digest(supplied, b'Bearer ' + self.key.encode()):
            return await JSONResponse({'detail': 'Bearer token required'}, 401,
                                      headers={'WWW-Authenticate': 'Bearer'})(scope, receive, send)
        if scope['method'] in ('POST', 'PUT', 'PATCH'):
            body = bytearray()
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                body.extend(message.get('body', b''))
                if len(body) > MAX_BODY:
                    return await JSONResponse({'detail': 'Request exceeds 16 MiB'}, 413)(scope, receive, send)
                if not message.get('more_body', False):
                    break
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
                return await receive()
            return await self.app(scope, replay, send)
        return await self.app(scope, receive, send)


class Service:
    def __init__(self, root, factory, allow_1024=False, max_pending=4):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.factory = factory
        self.allow_1024 = allow_1024
        self.jobs = {}
        self.lock = threading.RLock()
        self.pending = queue.Queue()
        self.slots = threading.BoundedSemaphore(max_pending)
        self.ready = False
        self.error = None
        self.stop = threading.Event()
        self.backend = None
        self.thread = None
        self.restore()

    def restore(self):
        for path in self.root.glob('*/job.json'):
            try:
                job = json.loads(path.read_text())
                if str(uuid.UUID(job['id'])) != path.parent.name:
                    continue
                if job['status'] in ('queued', 'running'):
                    job.update(status='failed', stage='interrupted', error='Runtime restarted; resubmit this job.')
                if job['status'] == 'succeeded' and not (path.parent/'model.glb').exists():
                    job.update(status='failed', error='Result file is missing.')
                self.jobs[job['id']] = job
                self.save(job)
            except (ValueError, KeyError, OSError):
                LOG.warning('Skipped an unreadable job manifest: %s', path.name)

    def save(self, job):
        directory = self.root / job['id']
        directory.mkdir(exist_ok=True)
        temporary = directory / 'job.json.tmp'
        temporary.write_text(json.dumps(job, indent=2))
        temporary.replace(directory / 'job.json')

    def start(self):
        self.thread = threading.Thread(target=self.work, daemon=True, name='trellis-gpu')
        self.thread.start()

    def update(self, ident, **values):
        with self.lock:
            self.jobs[ident].update(values, updated_at=time.time())
            self.save(self.jobs[ident])

    def work(self):
        try:
            self.backend = self.factory()
            self.ready = True
        except Exception:
            LOG.exception('Model initialization failed')
            self.error = 'Model initialization failed; inspect the notebook server log.'
            return
        while not self.stop.is_set():
            try:
                ident = self.pending.get(timeout=0.5)
            except queue.Empty:
                continue
            started = time.time()
            folder = self.root / ident
            try:
                self.update(ident, status='running', stage='starting', started_at=started)
                settings = self.jobs[ident]['settings']
                provenance = self.backend.generate(
                    folder/'input.png', folder/'model.glb', settings,
                    lambda stage: self.update(ident, stage=stage),
                )
                result = glb_info(folder/'model.glb')
                self.update(ident, status='succeeded', stage='complete', result=result,
                            provenance=provenance, seconds=round(time.time()-started, 2),
                            completed_at=time.time())
            except Exception as exc:
                LOG.exception('Job %s failed', ident)
                oom = 'out of memory' in str(exc).lower()
                self.update(ident, status='failed', stage='failed', completed_at=time.time(),
                            error=('GPU memory exhausted. Retry at 512 with 1024 textures and remesh=false.' if oom
                                   else 'Generation failed; inspect the notebook log for this job ID.'))
            finally:
                self.slots.release()
                self.pending.task_done()

    def submit(self, image_bytes, settings):
        if not self.ready:
            raise HTTPException(503, self.error or 'Models are still loading')
        if settings['pipeline'] != '512' and not self.allow_1024:
            raise HTTPException(422, '1024 is disabled; enable ALLOW_1024 and restart the server.')
        if not self.slots.acquire(blocking=False):
            raise HTTPException(429, 'Queue full; retry after a job completes', headers={'Retry-After': '30'})
        ident = str(uuid.uuid4())
        folder = self.root / ident
        accepted = False
        try:
            with self.lock:
                if len(self.jobs) >= 64:
                    raise HTTPException(507, '64 retained jobs reached. Download and delete completed jobs.')
                if shutil.disk_usage(self.root).free < 2 * 2**30:
                    raise HTTPException(507, 'Less than 2 GiB free disk space.')
                with Image.open(io.BytesIO(image_bytes)) as source:
                    if source.format not in ('PNG', 'JPEG', 'WEBP'):
                        raise HTTPException(415, 'Use PNG, JPEG or WebP')
                    if min(source.size) < 32 or source.width*source.height > Image.MAX_IMAGE_PIXELS:
                        raise HTTPException(422, 'Image must be at least 32x32 and at most 20 megapixels')
                    source.load()
                    image = ImageOps.exif_transpose(source).convert('RGBA')
                    if image.getchannel('A').getextrema() == (0, 0):
                        raise HTTPException(422, 'Image is fully transparent')
                    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                    folder.mkdir()
                    image.save(folder/'input.png')
                job = {'id': ident, 'status': 'queued', 'stage': 'queued', 'settings': settings,
                       'created_at': time.time(), 'updated_at': time.time(),
                       'input_sha256': hashlib.sha256(image_bytes).hexdigest()}
                self.jobs[ident] = job
                self.save(job)
                self.pending.put(ident)
                accepted = True
                return dict(job)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
            raise HTTPException(422, 'Invalid or oversized image') from exc
        finally:
            if not accepted:
                self.slots.release()
                with self.lock:
                    self.jobs.pop(ident, None)
                if folder.exists():
                    shutil.rmtree(folder)

    def get(self, ident):
        with self.lock:
            if ident not in self.jobs:
                raise HTTPException(404, 'Unknown job')
            return dict(self.jobs[ident])


def create_app(service, api_key):
    if len(api_key) < 32:
        raise ValueError('TRELLIS_API_KEY must contain at least 32 characters.')
    bearer = HTTPBearer()

    def auth(credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer)]):
        if not secrets.compare_digest(credentials.credentials, api_key):
            raise HTTPException(401, 'Invalid token')

    @asynccontextmanager
    async def lifespan(app):
        service.start()
        yield
        service.stop.set()
        # A running CUDA job cannot safely be killed by a Python thread.
        # Colab's stop cell terminates the owned server process when necessary.

    app = FastAPI(title='TRELLIS.2 personal asset API', version='1.0.0',
                  docs_url=None, redoc_url=None, openapi_url=None,
                  dependencies=[Depends(auth)], lifespan=lifespan)
    app.add_middleware(RequestGate, key=api_key)

    @app.get('/health')
    def health():
        with service.lock:
            return {'ready': service.ready, 'error': service.error,
                    'queued': sum(j['status']=='queued' for j in service.jobs.values()),
                    'running': sum(j['status']=='running' for j in service.jobs.values()),
                    'pipelines': ['512', '1024_cascade'] if service.allow_1024 else ['512']}

    @app.get('/openapi.json', include_in_schema=False)
    def schema():
        return app.openapi()

    @app.get('/docs', include_in_schema=False)
    def docs():
        # Best used by an authenticated API client; a plain address-bar visit gets 401.
        return get_swagger_ui_html(openapi_url='/openapi.json', title='TRELLIS API')

    @app.post('/jobs', status_code=202)
    async def submit(
        image: Annotated[UploadFile, File()],
        seed: Annotated[int, Form(ge=0, le=2147483647)] = 42,
        pipeline: Annotated[Literal['512', '1024_cascade'], Form()] = '512',
        steps: Annotated[int, Form(ge=4, le=30)] = 12,
        face_count: Annotated[int, Form(ge=1000, le=200000)] = 30000,
        texture_size: Annotated[int, Form()] = 2048,
        remove_background: Annotated[bool, Form()] = True,
        remesh: Annotated[bool, Form()] = True,
    ):
        if texture_size not in (1024, 2048, 4096):
            raise HTTPException(422, 'texture_size must be 1024, 2048 or 4096')
        try:
            data = await image.read(MAX_BODY+1)
        finally:
            await image.close()
        if len(data) > MAX_BODY:
            raise HTTPException(413, 'Image too large')
        settings = dict(seed=seed, pipeline=pipeline, steps=steps, face_count=face_count,
                        texture_size=texture_size, remove_background=remove_background, remesh=remesh)
        return await asyncio.to_thread(service.submit, data, settings)

    @app.get('/jobs')
    def list_jobs():
        with service.lock:
            return list(service.jobs.values())

    @app.get('/jobs/{ident}')
    def status(ident: uuid.UUID):
        return service.get(str(ident))

    @app.get('/jobs/{ident}/model.glb')
    def result(ident: uuid.UUID):
        job = service.get(str(ident))
        if job['status'] != 'succeeded':
            raise HTTPException(409, 'Job has not completed successfully')
        return FileResponse(service.root/str(ident)/'model.glb',
                            media_type='model/gltf-binary', filename=f'trellis-{ident}.glb')

    @app.get('/jobs/{ident}/manifest.json')
    def manifest(ident: uuid.UUID):
        return service.get(str(ident))

    @app.delete('/jobs/{ident}')
    def delete(ident: uuid.UUID):
        with service.lock:
            job = service.get(str(ident))
            if job['status'] in ('queued', 'running'):
                raise HTTPException(409, 'Cannot delete an active job')
            shutil.rmtree(service.root/str(ident))
            del service.jobs[str(ident)]
        return {'deleted': str(ident)}

    return app


if __name__ == '__main__':
    import uvicorn
    from backend import TrellisBackend
    logging.basicConfig(level=logging.INFO)
    service = Service(os.environ['JOBS_DIR'], TrellisBackend,
                      allow_1024=os.environ.get('ALLOW_1024')=='1')
    app = create_app(service, os.environ['TRELLIS_API_KEY'])
    uvicorn.run(app, host='127.0.0.1', port=8000, access_log=False)

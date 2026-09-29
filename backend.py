"""GPU implementation. Embedded verbatim in the delivered notebook."""
import gc
import json
import os
from pathlib import Path


class TrellisBackend:
    def __init__(self):
        import torch
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg

        self.torch = torch
        self.allow_1024 = os.environ.get('ALLOW_1024', '0') == '1'
        self.bg_session = None
        root = Path(os.environ['SERVICE_ROOT'])
        model_dir = root / 'resolved-model'
        model_dir.mkdir(exist_ok=True)
        api = HfApi()
        revisions = {}

        def revision(repo):
            if repo not in revisions:
                revisions[repo] = api.model_info(repo).sha
            return revisions[repo]

        def download(repo, filename):
            return hf_hub_download(repo, filename, revision=revision(repo))

        repo = 'microsoft/TRELLIS.2-4B'
        config = json.loads(Path(download(repo, 'pipeline.json')).read_text())
        args = config['args']
        # Resolve all weight references to local paths at a recorded HF revision.
        selected = {}
        for name, stem in args['models'].items():
            if not self.allow_1024 and name.endswith('_1024'):
                continue
            if stem.startswith('ckpts/'):
                source, relative = repo, stem
            else:
                owner, project, relative = stem.split('/', 2)
                source = owner + '/' + project
            json_path = download(source, relative + '.json')
            download(source, relative + '.safetensors')
            selected[name] = json_path[:-5]
        args['models'] = selected
        dino_repo = args['image_cond_model']['args']['model_name']
        dino_dir = snapshot_download(
            dino_repo, revision=revision(dino_repo),
            allow_patterns=['*.json', '*.safetensors', 'LICENSE*', 'README*'],
        )
        args['image_cond_model']['args']['model_name'] = dino_dir

        # Use our CPU foreground preparation, avoiding the default RMBG-2.0 dependency.
        # Pipeline.from_pretrained still constructs its configured rembg class.
        class PreparedForeground:
            def __init__(self):
                pass
        rembg.PreparedForeground = PreparedForeground
        args['rembg_model'] = {'name': 'PreparedForeground', 'args': {}}
        args['low_vram'] = True
        args['default_pipeline_type'] = '512'
        (model_dir / 'pipeline.json').write_text(json.dumps(config, indent=2))
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(model_dir))
        self.pipeline.low_vram = True
        self.pipeline.cuda()
        self.provenance = {
            'model_revisions': revisions,
            'code_revision': os.environ['TRELLIS_REV'],
            'gpu': torch.cuda.get_device_name(),
            'torch': torch.__version__,
            'low_vram': True,
            'allow_1024': self.allow_1024,
        }
        (root / 'model-provenance.json').write_text(json.dumps(self.provenance, indent=2))

    def prepare(self, path, remove_background):
        from PIL import Image, ImageOps
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGBA')
        image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        alpha = image.getchannel('A')
        if alpha.getextrema() == (255, 255) and remove_background:
            from rembg import new_session, remove
            if self.bg_session is None:
                self.bg_session = new_session('u2netp', providers=['CPUExecutionProvider'])
            image = remove(image.convert('RGB'), session=self.bg_session).convert('RGBA')
            alpha = image.getchannel('A')
        if alpha.getextrema() != (255, 255):
            bounds = alpha.point(lambda p: 255 if p > 204 else 0).getbbox()
            if bounds is None:
                raise ValueError('No opaque foreground found; upload a clearer reference.')
            image = image.crop(bounds)
        side = max(image.size)
        # Match upstream conditioning: centered foreground over black, square crop.
        canvas = Image.new('RGBA', (side, side), (0, 0, 0, 255))
        canvas.alpha_composite(image, ((side-image.width)//2, (side-image.height)//2))
        return canvas.convert('RGB').resize((1024, 1024), Image.Resampling.LANCZOS)

    def generate(self, source, output, settings, stage):
        import o_voxel
        torch = self.torch
        mesh = glb = image = None
        try:
            stage('preparing_image')
            image = self.prepare(source, settings['remove_background'])
            image.save(output.parent / 'prepared-reference.png')
            stage('generating_geometry_and_materials')
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                mesh = self.pipeline.run(
                    image, seed=settings['seed'], num_samples=1,
                    pipeline_type=settings['pipeline'], preprocess_image=False,
                    sparse_structure_sampler_params={'steps': settings['steps']},
                    shape_slat_sampler_params={'steps': settings['steps']},
                    tex_slat_sampler_params={'steps': settings['steps']},
                )[0]
                stage('baking_and_exporting_glb')
                mesh.simplify(16777216)
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices, faces=mesh.faces,
                    attr_volume=mesh.attrs, coords=mesh.coords,
                    attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=settings['face_count'],
                    texture_size=settings['texture_size'],
                    remesh=settings['remesh'], remesh_band=1, remesh_project=0,
                    verbose=True,
                )
                # Embedded PNG textures: no EXT_texture_webp requirement for Oni.
                glb.export(str(output), file_type='glb', extension_webp=False)
                torch.cuda.synchronize()
            return {**self.provenance,
                    'peak_gpu_allocated_gib': round(torch.cuda.max_memory_allocated()/2**30, 3)}
        finally:
            # An OOM may interrupt an upstream stage before it moves its model to CPU.
            mesh = glb = image = None
            for model in self.pipeline.models.values():
                model.cpu()
            self.pipeline.image_cond_model.cpu()
            gc.collect()
            torch.cuda.empty_cache()

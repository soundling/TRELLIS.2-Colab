import hashlib
from pathlib import Path
import requests, time, json

def submit_asset(base_url, api_key, reference_path, **settings):
    headers = {'Authorization': 'Bearer '+api_key, 'ngrok-skip-browser-warning':'1'}
    with open(reference_path, 'rb') as image:
        response = requests.post(base_url.rstrip('/')+'/jobs', headers=headers,
                                 files={'image': (Path(reference_path).name, image)},
                                 data=settings, timeout=(15,120))
    response.raise_for_status()
    job = response.json()
    print('Submitted job:', job['id'])
    return job['id']

def download_job(base_url, api_key, job_id, output_path='model.glb', timeout=1800):
    headers = {'Authorization':'Bearer '+api_key, 'ngrok-skip-browser-warning':'1'}
    endpoint = base_url.rstrip('/')+'/jobs/'+job_id
    deadline = time.monotonic()+timeout
    previous = None
    while time.monotonic() < deadline:
        response = requests.get(endpoint, headers=headers, timeout=30)
        response.raise_for_status()
        job = response.json()
        state = (job['status'], job['stage'])
        if state != previous:
            print(*state)
            previous = state
        if job['status'] == 'failed':
            raise RuntimeError(job.get('error', 'Generation failed'))
        if job['status'] == 'succeeded':
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            temp = output.with_suffix(output.suffix+'.part')
            checksum = hashlib.sha256()
            try:
                with requests.get(endpoint+'/model.glb', headers=headers, stream=True, timeout=(15,180)) as data:
                    data.raise_for_status()
                    with temp.open('wb') as f:
                        for chunk in data.iter_content(1024*1024):
                            if chunk:
                                checksum.update(chunk)
                                f.write(chunk)
                if checksum.hexdigest() != job['result']['sha256']:
                    raise RuntimeError('Download checksum mismatch; retry the download.')
                temp.replace(output)
            finally:
                temp.unlink(missing_ok=True)
            output.with_suffix('.manifest.json').write_text(json.dumps(job, indent=2))
            print('Saved:', output, job['result'])
            return output
        time.sleep(3)
    raise TimeoutError('Job still pending. Resume download_job with job ID '+job_id)

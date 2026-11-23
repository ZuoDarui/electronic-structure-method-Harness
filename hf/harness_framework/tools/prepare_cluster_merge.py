"""Create a credential-checked migration snapshot, keeping old evidence intact."""
from pathlib import Path
from datetime import datetime, timezone
import getpass
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import tempfile


def main():
    framework = Path(__file__).resolve().parents[1]
    root = framework.parent
    records = framework / 'outputs/reorganization_20261113'
    before = json.loads((records / 'remote_before.json').read_text())
    mapping = {name: 'tests/' + name if name.startswith('test_') else name for name in before['selected']}
    secrets = [getpass.getpass('Known SSH password for in-memory migration scan: ').encode()]
    assert secrets[0]
    for path in (Path.home() / '.codex/auth.json', Path.home() / 'secrets/tencentcloud_tione.json'):
        if path.is_file():
            secrets += [v.encode() for k, v in json.loads(path.read_text()).items()
                        if isinstance(v, str) and len(v) >= 8 and any(word in k.lower() for word in ('key', 'secret', 'token', 'password'))]
    files, payload = {}, {}
    for base, directories, names in os.walk(framework, followlinks=False):
        directories[:] = [name for name in directories if name not in {'__pycache__', '.git', '.pytest_cache'}]
        for name in names:
            path = Path(base) / name
            if name in {'.DS_Store', '.controller.lock'} or name.endswith(('.pyc', '.pyo')):
                continue
            if path.parent == records and name in {'bundle.json', 'upload_manifest.json'}:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError('Nonregular payload file: ' + str(path))
            data = path.read_bytes()
            key_block = re.search(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----\r?\n[A-Za-z0-9+/=\r\n]{32,}', data)
            if key_block or any(secret in data for secret in secrets):
                raise ValueError('Credential detected in project file: ' + str(path.relative_to(root)))
            relative = str(path.relative_to(framework))
            files[relative] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'mode': stat.S_IMODE(path.stat().st_mode) & 0o777}
            payload['framework/' + relative] = data
    readme = (root / 'README_GPU_HF_zh.md').read_bytes()
    if any(secret in readme for secret in secrets):
        raise ValueError('Credential in root README')
    payload['project/README_GPU_HF_zh.md'] = readme
    manifest = {'schema_version': 'harness-layout-upload/1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'mapping': mapping, 'files': files, 'remote_before': before,
                'root_readme_sha256': hashlib.sha256(readme).hexdigest(),
                'credential_scan': 'known credentials and private-key blocks clean'}
    payload['manifest.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    fd, filename = tempfile.mkstemp(prefix='hf-harness-merge-', suffix='.tar.gz', dir='/private/tmp')
    os.close(fd)
    with tarfile.open(filename, 'w:gz') as archive:
        for relative, data in payload.items():
            entry = tarfile.TarInfo(relative)
            entry.size, entry.mode = len(data), 0o600
            archive.addfile(entry, io.BytesIO(data))
    receipt = {'path': filename, 'sha256': hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
               'files': len(files), 'mapping_count': len(mapping), 'credential_scan': 'clean'}
    (records / 'bundle.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (records / 'upload_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

"""Reorganize the existing cluster harness, preserving its prior layout."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
import shutil
import stat
import sys
import tarfile

ROOT = Path('/home/tione/notebook/home/zuodarui/fieldquantum/hf')
LIMIT = 64 * 1024 * 1024


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or str(path) != name or not name or any(p in ('.', '..') for p in path.parts):
        raise ValueError('Invalid relative path')


def no_symlinks(path, stop):
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('Unexpected symlink: ' + str(parent))
        if parent == stop:
            break


def inventory(project, mapping):
    result = {}
    for name in mapping:
        source = project / name
        no_symlinks(source, project)
        if source.is_dir():
            entries = []
            for base, directories, names in os.walk(source, followlinks=False):
                for directory in list(directories):
                    path = Path(base) / directory
                    if path.is_symlink():
                        raise ValueError('Nested symlink in source: ' + str(path))
                    if directory == '__pycache__':
                        directories.remove(directory)
                entries.extend(Path(base) / child for child in names)
        else:
            entries = [source]
        for path in entries:
            if path.name in {'.DS_Store', '.controller.lock'} or path.suffix in {'.pyc', '.pyo'}:
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('Nonregular source: ' + str(path))
            result[str(path.relative_to(project))] = {
                'sha256': sha(path), 'bytes': info.st_size, 'mode': stat.S_IMODE(info.st_mode)}
    return result


def main():
    raw = sys.stdin.buffer.read(LIMIT + 1)
    if len(raw) > LIMIT or ROOT.resolve() != ROOT or not ROOT.is_dir():
        raise ValueError('Invalid upload or project directory')
    destination = ROOT / 'harness_framework'
    if destination.exists() or destination.is_symlink():
        raise ValueError('Destination already exists; no merge may overwrite an unknown framework')
    registry = ROOT.parent / '.hf-deployments'
    no_symlinks(registry / '.sync.lock', ROOT.parent)
    registry.mkdir(mode=0o700, exist_ok=True)
    with (registry / '.sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        name = 'harness-merge-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        deployment = registry / name
        deployment.mkdir(mode=0o700)
        stage, backup = deployment / 'framework-stage', deployment / 'prior-layout'
        stage.mkdir(mode=0o700)
        backup.mkdir(mode=0o700)
        (deployment / 'upload.tar.gz').write_bytes(raw)
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
            members = archive.getmembers()
            if len({m.name for m in members}) != len(members) or any(not m.isfile() for m in members):
                raise ValueError('Unsafe archive member')
            header = archive.getmember('manifest.json')
            if header.size > 3_000_000:
                raise ValueError('Manifest too large')
            manifest = json.loads(archive.extractfile(header).read())
            files, mapping, before = manifest['files'], manifest['mapping'], manifest['remote_before']
            for old, new in mapping.items():
                safe_relative(old)
                safe_relative(new)
            snapshot = inventory(ROOT, mapping)
            if snapshot != before['files']:
                raise ValueError('Remote source inventory changed since inspection')
            expected = {'manifest.json', 'project/README_GPU_HF_zh.md', *('framework/' + p for p in files)}
            if {m.name for m in members} != expected or sum(m.size for m in members) > LIMIT:
                raise ValueError('Manifest/member mismatch')
            # The pinned remote inventory protects remote edits made after inspection.
            for old, entry in before['files'].items():
                safe_relative(old)
                path = ROOT / old
                no_symlinks(path, ROOT)
                if not path.is_file() or sha(path) != entry['sha256']:
                    raise ValueError('Remote file changed before migration: ' + old)
            for old, digest in before['protected'].items():
                if sha(ROOT / old) != digest:
                    raise ValueError('Protected production file changed before migration')
            for old, new in mapping.items():
                safe_relative(old)
                safe_relative(new)
                source, target = ROOT / old, stage / new
                no_symlinks(source, ROOT)
                if source.is_dir():
                    shutil.copytree(source, target,
                                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '.DS_Store', '.controller.lock'))
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            for relative, entry in files.items():
                safe_relative(relative)
                if type(entry['mode']) is not int or not 0 <= entry['mode'] <= 0o777:
                    raise ValueError('Invalid source permissions')
                member = archive.getmember('framework/' + relative)
                if member.size != entry['bytes']:
                    raise ValueError('Unexpected source size')
                data = archive.extractfile(member).read(entry['bytes'] + 1)
                if hashlib.sha256(data).hexdigest() != entry['sha256']:
                    raise ValueError('Source upload hash mismatch')
                target = stage / relative
                no_symlinks(target, stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(entry['mode'])
            readme = archive.extractfile('project/README_GPU_HF_zh.md').read()
            if hashlib.sha256(readme).hexdigest() != manifest['root_readme_sha256']:
                raise ValueError('Root README hash mismatch')
        # Historical outputs and dated reports keep their original bytes.
        preserved = {}
        for old, entry in before['files'].items():
            source_root = next(p for p in mapping if old == p or old.startswith(p + '/'))
            new = mapping[source_root] + old[len(source_root):]
            if old.startswith(('outputs/', 'docs/')) or new not in files:
                if sha(stage / new) != entry['sha256']:
                    raise ValueError('Historical/remote-only evidence changed: ' + old)
                preserved[new] = entry['sha256']
        uploaded_bad = [p for p, e in files.items() if sha(stage / p) != e['sha256']]
        if uploaded_bad:
            raise ValueError('Staged files differ from upload')
        moved, installed, readme_replaced = [], False, False
        root_readme = ROOT / 'README_GPU_HF_zh.md'
        no_symlinks(root_readme, ROOT)
        readme_before_sha = sha(root_readme)
        shutil.copy2(root_readme, deployment / 'README_GPU_HF_zh.md.before')
        if sha(deployment / 'README_GPU_HF_zh.md.before') != readme_before_sha:
            raise ValueError('Root documentation changed during backup')
        try:
            for old in mapping:
                source, target = ROOT / old, backup / old
                target.parent.mkdir(parents=True, exist_ok=True)
                moved.append(old)
                os.rename(source, target)
            if inventory(backup, mapping) != snapshot:
                raise ValueError('Source files changed during migration; restoring prior layout')
            if sha(root_readme) != readme_before_sha:
                raise ValueError('Root documentation changed before commit')
            os.rename(stage, destination)
            installed = True
            temporary = ROOT / ('.README_GPU_HF-' + name)
            temporary.write_bytes(readme)
            os.replace(temporary, root_readme)
            readme_replaced = True
            for old, digest in before['protected'].items():
                if sha(ROOT / old) != digest:
                    raise ValueError('Production source identity changed')
            if any((ROOT / old).exists() for old in mapping):
                raise ValueError('Old harness location still active')
        except BaseException:
            if installed or destination.exists():
                os.rename(destination, deployment / 'failed-layout')
            for old in reversed(moved):
                if (backup / old).exists() and not (ROOT / old).exists():
                    os.rename(backup / old, ROOT / old)
            if readme_replaced:
                shutil.copy2(deployment / 'README_GPU_HF_zh.md.before', root_readme)
            raise
        receipt = {
            'schema_version': 'hf-harness-merge/1', 'status': 'migrated',
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'framework_root': str(destination), 'backup_directory': str(backup),
            'mapping': mapping, 'upload_files_verified': len(files),
            'historical_or_remote_only_files_preserved': len(preserved),
            'preserved_sha256': preserved, 'protected_production_sha256': before['protected'],
            'old_locations_removed_from_active_tree': True,
            'gpu_operations': 0, 'model_requests': 0,
            'historical_identity_rewritten': False,
        }
        output = destination / 'outputs/reorganization_20261113'
        output.mkdir(parents=True, exist_ok=True)
        (output / 'cluster_migration.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        (output / 'upload_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        (deployment / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()

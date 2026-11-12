from pathlib import Path


import hashlib


import json


import os


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(repo, path, must_exist=True):
    repo = Path(repo).resolve(strict=True)
    value = Path(path)
    value = (value if value.is_absolute() else repo / value).resolve(strict=must_exist)
    if not value.is_relative_to(repo) or value == repo:
        raise ValueError('Path must be a strict child of the declared repository')
    return value


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)

#!/usr/bin/env python
"""Unpack fixed supplementary inputs into their isolated analysis workspace."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import argparse
import hashlib
import shutil

import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/default.yaml')
    args=parser.parse_args()
    config=Path(args.config).resolve()
    root=config.parent.parent
    raw=yaml.safe_load(config.read_text(encoding='utf-8'))
    settings=raw['release_data']['supplement']
    archive=root/settings['snapshot_archive']
    destination=(root/settings['workspace']).resolve()
    if not destination.is_relative_to(root):
        raise ValueError('The supplementary workspace must remain inside the repository')
    expected=settings['snapshot_sha256']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:
        raise ValueError('The released supplementary snapshot does not match its configuration hash')
    with ZipFile(archive) as source:
        for item in source.infolist():
            relative=PurePosixPath(item.filename)
            if item.is_dir():continue
            if relative.is_absolute() or '..' in relative.parts or '\\' in item.filename or ':' in item.filename:
                raise ValueError('Unsafe supplementary archive member')
            target=(destination/Path(*relative.parts)).resolve()
            if not target.is_relative_to(destination):raise ValueError('Unsafe destination')
            if target.exists():
                with source.open(item) as stream:
                    expected_file=hashlib.sha256(stream.read()).digest()
                if hashlib.sha256(target.read_bytes()).digest()!=expected_file:
                    raise FileExistsError('Refusing to replace a changed supplementary file: '+str(relative))
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with source.open(item) as reader,target.open('wb') as writer:
                shutil.copyfileobj(reader,writer)
    print('Supplementary inputs unpacked to '+str(destination))


if __name__=='__main__':main()

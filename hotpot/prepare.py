"""Copy bundled metadata into a data workspace and validate supplied arrays."""
import argparse
import json
from pathlib import Path
import shutil
from ._internal.data_protocol.manifest import BASE6


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--check',action='store_true',help='Check the six pre-training datasets')
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    source=repo/'metadata'
    if not source.is_dir():raise RuntimeError('Run from the source checkout with its metadata directory')
    destination=a.workspace.resolve()/'metadata'
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('cfdb_train.json','cfdb_test.json','cfdb_schema.json','wave_split.json'):
        if (source/name).resolve()!=(destination/name).resolve():shutil.copyfile(source/name,destination/name)
    cases=json.loads((source/'cfdb_cases.json').read_text(encoding='utf-8'))
    for key,values in cases.items():
        path=(destination/'cases'/key/'case.json').resolve()
        if not path.is_relative_to(destination):raise ValueError('Invalid case path')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(values,indent=2)+'\n',encoding='utf-8')
    from ._internal.data_protocol.cfdb_sidecar import CertifiedCFDBSidecar
    sidecar=CertifiedCFDBSidecar(destination/'cfdb_train.json')
    print(f'Metadata installed: {len(sidecar.entries)} training rows; {len(cases)} CFDB cases')
    if a.check:
        import h5py
        missing=[]
        for spec in BASE6:
            path=a.workspace/spec.train_path
            sample=path if spec.storage=='packed_hdf5' else path/'data_0.hdf5'
            if not sample.is_file():missing.append(str(sample));continue
            with h5py.File(sample,'r') as f:
                shape=f['data'].shape
                print(spec.key,shape)
                expected=(spec.native_size,spec.native_size,spec.total_frames)
                actual=shape[1:] if spec.storage=='packed_hdf5' else shape
                if spec.channels>1:expected=(*expected,spec.channels)
                elif len(actual)==4:expected=(*expected,1)
                if tuple(actual)!=expected:raise ValueError(f'{spec.key}: {actual} != {expected}')
        if missing:raise FileNotFoundError('Missing datasets:\n'+'\n'.join(missing))


if __name__=='__main__':main()

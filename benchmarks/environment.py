"""Print software versions and CuTe source fingerprints for experiment records."""
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path

TESTED = {
 'flash_attn.cute.interface':'0a6966d5aeccc05b626a117fb7682e64035148bb656216db5027dafc55e9bb7e',
 'flash_attn.cute.flash_fwd_sm90':'0b7194125e722c7b2f1b9780d308cb85f381a72455d5e35e37a5f0b7c86207e5',
}


def collect():
    result=dict(python=platform.python_version(),packages={},sources={})
    for name in ('torch','nvidia-cutlass-dsl','quack-kernels','apache-tvm-ffi','cuda-python','magi-attention'):
        try: result['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['packages'][name]=None
    for name,tested in TESTED.items():
        try:
            spec=importlib.util.find_spec(name)
            digest=hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
            result['sources'][name]=dict(sha256=digest,tested_sha256=tested,matches_tested=digest==tested)
        except (ModuleNotFoundError,AttributeError,ImportError) as error:
            result['sources'][name]=dict(error=str(error),tested_sha256=tested)
    return result

if __name__=='__main__': print(json.dumps(collect(),indent=2))

"""Allowlisted loader for LIBERO / LIBERO-PRO *.pruned_init files (torch zip containing a pickled
numpy array). Third-party pickles must not execute code: only numpy array/dtype reconstruction
(+ _codecs.encode used by numpy byte payloads) is permitted; anything else raises.

install() replaces torch.load ONLY for paths ending in .pruned_init; all other torch.load calls
are untouched (and keep torch's own weights_only default)."""
import io
import pickle
import zipfile

import numpy as np

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),
}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            if module == "_codecs":
                import _codecs
                return getattr(_codecs, name)
            mod = __import__(module, fromlist=[name])
            return getattr(mod, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError(f"blocked persistent id {pid!r}")


def load_init(path):
    raw = open(path, "rb").read()
    buf = io.BytesIO(raw)
    if zipfile.is_zipfile(buf):
        z = zipfile.ZipFile(buf)
        name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        data = z.read(name)
    else:
        data = raw
    out = _Restricted(io.BytesIO(data)).load()
    return np.asarray(out)


def install():
    import torch
    _orig = torch.load

    def _load(f, *a, **k):
        if isinstance(f, (str, bytes)) and str(f).endswith(".pruned_init"):
            return load_init(f)
        return _orig(f, *a, **k)

    torch.load = _load

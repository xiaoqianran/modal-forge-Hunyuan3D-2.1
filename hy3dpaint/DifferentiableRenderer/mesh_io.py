"""GLB unknown-chunk compatibility for the project's pinned trimesh.

Only the parser's in-memory view omits unknown chunks. The source file (which
may carry provenance credentials) is never rewritten. Re-exporting geometry
does not preserve or validate those credentials.
"""
import io
import os
import struct
from pathlib import Path

import trimesh


def load_mesh_file(path, **kwargs):
    if not isinstance(path, (str, os.PathLike)) or Path(path).suffix.lower() != ".glb":
        return trimesh.load(path, **kwargs)

    data = Path(path).read_bytes()
    if len(data) < 12:
        raise ValueError("Truncated GLB header")
    magic, version, length = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or length != len(data):
        raise ValueError("Invalid GLB header or declared length")

    chunks = []
    unknown = False
    offset = 12
    while offset < length:
        if offset + 8 > length:
            raise ValueError("Truncated GLB chunk header")
        size, kind = struct.unpack_from("<I4s", data, offset)
        end = offset + 8 + size
        if size % 4 or end > length:
            raise ValueError("Invalid GLB chunk length")
        if offset == 12 and kind != b"JSON":
            raise ValueError("First GLB chunk must be JSON")
        if kind in (b"JSON", b"BIN\x00"):
            chunks.append(data[offset:end])
        else:
            unknown = True
        offset = end

    if not unknown:
        return trimesh.load(path, **kwargs)
    payload = b"".join(chunks)
    core = struct.pack("<4sII", b"glTF", 2, 12 + len(payload)) + payload
    kwargs.setdefault("resolver", trimesh.resolvers.FilePathResolver(str(Path(path).resolve().parent)))
    return trimesh.load(io.BytesIO(core), file_type="glb", **kwargs)

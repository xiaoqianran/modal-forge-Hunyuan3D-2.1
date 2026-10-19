import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hy3dpaint"))
from DifferentiableRenderer.mesh_io import load_mesh_file


class MeshIOTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.box()
        self.original = self.mesh.export(file_type="glb")

    def test_unknown_chunks_preserve_geometry_and_source(self):
        for kind in (b"C2PA", b"TEST"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                extra = struct.pack("<I4s", 4, kind) + b"test"
                data = self.original[:8] + struct.pack("<I", len(self.original) + len(extra)) + self.original[12:] + extra
                path = Path(folder) / "mesh.glb"
                path.write_bytes(data)
                loaded = load_mesh_file(path, force="mesh", process=False)
                np.testing.assert_allclose(loaded.vertices, self.mesh.vertices)
                np.testing.assert_array_equal(loaded.faces, self.mesh.faces)
                self.assertEqual(path.read_bytes(), data)

    def test_plain_glb_and_obj(self):
        with tempfile.TemporaryDirectory() as folder:
            for suffix in ("glb", "obj"):
                path = Path(folder) / f"mesh.{suffix}"
                self.mesh.export(path)
                self.assertEqual(len(load_mesh_file(path, force="mesh").faces), 12)

    def test_truncated_extra_chunk_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            extra = struct.pack("<I4s", 100, b"TEST") + b"test"
            data = self.original[:8] + struct.pack("<I", len(self.original) + len(extra)) + self.original[12:] + extra
            path = Path(folder) / "mesh.glb"
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "chunk length"):
                load_mesh_file(path)


if __name__ == "__main__":
    unittest.main()

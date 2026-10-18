"""Unit-test #215 v2 without loading CUDA, Blender or Paint dependencies."""
import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[1] / "hy3dpaint/DifferentiableRenderer/mesh_utils.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "convert_obj_to_glb")
function.decorator_list = []
code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(SOURCE), "exec")

class ExportTests(unittest.TestCase):
    def invoke(self, converter, output):
        namespace = {"HAS_BPY": False, "_convert_obj_to_glb_trimesh": converter}
        exec(code, namespace)
        return namespace["convert_obj_to_glb"]("example.obj", str(output))

    def test_success(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "model.glb"
            def success(source, target):
                Path(target).write_bytes(b"glTF")
                return True
            self.assertTrue(self.invoke(success, out))

    def test_false_return(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "returned False"):
                self.invoke(lambda *_: False, Path(directory) / "model.glb")

    def test_missing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                self.invoke(lambda *_: True, Path(directory) / "missing.glb")

    def test_empty_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "empty.glb"
            out.touch()
            with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                self.invoke(lambda *_: True, out)

    def test_original_error_preserved(self):
        def failure(*_):
            raise ValueError("original conversion error")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "conversion failed") as caught:
                self.invoke(failure, Path(directory) / "model.glb")
            self.assertIsInstance(caught.exception.__cause__, ValueError)
            self.assertEqual(str(caught.exception.__cause__), "original conversion error")

if __name__ == "__main__":
    unittest.main()

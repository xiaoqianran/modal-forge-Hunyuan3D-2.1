"""Quality gates must reject known collapses without treating legitimate boxes as errors."""
import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "hy3dshape/hy3dshape"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality = load_file("quality_under_test", ROOT / "quality.py")
rembg = load_file("foreground_under_test", ROOT / "rembg.py")
tree = ast.parse((ROOT / "models/denoisers/hunyuandit.py").read_text(encoding="utf-8"))
nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
         and node.name in ("StableLayerNorm", "StableRMSNorm", "shape_attention")]
namespace = dict(torch=torch, nn=nn, F=F)
exec(compile(ast.Module(body=nodes, type_ignores=[]), "hunyuandit.py", "exec"), namespace)


class PrecisionTests(unittest.TestCase):
    def test_chunked_fp32_attention_matches_full_fp32_attention(self):
        torch.manual_seed(1)
        q, k, v = [torch.randn(2, 3, 7, 8).half() * 100 for _ in range(3)]
        result = namespace["shape_attention"](q, k, v, upcast=True)
        expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).half()
        torch.testing.assert_close(result, expected)
        self.assertTrue(torch.isfinite(result).all())

    def test_epsilon_follows_runtime_dtype_and_preserves_checkpoint_keys(self):
        for stable, original, operation in (
            (namespace["StableLayerNorm"], nn.LayerNorm, F.layer_norm),
            (namespace["StableRMSNorm"], nn.RMSNorm, F.rms_norm),
        ):
            norm = stable(8, eps=1e-6)
            self.assertEqual(set(norm.state_dict()), set(original(8).state_dict()))
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                x = torch.linspace(-.001, .001, 16).reshape(2, 8).to(dtype)
                norm.to(dtype)
                eps = 1 / 65504 if dtype == torch.float16 else 1e-6
                args = (norm.weight, norm.bias, eps) if stable.__name__ == "StableLayerNorm" else (norm.weight, eps)
                torch.testing.assert_close(norm(x), operation(x, (8,), *args))
                self.assertTrue(torch.isfinite(norm(x)).all())

    def test_legacy_switch_is_an_actual_control(self):
        norm = namespace["StableRMSNorm"](8, eps=1e-6).half()
        x = torch.full((2, 8), .001, dtype=torch.float16)
        stable = norm(x)
        norm.fp16_safe = False
        torch.testing.assert_close(norm(x), F.rms_norm(x, (8,), norm.weight, 1e-6))
        self.assertFalse(torch.equal(norm(x), stable))


class ForegroundTests(unittest.TestCase):
    def test_manual_mask_wins_and_preserves_existing_alpha(self):
        image = Image.new("RGBA", (10, 10), (20, 30, 40, 128))
        mask = Image.new("L", image.size, 255)
        mask.putpixel((0, 0), 0)
        result = rembg.prepare_foreground(image, remover=lambda _: self.fail("remover called"), mask=mask)
        self.assertEqual(result.getpixel((0, 0))[3], 0)
        self.assertEqual(result.getpixel((1, 1))[3], 128)

    def test_wrong_size_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "尺寸"):
            rembg.prepare_foreground(Image.new("RGB", (8, 8)), mask=Image.new("L", (9, 8)))

    def test_almost_empty_mask_is_rejected(self):
        image = Image.new("RGBA", (100, 100), (10, 20, 30, 0))
        for y in range(4):
            for x in range(5):
                image.putpixel((x, y), (10, 20, 30, 255))
        row = quality.inspect_foreground(image)
        self.assertAlmostEqual(row["visible_fraction"], .002)
        self.assertFalse(row["accepted"])
        self.assertFalse(quality.inspect_foreground(Image.new("RGBA", (100, 100)))["accepted"])

    def test_normal_mask_is_preserved_without_rembg_dependency(self):
        image = Image.new("RGBA", (20, 20), (10, 20, 30, 0))
        image.paste((10, 20, 30, 255), (5, 5, 15, 15))
        output = rembg.prepare_foreground(image)
        self.assertEqual(output.tobytes(), image.tobytes())
        self.assertTrue(quality.inspect_foreground(output)["accepted"])


class GeometryTests(unittest.TestCase):
    def test_cuboid_and_slab_are_flagged_but_can_be_explicitly_allowed(self):
        cube = trimesh.creation.box(extents=[1.1, 2, 1.66])
        self.assertIn("suspected_box_collapse", quality.inspect_mesh(cube)["issues"])
        self.assertTrue(quality.inspect_mesh(cube, allow_box_like=True)["accepted"])
        slab = trimesh.creation.box(extents=[2, 2, .0136])
        self.assertIn("suspected_flat_collapse", quality.inspect_mesh(slab)["issues"])
        self.assertTrue(quality.inspect_mesh(slab, allow_flat=True)["accepted"])

    def test_curved_object_is_accepted(self):
        sphere = trimesh.creation.icosphere(subdivisions=2)
        self.assertTrue(quality.inspect_mesh(sphere)["accepted"])

    def test_rotated_cuboids_and_slabs_do_not_evade_detection(self):
        for extents, issue in (([1, 2, 1.6], "suspected_box_collapse"),
                               ([2, 2, .01], "suspected_flat_collapse")):
            mesh = trimesh.creation.box(extents=extents)
            mesh.apply_transform(trimesh.transformations.euler_matrix(.23, .51, .79))
            self.assertIn(issue, quality.inspect_mesh(mesh)["issues"])

    def test_empty_and_nonfinite_geometry_is_rejected(self):
        self.assertFalse(quality.inspect_mesh(None)["accepted"])
        sphere = trimesh.creation.icosphere()
        sphere.vertices[0] = np.nan
        self.assertFalse(quality.inspect_mesh(sphere)["accepted"])

    def test_rejected_candidate_never_wins_even_with_best_silhouette(self):
        rejected = dict(accepted=False, score=1, seed=42)
        accepted = dict(accepted=True, score=.7, seed=1051)
        self.assertIs(quality.select_candidate([rejected, accepted]), accepted)
        self.assertIsNone(quality.select_candidate([rejected]))

    def test_silhouette_comparison_uses_depth_visibility(self):
        mask = np.array([[255, 0], [255, 0]], dtype=np.uint8)
        depth = np.array([[2., 0], [3., 0]])
        self.assertEqual(quality.silhouette_iou(mask, depth), 1.)
        self.assertEqual(quality.silhouette_iou(mask, np.zeros_like(depth)), 0.)


if __name__ == "__main__":
    unittest.main()

"""Run real renderer methods on CPU with a deterministic synthetic rasterizer."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PATH = Path(__file__).resolve().parents[1] / "hy3dpaint/DifferentiableRenderer/MeshRender.py"
tree = ast.parse(PATH.read_text(encoding="utf-8"))
renderer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MeshRender")


def method(name, **extra):
    node = next(n for n in renderer.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    namespace = dict(torch=torch, np=np, F=F, Image=Image,
                     get_mv_matrix=lambda **kw: np.eye(4),
                     transform_pos=lambda matrix, vertices, **kw: vertices,
                     meshVerticeInpaint=None,
                     _inpaint_import_error=ImportError("synthetic extension failure"))
    namespace.update(extra)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PATH), "exec"), namespace)
    return namespace[name]


class RenderEdgesTests(unittest.TestCase):
    def make_renderer(self, visible, shrink=0):
        rast = torch.zeros(1, 2, 2, 4)
        rast[..., 3] = float(visible)
        interpolations = iter((torch.zeros(1, 2, 2, 2), torch.ones(1, 2, 2, 1)))
        return SimpleNamespace(
            device="cpu", texture_size=(4, 4), camera_proj_mat=np.eye(4),
            camera_distance=1, shader_type="face", bake_mode="linear", bake_angle_thres=75,
            bake_unreliable_kernel_size=shrink,
            vtx_pos=torch.tensor([[0., 0., 1., 1.], [0., 1., 1., 1.], [1., 0., 1., 1.]]),
            pos_idx=torch.tensor([[0, 1, 2]]), vtx_uv=torch.zeros(3, 2),
            uv_idx=torch.tensor([[0, 1, 2]]),
            raster_rasterize=lambda *args, **kw: (rast, None),
            raster_interpolate=lambda *args: (next(interpolations), None),
            render_sketch_from_depth=lambda depth: torch.zeros_like(depth),
        )

    def test_empty_view_contributes_zero(self):
        outputs = method("back_project")(self.make_renderer(False), torch.ones(2, 2, 3), 0, 0)
        self.assertEqual([tuple(x.shape) for x in outputs], [(4, 4, 3), (4, 4, 1), (4, 4, 1)])
        self.assertTrue(all(torch.count_nonzero(x) == 0 for x in outputs))

    def test_constant_depth_is_finite(self):
        render = self.make_renderer(True)
        render.render_sketch_from_depth = Mock(side_effect=lambda depth: torch.zeros_like(depth))
        scatter = lambda height, width, uv, data: data.mean(0).expand(height, width, data.shape[-1])
        outputs = method("back_project", linear_grid_put_2d=scatter)(render, torch.ones(2, 2, 3), 0, 0)
        depth = render.render_sketch_from_depth.call_args.args[0]
        self.assertTrue(torch.isfinite(depth).all())
        self.assertTrue(all(torch.isfinite(x).all() for x in outputs))

    def test_eroded_view_contributes_zero(self):
        render = self.make_renderer(True, shrink=1)
        render.render_sketch_from_depth = lambda depth: torch.ones_like(depth)
        outputs = method("back_project")(render, torch.ones(2, 2, 3), 0, 0)
        self.assertTrue(all(torch.count_nonzero(x) == 0 for x in outputs))

    def test_extension_failure_preserves_cause(self):
        with self.assertRaisesRegex(RuntimeError, "compile_mesh_painter") as caught:
            method("uv_inpaint")(SimpleNamespace(), np.ones((2, 2, 3)), np.ones((2, 2)))
        self.assertIsInstance(caught.exception.__cause__, ImportError)


if __name__ == "__main__":
    unittest.main()

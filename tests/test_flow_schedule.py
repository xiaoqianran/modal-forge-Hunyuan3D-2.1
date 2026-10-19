"""CPU regression for #209; no weights or CUDA extensions required.

Load the real sampling method via AST to avoid importing the unrelated mesh
and VAE stack. Only the model, image conditioning and mesh export are mocked;
the pipeline loop, timestep retrieval and Euler scheduler run unchanged.
Run from the repository root: python -m unittest discover -s tests -p test_flow_schedule.py
"""
import ast
import importlib.util
import inspect
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1] / "hy3dshape" / "hy3dshape"
spec = importlib.util.spec_from_file_location("flow_scheduler_under_test", ROOT / "schedulers.py")
schedulers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = schedulers
spec.loader.exec_module(schedulers)


def load_sampling_method():
    path = ROOT / "pipelines.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    retrieve = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "retrieve_timesteps")
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "Hunyuan3DDiTFlowMatchingPipeline")
    sampling = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef)
                    and node.name == "__call__")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        retrieve, sampling,
    ], type_ignores=[])
    namespace = dict(np=np, torch=torch, inspect=inspect,
                     synchronize_timer=lambda *args: nullcontext(),
                     tqdm=lambda iterable, **kwargs: iterable)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["__call__"]


SAMPLE = load_sampling_method()


class FlowScheduleTests(unittest.TestCase):
    def run_sampling(self, steps, shift=1.0, sigmas=None):
        scheduler = schedulers.FlowMatchEulerDiscreteScheduler(shift=shift)
        changes = []
        calls = []

        def model(latents, timestep, cond, **kwargs):
            calls.append(timestep.clone())
            return torch.ones_like(latents)

        pipeline = SimpleNamespace(
            device=torch.device("cpu"), dtype=torch.float32, model=model,
            scheduler=scheduler, set_surface_extractor=lambda *args: None,
            prepare_image=lambda *args: {"image": torch.zeros(1, 3, 2, 2)},
            encode_cond=lambda **kwargs: None,
            prepare_latents=lambda *args: torch.zeros(1, 2),
            _export=lambda latents, *args, **kwargs: latents,
        )
        previous = torch.zeros(1, 2)

        def callback(index, timestep, output):
            nonlocal previous
            changes.append(torch.any(output.prev_sample != previous).item())
            previous = output.prev_sample.clone()

        result = SAMPLE(pipeline, num_inference_steps=steps, sigmas=sigmas,
                        guidance_scale=-1, enable_pbar=False, output_type="latent",
                        callback=callback, callback_steps=1)
        return scheduler, calls, changes, result

    def test_default_has_requested_number_of_effective_updates(self):
        for steps in (1, 2, 50):
            for shift in (1.0, 3.0):
                with self.subTest(steps=steps, shift=shift):
                    scheduler, calls, changes, result = self.run_sampling(steps, shift)
                    self.assertEqual(len(calls), steps)
                    self.assertEqual(len(scheduler.sigmas), steps + 1)
                    self.assertEqual(scheduler.sigmas[0].item(), 0.0)
                    self.assertEqual(scheduler.sigmas[-1].item(), 1.0)
                    self.assertTrue(torch.all(torch.diff(scheduler.sigmas) > 0))
                    self.assertEqual(changes, [True] * steps)
                    torch.testing.assert_close(result, torch.ones_like(result))

    def test_custom_sigmas_remain_unchanged(self):
        for values in ([0.0, 0.2, 0.8], [0.0, 0.5, 1.0]):
            with self.subTest(values=values):
                custom = np.array(values)
                original = custom.copy()
                scheduler, calls, changes, _ = self.run_sampling(50, sigmas=custom)
                np.testing.assert_array_equal(custom, original)
                torch.testing.assert_close(scheduler.sigmas,
                                           torch.tensor(values + [1.0]))
                self.assertEqual(len(calls), len(values))
                self.assertEqual(changes[-1], values[-1] != 1.0)


if __name__ == "__main__":
    unittest.main()

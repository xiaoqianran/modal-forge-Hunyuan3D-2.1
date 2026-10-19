# Shape quality changes

Adapted individually, rather than merging unrelated community branches:

* [ComfyUI Hunyuan3D 2.1](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/hunyuan3dv2_1/hunyuandit.py): dtype-aware FP16 epsilon for Q/K, block, skip, and final norms. Checkpoint keys stay unchanged. FP32/BF16 keep their original epsilon. `fp16_safe=False` allows a legacy control.
* [Strix Halo](https://github.com/kroqueta-s/hunyuan3d-strix-halo/blob/main/runners/hunyuan3d/shape.py): optional head-wise FP32 math attention, scoped to this DiT. CUDA fused attention remains default. No ROCm environment changes or global SDPA patch.
* [ComfyUI #12772](https://github.com/Comfy-Org/ComfyUI/pull/12772): numerical guards stop non-finite DiT predictions, sampled latents and decoded features. SageAttention is not used here.
* [Official #155](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/pull/155): optional lazy RMBG-2.0 segmentation using 1024px normalized input and restored-size alpha. Uses the model owner's Hugging Face distribution. Access errors are explicit; the original rembg default is preserved.
* [photo-to-mesh](https://github.com/Hasasasaki/photo-to-mesh): explicit white-foreground masks, foreground inspection and preview; no silent full-frame fallback on mask failure.
* [Abstract3D](https://github.com/lpalbou/abstract3d#quality-presets-best-of-n-shape-selection): workflow pattern of generating/ranking raw Shape candidates before one Paint pass. Our implementation uses silhouette IoU and separate conservative box/slab rejection; it does not copy or claim their topology/detail scoring.

`hy3dshape.quality` contains reusable inspection, four clay renders, silhouette comparison and candidate selection. The Modal wrapper saves conditions, masks, all candidate meshes, views and metadata, including rejected runs. A cuboid/flat asset can be allowed explicitly. The heuristic checks cannot certify semantic fidelity, missing limbs, identity, or hidden geometry.

Same-initial-noise L40S comparison on t009 (2026-10-19): original FP16 98.2314%, stable FP16 98.2320%, BF16 97.9344% axis-aligned surface area. All three still collapse to a cuboid. The norm changes are a stability improvement, **not a demonstrated fix for this input**. Numerical checks alone do not detect a finite but wrong mesh.

Automatic bottom clipping was deliberately excluded: the community script can cut the subject and cannot reconstruct absent geometry. Synthesized reference images and alternative model families are separate generation choices, not transparent bug fixes.

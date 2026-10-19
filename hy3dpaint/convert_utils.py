import trimesh
import pygltflib
import numpy as np
from PIL import Image
import base64
import io

try:
    from .DifferentiableRenderer.mesh_io import load_mesh_file
except ImportError:
    from DifferentiableRenderer.mesh_io import load_mesh_file


def combine_metallic_roughness(metallic_path, roughness_path, output_path):
    """
    将metallic和roughness贴图合并为一张贴图
    GLB格式要求metallic在B通道，roughness在G通道
    """
    # 加载贴图
    metallic_img = Image.open(metallic_path).convert("L")  # 转为灰度
    roughness_img = Image.open(roughness_path).convert("L")  # 转为灰度

    # 确保尺寸一致
    if metallic_img.size != roughness_img.size:
        roughness_img = roughness_img.resize(metallic_img.size)

    # 创建RGB图像
    width, height = metallic_img.size
    combined = Image.new("RGB", (width, height))

    # 转为numpy数组便于操作
    metallic_array = np.array(metallic_img)
    roughness_array = np.array(roughness_img)

    # 创建合并的数组 (R, G, B) = (AO, Roughness, Metallic)
    combined_array = np.zeros((height, width, 3), dtype=np.uint8)
    combined_array[:, :, 0] = 255  # R通道：AO (如果没有AO贴图，设为白色)
    combined_array[:, :, 1] = roughness_array  # G通道：Roughness
    combined_array[:, :, 2] = metallic_array  # B通道：Metallic

    # 转回PIL图像并保存
    combined = Image.fromarray(combined_array)
    combined.save(output_path)
    return output_path


def create_glb_with_pbr_materials(obj_path, textures_dict, output_path):
    """Export PBR GLB without shared temp files; preserve all mesh primitives."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    inputs = dict(textures_dict)
    mesh = load_mesh_file(obj_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    def image_uri(path):
        with Image.open(path) as picture:
            fmt = picture.format
        mime = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(fmt)
        if mime is None:
            raise ValueError(f"Unsupported PBR image format: {path} ({fmt})")
        return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode('ascii')}"

    with TemporaryDirectory(prefix="hy3d-pbr-") as folder:
        temporary = Path(folder)
        intermediate = temporary / "mesh.glb"
        mesh.export(str(intermediate))
        gltf = pygltflib.GLTF2().load(str(intermediate))
        if inputs.get("metallic") and inputs.get("roughness"):
            packed = temporary / "metallic_roughness.png"
            combine_metallic_roughness(inputs["metallic"], inputs["roughness"], str(packed))
            inputs["metallicRoughness"] = str(packed)
        elif inputs.get("metallic") or inputs.get("roughness"):
            raise ValueError("Both metallic and roughness maps are required for PBR packing")

        texture_indices = {}
        images, textures = [], []
        for name in ("albedo", "metallicRoughness", "normal", "ao"):
            path = inputs.get(name)
            if not path:
                continue
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing {name} texture: {path}")
            texture_indices[name] = len(textures)
            images.append(pygltflib.Image(uri=image_uri(path)))
            textures.append(pygltflib.Texture(source=len(images) - 1))

        pbr = pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=1.0,
            roughnessFactor=1.0,
        )
        if "albedo" in texture_indices:
            pbr.baseColorTexture = pygltflib.TextureInfo(index=texture_indices["albedo"])
        if "metallicRoughness" in texture_indices:
            pbr.metallicRoughnessTexture = pygltflib.TextureInfo(index=texture_indices["metallicRoughness"])
        material = pygltflib.Material(name="PBR_Material", pbrMetallicRoughness=pbr)
        if "normal" in texture_indices:
            material.normalTexture = pygltflib.NormalTextureInfo(index=texture_indices["normal"])
        if "ao" in texture_indices:
            material.occlusionTexture = pygltflib.OcclusionTextureInfo(index=texture_indices["ao"])

        gltf.images = images
        gltf.textures = textures
        gltf.materials = [material]
        for gltf_mesh in gltf.meshes:
            for primitive in gltf_mesh.primitives:
                primitive.material = 0
        gltf.save(str(output))
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"PBR GLB output missing or empty: {output}")
    return str(output)

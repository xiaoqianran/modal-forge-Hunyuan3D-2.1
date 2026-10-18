import concurrent.futures
import base64
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from pygltflib import GLTF2
from convert_utils import create_glb_with_pbr_materials

def fixture(folder, n):
    folder.mkdir(parents=True, exist_ok=True)
    obj=folder/"mesh.obj"
    obj.write_text("o object\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
    color=(n, 20, 30)
    Image.new("RGB",(8,8),color).save(folder/"albedo.jpg")
    Image.new("L",(8,8),n).save(folder/"metallic.jpg")
    Image.new("L",(8,8),70).save(folder/"roughness.jpg")
    return obj, {"albedo":str(folder/"albedo.jpg"),"metallic":str(folder/"metallic.jpg"),"roughness":str(folder/"roughness.jpg")}

class GLBTests(unittest.TestCase):
    def test_success_mime_material(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);obj,tex=fixture(folder,190)
            out=folder/"result.glb"
            self.assertEqual(create_glb_with_pbr_materials(str(obj),tex,str(out)),str(out))
            gltf=GLTF2().load(str(out))
            self.assertTrue(gltf.images[0].uri.startswith("data:image/jpeg;"))
            self.assertEqual(gltf.meshes[0].primitives[0].material,0)
            self.assertIsNotNone(gltf.materials[0].pbrMetallicRoughness.metallicRoughnessTexture)
            self.assertFalse((Path.cwd()/"temp.glb").exists())

    def test_missing_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);obj,tex=fixture(folder,150)
            Path(tex["roughness"]).unlink()
            with self.assertRaises(FileNotFoundError):
                create_glb_with_pbr_materials(str(obj),tex,str(folder/"result.glb"))

    def test_optional_missing_albedo(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);obj,tex=fixture(folder,130);tex["albedo"]=None
            out=folder/"result.glb"
            create_glb_with_pbr_materials(str(obj),tex,str(out))
            gltf=GLTF2().load(str(out))
            self.assertIsNone(gltf.materials[0].pbrMetallicRoughness.baseColorTexture)
            self.assertIsNotNone(gltf.materials[0].pbrMetallicRoughness.metallicRoughnessTexture)

    def test_multiple_gltf_meshes(self):
        import trimesh
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);_,tex=fixture(folder,110)
            scene=trimesh.Scene()
            scene.add_geometry(trimesh.creation.box(), geom_name="box")
            scene.add_geometry(trimesh.creation.icosphere(subdivisions=1), geom_name="sphere")
            source=folder/"scene.glb"
            scene.export(str(source))
            out=folder/"out.glb"
            create_glb_with_pbr_materials(str(source),tex,str(out))
            gltf=GLTF2().load(str(out))
            self.assertGreaterEqual(len(gltf.meshes),2)
            self.assertTrue(all(p.material==0 for m in gltf.meshes for p in m.primitives))

    def test_parallel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def task(n):
                folder=root/str(n);obj,tex=fixture(folder,n)
                out=folder/"result.glb"
                create_glb_with_pbr_materials(str(obj),tex,str(out))
                gltf=GLTF2().load(str(out))
                encoded=gltf.images[0].uri.partition(",")[2]
                return (out.is_file(), out.stat().st_size,
                        base64.b64decode(encoded)==Path(tex["albedo"]).read_bytes())
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results=list(pool.map(task,range(30,46)))
            self.assertEqual(len(results),16)
            self.assertTrue(all(ok and size>0 and correct for ok,size,correct in results))

if __name__=="__main__":
    unittest.main()

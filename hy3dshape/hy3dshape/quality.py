"""Inspect foreground and raw Shape before spending time on texture generation.

Inspired by photo-to-mesh's explicit mask preview and Abstract3D's best-of-N
workflow. Scores are heuristics, not a guarantee of faithful 3D reconstruction.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def inspect_foreground(rgba, min_fraction=.01):
    alpha = np.asarray(rgba.convert("RGBA"))[..., 3]
    visible = alpha > 127
    count = int(visible.sum())
    fraction = float(visible.mean())
    issues = []
    if count < 16 or fraction < min_fraction:
        issues.append("foreground_too_small")
    bbox = None
    if count:
        y, x = np.nonzero(visible)
        bbox = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
    return {"visible_fraction": fraction, "visible_pixels": count,
            "bbox": bbox, "issues": issues, "accepted": not issues}


def foreground_preview(rgba):
    """Checkerboard makes lost foreground and opaque backgrounds visible."""
    y, x = np.indices((rgba.height, rgba.width))
    gray = np.where((x // 24 + y // 24) % 2, 205, 245).astype(np.uint8)
    background = Image.fromarray(np.dstack([gray, gray, gray, np.full_like(gray, 255)]), "RGBA")
    return Image.alpha_composite(background, rgba.convert("RGBA")).convert("RGB")


def inspect_mesh(mesh, allow_box_like=False, allow_flat=False):
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return {"accepted": False, "issues": ["empty_mesh"]}
    if not np.isfinite(mesh.vertices).all():
        return {"accepted": False, "issues": ["non_finite_vertices"]}
    extents = np.asarray(mesh.extents)
    area = float(mesh.area)
    if not np.isfinite(area) or area <= 0 or extents.max() <= 0:
        return {"accepted": False, "issues": ["zero_area_mesh"]}
    thinness = float(extents.min() / extents.max())
    aligned = np.max(np.abs(mesh.face_normals), axis=1) > .999
    axis_area = float(mesh.area_faces[aligned].sum() / area)
    box_volume = float(np.prod(extents))
    volume_fill = float(abs(mesh.volume) / box_volume) if box_volume > 0 else 0.0
    # A collapse can rotate between seeds. Infer an orthogonal frame from the
    # area-weighted dominant face normals, rather than assuming world XYZ axes.
    normals = mesh.face_normals.copy()
    signs = np.sign(normals[np.arange(len(normals)), np.abs(normals).argmax(axis=1)])
    normals *= np.where(signs == 0, 1, signs)[:, None]
    bins, inverse = np.unique(np.rint(normals / .025).astype(np.int16), axis=0, return_inverse=True)
    weights = np.bincount(inverse, weights=mesh.area_faces)
    order = np.argsort(weights)[::-1]
    chosen = inverse == order[0]
    first = (normals[chosen] * mesh.area_faces[chosen, None]).sum(axis=0)
    first /= np.linalg.norm(first)
    second = None
    for index in order[1:]:
        normal = bins[index].astype(float)
        normal /= np.linalg.norm(normal)
        if abs(np.dot(first, normal)) < .1:
            chosen = inverse == index
            normal = (normals[chosen] * mesh.area_faces[chosen, None]).sum(axis=0)
            normal /= np.linalg.norm(normal)
            second = normal - first * np.dot(first, normal)
            break
    if second is None:
        second = np.eye(3)[np.abs(first).argmin()]
        second -= first * np.dot(first, second)
    second /= np.linalg.norm(second)
    frame = np.column_stack((first, second, np.cross(first, second)))
    frame_extents = np.ptp(mesh.vertices @ frame, axis=0)
    oriented_area = float(mesh.area_faces[np.max(np.abs(mesh.face_normals @ frame), axis=1) > .999].sum() / area)
    oriented_volume = float(np.prod(frame_extents))
    oriented_fill = float(abs(mesh.volume) / oriented_volume) if oriented_volume > 0 else 0.
    oriented_thinness = float(frame_extents.min() / frame_extents.max())
    issues = []
    # Match our observed >98% axis-aligned failures, while allowing callers to
    # explicitly describe legitimate boxes or thin assets. Do not cut geometry.
    if oriented_area > .95 and oriented_thinness >= .02 and oriented_fill > .75 and not allow_box_like:
        issues.append("suspected_box_collapse")
    if oriented_thinness < .02 and oriented_area > .90 and not allow_flat:
        issues.append("suspected_flat_collapse")
    return {"accepted": not issues, "issues": issues, "vertices": len(mesh.vertices),
            "faces": len(mesh.faces), "extents": extents.tolist(),
            "axis_aligned_area_ratio": axis_area, "thinness": thinness,
            "volume_bbox_ratio": volume_fill, "oriented_planar_area_ratio": oriented_area,
            "oriented_thinness": oriented_thinness, "oriented_volume_fill": oriented_fill}


def silhouette_iou(target, rendered):
    target = np.asarray(target) > 127
    rendered = np.asarray(rendered) > 0
    union = np.logical_or(target, rendered).sum()
    return float(np.logical_and(target, rendered).sum() / union) if union else 0.0


def render_shape(mesh, target_mask, output_dir, size=256, textured=False):
    """Four real clay renders + silhouette comparison; textures never hide shape.

    Camera search is deliberately limited: the input viewpoint is unknown, and
    matching a silhouette does not validate the unseen side of an object.
    """
    import pyrender

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(size, size)
    target = Image.fromarray(np.asarray(target_mask).squeeze().astype(np.uint8)).resize(
        (size, size), Image.Resampling.NEAREST)
    material = None if textured else pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[.6, .67, .78, 1], metallicFactor=0, roughnessFactor=.8,
    )
    drawable = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    center = mesh.bounds.mean(axis=0)
    directions = ((0, .05, 1), (1, .05, 0), (0, .05, -1), (1, .25, 1))
    sheet = Image.new("RGB", (size * len(directions), size + 24), "white")
    scores = []
    try:
        for index, direction in enumerate(directions):
            z = np.asarray(direction, dtype=float)
            z /= np.linalg.norm(z)
            x = np.cross([0, 1, 0], z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            rotation = np.column_stack((x, y, z))
            projected = (mesh.vertices - center) @ rotation
            span = np.ptp(projected[:, :2], axis=0).max() / .85
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = center + z * max(mesh.extents) * 3
            scene = pyrender.Scene(bg_color=[.95, .96, .98, 1], ambient_light=[.45] * 3)
            scene.add(drawable)
            scene.add(pyrender.OrthographicCamera(xmag=span / 2, ymag=span / 2,
                                                  znear=.001, zfar=max(mesh.extents) * 10), pose=pose)
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=pose)
            pixels, depth = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
            scores.append(silhouette_iou(target, depth))
            sheet.paste(Image.fromarray(pixels), (index * size, 24))
            ImageDraw.Draw(sheet).text((index * size + 6, 6), f"view {index + 1} IoU {scores[-1]:.3f}", fill="black")
        sheet.save(output_dir / "shape-preview.png")
    finally:
        renderer.delete()
    return {"silhouette_iou": max(scores), "view_ious": scores,
            "preview": "shape-preview.png"}


def select_candidate(candidates):
    """Never choose a rejected candidate just because all the others failed."""
    accepted = [row for row in candidates if row.get("accepted", False)]
    return max(accepted, key=lambda row: row["score"]) if accepted else None

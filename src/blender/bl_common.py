"""Shared Blender helpers: scene reset, render settings, the solved camera,
small mesh and material builders. Scene units are metres in the tank frame
(X right, Y away from the camera, Z up, origin at the tank's outer
front-left-bottom corner). Config files are in millimetres; MM converts.
"""
import json
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import paths  # noqa: E402

MM = 0.001


def cfg(name):
    return json.loads((paths.CONFIG / name).read_text(encoding="utf-8"))


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    return scene


QUALITY = {
    #            samples, adaptive threshold, resolution %
    "draft":   dict(samples=48,  threshold=0.08, pct=50),
    "preview": dict(samples=128, threshold=0.04, pct=100),
    "final":   dict(samples=384, threshold=0.015, pct=100),
}


def setup_render(scene, quality="preview", out_path=None):
    q = QUALITY[quality]
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                break
        except Exception:
            continue
    else:
        scene.cycles.device = "CPU"

    c = scene.cycles
    c.samples = q["samples"]
    c.use_adaptive_sampling = True
    c.adaptive_threshold = q["threshold"]
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    # Denoising on the GPU is faster but needs spare memory; when several renders
    # share the card, set AQUARIUM_CPU_DENOISE=1 to keep it off the GPU.
    try:
        c.denoising_use_gpu = os.environ.get("AQUARIUM_CPU_DENOISE") != "1"
    except Exception:
        pass
    # A tank is glass inside glass: long refraction chains must not be cut short.
    c.max_bounces = 24
    c.transmission_bounces = 24
    c.glossy_bounces = 12
    c.diffuse_bounces = 4
    c.volume_bounces = 2
    c.transparent_max_bounces = 32
    c.caustics_reflective = True
    c.caustics_refractive = True
    c.blur_glossy = 0.3
    c.sample_clamp_indirect = 8.0
    c.use_persistent_data = True

    r = scene.render
    r.resolution_x, r.resolution_y = 1080, 1920
    r.resolution_percentage = q["pct"]
    r.fps = 30
    r.film_transparent = False
    r.image_settings.file_format = "OPEN_EXR"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "16"
    r.image_settings.exr_codec = "DWAA"
    # Renders stay scene-linear; the display transform is applied in src/comp.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    if out_path:
        r.filepath = str(out_path)
    return scene.cycles.device


def add_camera(scene, focus_distance=0.70, fstop=9.5):
    """The solved phone camera. fstop is for the virtual 36 mm sensor and is
    chosen to give the phone's real aperture of roughly 3.3 mm."""
    cam = cfg("camera.json")
    data = bpy.data.cameras.new("PlateCamera")
    obj = bpy.data.objects.new("PlateCamera", data)
    scene.collection.objects.link(obj)
    b = cam["blender"]
    obj.matrix_world = Matrix(b["matrix_world"])
    data.sensor_fit = b["sensor_fit"]
    data.sensor_height = b["sensor_height_mm"]
    data.lens = b["lens_mm"]
    data.clip_start = 0.01
    data.clip_end = 50.0
    data.dof.use_dof = True
    data.dof.focus_distance = focus_distance
    data.dof.aperture_fstop = fstop
    scene.camera = obj
    return obj


def new_mesh_object(name, verts, faces, collection=None, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def box_object(name, lo, hi, collection=None):
    """Axis-aligned box with outward normals. lo and hi in metres."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return new_mesh_object(name, v, f, collection)


def new_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    return mat, nt, out


def emission_material(name, colour, strength=1.0):
    mat, nt, out = new_material(name)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*colour, 1.0)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return mat


def principled_material(name, base, roughness=0.5, metallic=0.0, **extra):
    mat, nt, out = new_material(name)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    for k, v in extra.items():
        bsdf.inputs[k].default_value = v
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def script_args():
    """Arguments after the '--' on Blender's command line."""
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []

def shot_frames(scape_id, pad=75):
    """First and one-past-last frame a scape is ever on screen, with padding to
    cover the transitions into and out of it."""
    seg = next(s for s in paths.timeline()["segments"] if s["id"] == scape_id)
    return seg["start"] - pad, seg["end"] + pad

def _rodrigues(rvec):
    import numpy as np
    r = np.array(rvec, dtype=np.float64)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def move_camera(cam_obj, frame):
    """Place the camera for this frame of the film's slow move (see src/camera_move.py)."""
    import numpy as np
    import camera_move
    cam = cfg("camera.json")
    R0 = _rodrigues(cam["opencv"]["rvec"])
    C0 = -R0.T @ np.array(cam["opencv"]["tvec_mm"])
    R, C = camera_move.moved(R0, C0, camera_move.offset(frame))
    Rb = R.T @ np.diag([1.0, -1.0, -1.0])       # OpenCV camera axes to Blender's
    M = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M[i][j] = float(Rb[i, j])
        M[i][3] = float(C[i]) * MM
    cam_obj.matrix_world = M

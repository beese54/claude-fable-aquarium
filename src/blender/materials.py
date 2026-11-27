"""Materials from the downloaded CC0 texture sets.

pbr_box() lays a scanned texture on any mesh by box projection in object space,
so generated geometry (substrate beds, driftwood) needs no UVs and shows no seams.
"""
import bpy

from bl_common import new_material
import paths

_KEYS = {
    "colour": ("_diff_", "_Color", "_col_"),
    "rough": ("_rough_", "_Roughness"),
    "height": ("_disp_", "_Displacement"),
    "normal": ("_nor_gl_", "_NormalGL"),
    "ao": ("_ao_", "_AmbientOcclusion"),
}


def texture_files(asset_id, source="polyhaven"):
    folder = paths.ASSETS / source / asset_id
    if not folder.is_dir():
        raise FileNotFoundError(f"texture set {asset_id} not found in {folder}; run src/assets/fetch_assets.py")
    found = {}
    for kind, keys in _KEYS.items():
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() in (".jpg", ".png") and any(k in f.name for k in keys):
                found[kind] = f
                break
    return found


def pbr_box(name, asset_id, source="polyhaven", scale=10.0, stretch=(1.0, 1.0, 1.0), tint=None,
            saturation=1.0, value=1.0, bump=0.6, bump_distance=0.001, roughness=None, blend=0.3,
            use_ao=True):
    """scale: texture repeats per metre of object space. tint multiplies the colour.
    bump uses the set's height map. roughness overrides the set's map when given."""
    files = texture_files(asset_id, source)
    mat, nt, out = new_material(name)
    N, L = nt.nodes, nt.links
    coord = N.new("ShaderNodeTexCoord")
    mapping = N.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = tuple(scale * s for s in stretch)
    L.new(coord.outputs["Object"], mapping.inputs["Vector"])

    def tex(kind, colour_data):
        if kind not in files:
            return None
        t = N.new("ShaderNodeTexImage")
        t.image = bpy.data.images.load(str(files[kind]), check_existing=True)
        if not colour_data:
            t.image.colorspace_settings.name = "Non-Color"
        t.projection = "BOX"
        t.projection_blend = blend
        L.new(mapping.outputs["Vector"], t.inputs["Vector"])
        return t.outputs["Color"]

    bsdf = N.new("ShaderNodeBsdfPrincipled")
    col = tex("colour", True)
    if col is not None:
        hsv = N.new("ShaderNodeHueSaturation")
        hsv.inputs["Saturation"].default_value = saturation
        hsv.inputs["Value"].default_value = value
        L.new(col, hsv.inputs["Color"])
        col = hsv.outputs["Color"]
        ao = tex("ao", False) if use_ao else None
        if ao is not None:
            m = N.new("ShaderNodeMix")
            m.data_type = "RGBA"
            m.blend_type = "MULTIPLY"
            m.inputs["Factor"].default_value = 0.8
            L.new(col, m.inputs["A"])
            L.new(ao, m.inputs["B"])
            col = m.outputs["Result"]
        if tint is not None:
            m = N.new("ShaderNodeMix")
            m.data_type = "RGBA"
            m.blend_type = "MULTIPLY"
            m.inputs["Factor"].default_value = 1.0
            m.inputs["B"].default_value = (*tint, 1.0)
            L.new(col, m.inputs["A"])
            col = m.outputs["Result"]
        L.new(col, bsdf.inputs["Base Color"])
    if roughness is not None:
        bsdf.inputs["Roughness"].default_value = roughness
    else:
        r = tex("rough", False)
        if r is not None:
            L.new(r, bsdf.inputs["Roughness"])
    h = tex("height", False)
    if h is not None and bump > 0:
        b = N.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump
        b.inputs["Distance"].default_value = bump_distance
        L.new(h, b.inputs["Height"])
        L.new(b.outputs["Normal"], bsdf.inputs["Normal"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


# Ready-made looks. Scales are repeats per metre; the sets are photographed at
# 0.4 to 2 m per tile, so a nano tank needs them shrunk hard to keep grain small.
def white_sand(name="WhiteSand"):
    return pbr_box(name, "moon_dusted_02", scale=9.0, tint=(1.0, 0.97, 0.90), saturation=0.6, value=1.55, bump=0.5, bump_distance=0.0006, roughness=0.55)


def beach_sand(name="BeachSand"):
    return pbr_box(name, "damp_beach_sand", scale=8.0, tint=(1.0, 0.95, 0.85), saturation=0.8, value=1.25, bump=0.5, bump_distance=0.0006, roughness=0.6)


def dark_gravel(name="DarkGravel"):
    return pbr_box(name, "pebbles", scale=11.0, value=0.75, bump=0.9, bump_distance=0.0012)


def river_pebbles(name="RiverPebbles"):
    return pbr_box(name, "ganges_river_pebbles", scale=7.0, value=0.9, bump=1.0, bump_distance=0.002)


def leaf_litter_floor(name="LeafLitterFloor"):
    return pbr_box(name, "dry_decay_leaves", scale=5.5, value=0.6, tint=(1.0, 0.82, 0.62), bump=1.0, bump_distance=0.002)


def driftwood(name="Driftwood", tint=(0.84, 0.60, 0.40), value=0.9):
    return pbr_box(name, "bark_brown_02", scale=14.0, stretch=(1.0, 1.0, 0.45), tint=tint, saturation=0.55, value=value, bump=0.9, bump_distance=0.0015, roughness=0.75, use_ao=False)


def dark_bogwood(name="Bogwood"):
    return pbr_box(name, "jolcham_oak_bark_01", scale=10.0, stretch=(1.0, 1.0, 0.5), tint=(0.55, 0.42, 0.33), saturation=0.5, value=0.7, bump=1.0, bump_distance=0.002, roughness=0.8, use_ao=False)


def moss_surface(name="MossSurface", dark=False):
    return pbr_box(name, "Moss004" if dark else "Moss002", source="ambientcg", scale=16.0, value=0.8 if dark else 0.9, bump=1.0, bump_distance=0.002, roughness=0.8)

"""Neon tetra (Paracheirodon innesi), built and posed by code.

The fish faces +X with its snout at the origin; Z is up. One mesh holds body,
fins and eyes on separate material slots, so a pose is a single pass over the
vertex array: a travelling lateral wave that grows toward the tail, plus a bend
when turning. Proportions follow the real animal at about 33 mm total length.
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from bl_common import MM, new_material

BODY_MM = 27.0        # snout to tail base
TAIL_MM = 7.5
TOTAL_MM = BODY_MM + TAIL_MM

# station, top, bottom, half-width (mm), measured from the lateral midline
_PROFILE = np.array([
    (0.00, 0.25, -0.25, 0.15),
    (0.03, 1.35, -1.20, 0.95),
    (0.08, 2.35, -2.05, 1.60),
    (0.14, 3.00, -2.75, 2.00),
    (0.22, 3.55, -3.40, 2.25),
    (0.34, 3.95, -3.95, 2.30),
    (0.48, 3.70, -3.85, 2.10),
    (0.62, 3.00, -3.15, 1.65),
    (0.76, 2.10, -2.15, 1.10),
    (0.88, 1.45, -1.45, 0.62),
    (0.95, 1.30, -1.30, 0.42),
    (1.00, 1.45, -1.45, 0.25),
])


def _profile(s):
    s = np.asarray(s, dtype=np.float64)
    return tuple(np.interp(s, _PROFILE[:, 0], _PROFILE[:, k]) for k in (1, 2, 3))


class _Builder:
    def __init__(self):
        self.verts, self.faces, self.uvs, self.mats = [], [], [], []

    def grid(self, pts, uv, nu, nv, mat, close_v=False):
        """pts indexed [u][v]; adds a quad sheet (or tube when close_v)."""
        off = len(self.verts)
        for i in range(nu):
            for j in range(nv):
                self.verts.append(tuple(pts[i][j]))
                self.uvs.append(tuple(uv[i][j]))
        vmax = nv if close_v else nv - 1
        for i in range(nu - 1):
            for j in range(vmax):
                a = off + i * nv + j
                b = off + i * nv + (j + 1) % nv
                self.faces.append((a, b, b + nv, a + nv))
                self.mats.append(mat)
        return off


def _fin(b, base_pts, dirs, lengths, mat, nu=6):
    """A fin: rays leave each base point along its direction for its length."""
    nv = len(base_pts)
    pts, uv = [], []
    for i in range(nu):
        t = i / (nu - 1)
        row, urow = [], []
        for j in range(nv):
            p = np.array(base_pts[j]) + np.array(dirs[j]) * lengths[j] * t
            row.append(p)
            urow.append((t, j / (nv - 1)))
        pts.append(row)
        uv.append(urow)
    b.grid(pts, uv, nu, nv, mat)


def _unit(v):
    v = np.array(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def build_tetra_mesh(name="NeonTetra"):
    b = _Builder()
    L = BODY_MM

    # --- body: lofted rings ---------------------------------------------------
    K, M = 44, 22
    ss = np.linspace(0, 1, K) ** 0.9
    top, bot, wid = _profile(ss)
    pts, uv = [], []
    for k in range(K):
        mid, half = (top[k] + bot[k]) / 2, (top[k] - bot[k]) / 2
        row, urow = [], []
        for j in range(M):
            th = 2 * math.pi * j / M
            c, s = math.cos(th), math.sin(th)
            y = wid[k] * math.copysign(abs(s) ** 0.85, s)
            z = mid + half * math.copysign(abs(c) ** 0.92, c)
            row.append((-ss[k] * L, y, z))
            urow.append((float(ss[k]), z / 10.0 + 0.5))
        pts.append(row)
        uv.append(urow)
    b.grid(pts, uv, K, M, 0, close_v=True)

    # --- eyes ----------------------------------------------------------------
    es = 0.105
    etop, ebot, ewid = _profile(es)
    for side in (1, -1):
        centre = np.array((-es * L, side * (float(ewid) - 0.52), 0.55))
        R = 1.08
        nu_e, nv_e = 9, 14
        pts, uv = [], []
        for i in range(nu_e):
            lat = math.pi * i / (nu_e - 1)          # 0 at the outward pole
            row, urow = [], []
            for j in range(nv_e):
                lon = 2 * math.pi * j / nv_e
                d = np.array((math.sin(lat) * math.cos(lon), side * math.cos(lat), math.sin(lat) * math.sin(lon)))
                row.append(centre + d * R * np.array((1.0, 0.72, 1.0)))
                urow.append((i / (nu_e - 1), j / nv_e))
            pts.append(row)
            uv.append(urow)
        off = b.grid(pts, uv, nu_e, nv_e, 2, close_v=True)
        # the two rings nearest the outward pole are the pupil
        first_face = len(b.faces) - (nu_e - 1) * nv_e
        for f in range(first_face, first_face + 2 * nv_e):
            b.mats[f] = 3

    # --- fins ----------------------------------------------------------------
    def top_at(s):
        return float(_profile(s)[0])

    def bot_at(s):
        return float(_profile(s)[1])

    def wid_at(s):
        return float(_profile(s)[2])

    # caudal: forked fan
    vs = np.linspace(-1, 1, 15)
    base = [(-L + 0.4, 0.0, 1.40 * v) for v in vs]
    dirs = [_unit((-math.cos(0.66 * v), 0.0, math.sin(0.66 * v))) for v in vs]
    lens = [2.3 + 5.6 * abs(v) ** 1.3 for v in vs]
    _fin(b, base, dirs, lens, 1, nu=7)

    # dorsal
    sv = np.linspace(0.455, 0.60, 7)
    base = [(-s * L, 0.0, top_at(s) - 0.25) for s in sv]
    dirs = [_unit((-math.sin(math.radians(32 + 16 * i / 6)), 0.0, math.cos(math.radians(32 + 16 * i / 6)))) for i in range(7)]
    lens = [5.2 * (1 - 0.70 * i / 6) * (0.80 if i == 0 else 1.0) for i in range(7)]
    _fin(b, base, dirs, lens, 1)

    # adipose
    sv = np.linspace(0.765, 0.80, 3)
    base = [(-s * L, 0.0, top_at(s) - 0.15) for s in sv]
    dirs = [_unit((-0.80, 0.0, 0.60))] * 3
    _fin(b, base, dirs, [1.0, 1.35, 0.9], 1, nu=3)

    # anal
    sv = np.linspace(0.565, 0.775, 8)
    base = [(-s * L, 0.0, bot_at(s) + 0.25) for s in sv]
    dirs = [_unit((-math.sin(math.radians(38 + 14 * i / 7)), 0.0, -math.cos(math.radians(38 + 14 * i / 7)))) for i in range(8)]
    lens = [4.3 * (1 - 0.66 * i / 7) * (0.85 if i == 0 else 1.0) for i in range(8)]
    _fin(b, base, dirs, lens, 1)

    # pelvic pair
    for side in (1, -1):
        sv = np.linspace(0.425, 0.47, 4)
        base = [(-s * L, side * 0.7, bot_at(s) + 0.35) for s in sv]
        dirs = [_unit((-0.58, side * 0.30, -0.76))] * 4
        _fin(b, base, dirs, [3.3, 3.0, 2.4, 1.6], 1, nu=4)

    # pectoral pair
    for side in (1, -1):
        zs = np.linspace(-1.35, -2.35, 5)
        s0 = 0.235
        base = [(-s0 * L - 0.25 * i, side * (wid_at(s0) - 0.12), z) for i, z in enumerate(zs)]
        dirs = [_unit((-0.80, side * 0.50, -0.20 - 0.10 * i)) for i in range(5)]
        _fin(b, base, dirs, [3.0, 3.7, 3.8, 3.3, 2.4], 1, nu=4)

    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(float(c) * MM for c in v) for v in b.verts], [], b.faces)
    uvl = me.uv_layers.new(name="UVMap")
    for loop in me.loops:
        uvl.data[loop.index].uv = b.uvs[loop.vertex_index]
    me.polygons.foreach_set("material_index", b.mats)
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()
    return me


# --- materials -----------------------------------------------------------------

def _smooth(nt, value, lo, hi, out_lo=0.0, out_hi=1.0):
    n = nt.nodes.new("ShaderNodeMapRange")
    n.interpolation_type = "SMOOTHSTEP"
    n.inputs["From Min"].default_value = lo
    n.inputs["From Max"].default_value = hi
    n.inputs["To Min"].default_value = out_lo
    n.inputs["To Max"].default_value = out_hi
    nt.links.new(value, n.inputs["Value"])
    return n.outputs["Result"]


def _math(nt, op, a, b=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    for sock, v in ((n.inputs[0], a), (n.inputs[1], b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            sock.default_value = v
        else:
            nt.links.new(v, sock)
    return n.outputs[0]


def _mix(nt, fac, a, b):
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = "RGBA"
    for sock, v in ((n.inputs["Factor"], fac), (n.inputs["A"], a), (n.inputs["B"], b)):
        if isinstance(v, (int, float)):
            sock.default_value = v
        elif isinstance(v, tuple):
            sock.default_value = (*v, 1.0)
        else:
            nt.links.new(v, sock)
    return n.outputs["Result"]


def tetra_materials():
    # body ------------------------------------------------------------------
    body, nt, out = new_material("TetraBody")
    N, L = nt.nodes, nt.links
    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    s = sep.outputs["X"]
    z = _math(nt, "MULTIPLY_ADD", sep.outputs["Y"], 10.0)
    N[-1].inputs[2].default_value = -5.0                       # z in mm from the midline
    zc = _math(nt, "MULTIPLY_ADD", s, -1.55)                   # stripe centre drops toward the tail
    N[-1].inputs[2].default_value = 2.00
    dz = _math(nt, "SUBTRACT", z, zc)
    adz = _math(nt, "ABSOLUTE", dz)
    band = _smooth(nt, adz, 0.62, 0.92, 1.0, 0.0)
    along = _math(nt, "MULTIPLY", _smooth(nt, s, 0.035, 0.07), _smooth(nt, s, 0.74, 0.80, 1.0, 0.0))
    blue_m = _math(nt, "MULTIPLY", band, along)
    red_m = _math(nt, "MULTIPLY", _smooth(nt, dz, -0.85, -0.50, 1.0, 0.0), _smooth(nt, s, 0.40, 0.52))
    back_m = _smooth(nt, dz, 0.55, 0.95)

    lw = N.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.45
    blue = _mix(nt, lw.outputs["Facing"], (0.00, 0.30, 1.00), (0.00, 0.95, 0.85))
    col = _mix(nt, red_m, (0.30, 0.32, 0.31), (0.50, 0.004, 0.004))
    col = _mix(nt, back_m, col, (0.085, 0.072, 0.030))
    col = _mix(nt, blue_m, col, blue)

    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col, bsdf.inputs["Base Color"])
    silver_m = _math(nt, "SUBTRACT", 1.0, _math(nt, "MAXIMUM", red_m, back_m))
    metal = _math(nt, "MAXIMUM", _math(nt, "MULTIPLY", blue_m, 0.9), _math(nt, "MULTIPLY", silver_m, 0.55))
    L.new(metal, bsdf.inputs["Metallic"])
    bsdf.inputs["Roughness"].default_value = 0.28
    L.new(blue, bsdf.inputs["Emission Color"])
    L.new(_math(nt, "MULTIPLY", blue_m, 0.85), bsdf.inputs["Emission Strength"])
    # A tetra is a small, thin, half-transparent animal: light gets into the body.
    bsdf.inputs["Subsurface Weight"].default_value = 0.35
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.6, 0.45)
    bsdf.inputs["Subsurface Scale"].default_value = 0.002
    L.new(bsdf.outputs[0], out.inputs["Surface"])

    # fins ------------------------------------------------------------------
    fin, nt, out = new_material("TetraFin")
    N, L = nt.nodes, nt.links
    uv = N.new("ShaderNodeUVMap")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(uv.outputs["UV"], sep.inputs[0])
    rays = N.new("ShaderNodeTexWave")
    rays.inputs["Scale"].default_value = 9.0
    rays.inputs["Distortion"].default_value = 0.0
    comb = N.new("ShaderNodeCombineXYZ")
    L.new(sep.outputs["Y"], comb.inputs[0])
    L.new(comb.outputs[0], rays.inputs["Vector"])
    alpha = _math(nt, "ADD", _smooth(nt, sep.outputs["X"], 0.0, 1.0, 0.30, 0.07), _math(nt, "MULTIPLY", rays.outputs["Fac"], 0.07))
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.55, 0.60, 0.56, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.25
    L.new(alpha, bsdf.inputs["Alpha"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])

    # eye -------------------------------------------------------------------
    iris, nt, out = new_material("TetraIris")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.72, 0.80, 0.86, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.9
    bsdf.inputs["Roughness"].default_value = 0.22
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])

    pupil, nt, out = new_material("TetraPupil")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.004, 0.004, 0.005, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.08
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return [body, fin, iris, pupil]


# --- posing ---------------------------------------------------------------------

class Tetra:
    """One fish in the scene. pose() is a pure function of its arguments."""

    _materials = None

    def __init__(self, name, collection=None, size=1.0):
        if Tetra._materials is None:
            Tetra._materials = tetra_materials()
        me = build_tetra_mesh(name)
        for m in Tetra._materials:
            me.materials.append(m)
        self.obj = bpy.data.objects.new(name, me)
        (collection or bpy.context.scene.collection).objects.link(self.obj)
        self.size = size
        n = len(me.vertices)
        rest = np.empty(n * 3, dtype=np.float32)
        me.vertices.foreach_get("co", rest)
        self.rest = rest.reshape(n, 3)
        self.d = np.clip(-self.rest[:, 0] / (TOTAL_MM * MM), 0.0, 1.0)     # 0 snout, 1 tail tip
        self.obj.cycles.is_caustics_receiver = True

    def pose(self, position, forward, phase=0.0, amp=0.5, bend=0.0, roll=0.0):
        """position: metres. forward: direction of travel. phase: tail-beat phase
        in radians. amp: 0 still .. 1 hard swimming. bend: -1..1 turn (positive
        swings the tail toward local -Y, steering the head toward +Y)."""
        d = self.d
        Lm = TOTAL_MM * MM
        env = (0.018 - 0.085 * d + 0.235 * d * d) * amp
        wave = env * Lm * np.sin(phase - 2 * math.pi * d / 1.05)
        co = self.rest.copy()
        co[:, 1] += wave - bend * 0.16 * Lm * d * d
        self.obj.data.vertices.foreach_set("co", co.ravel())
        self.obj.data.update()

        f = Vector(forward).normalized()
        up = Vector((0, 0, 1))
        side = up.cross(f)
        if side.length < 1e-6:
            side = Vector((0, 1, 0))
        side.normalize()
        up2 = f.cross(side)
        rot = Matrix((f, side, up2)).transposed().to_4x4()
        if roll:
            rot = rot @ Matrix.Rotation(roll, 4, "X")
        self.obj.matrix_world = Matrix.Translation(Vector(position)) @ rot @ Matrix.Scale(self.size, 4)

def shoal(n, centre_mm, spread_mm, seed=0, heading_deg=180.0, name="Tetra", keep_out=(), floor=None, z_max=250.0):
    """Place a loose shoal for a still: fish cluster round a centre, mostly
    sharing a heading but each with its own turn, pitch and roll, so the group
    shows every angle to the camera rather than a row of side views.

    keep_out   (x, y, radius, top_z) cylinders in mm that fish must stay clear of
    floor      function (x_mm, y_mm) -> ground height in mm, to stay above
    Returns the Tetra objects.
    """
    import random
    rng = random.Random(seed)
    spots, tries = [], 0
    while len(spots) < n and tries < 20000:
        tries += 1
        p = tuple(rng.gauss(c, s) for c, s in zip(centre_mm, spread_mm))
        if not (20 < p[0] < 280 and 20 < p[1] < 280 and p[2] < z_max):
            continue
        if floor is not None and p[2] < floor(p[0], p[1]) + 22:
            continue
        if any(math.hypot(p[0] - kx, p[1] - ky) < kr + 14 and p[2] < kz + 12 for kx, ky, kr, kz in keep_out):
            continue
        if any(math.dist(p, q) < 24 for q in spots):
            continue
        spots.append(p)
    out = []
    for i, (x, y, z) in enumerate(spots):
        stray = rng.random() < 0.2
        yaw = math.radians(heading_deg + (rng.uniform(-180, 180) if stray else rng.gauss(0, 38)))
        pitch = rng.gauss(0, 0.16)
        fwd = (math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch))
        t = Tetra(f"{name}_{i:02d}", size=rng.uniform(0.84, 1.08))
        t.pose((x * MM, y * MM, z * MM), fwd, phase=rng.uniform(0, 6.283), amp=rng.uniform(0.25, 0.7),
               bend=rng.gauss(0, 0.3), roll=rng.gauss(0, 0.09))
        out.append(t)
    return out

"""Neon tetra (Paracheirodon innesi), built and posed by code.

The fish faces +X with its snout at the origin; Z is up. One mesh holds body,
fins and eyes on separate material slots, so a pose is a single pass over the
vertex array: a travelling lateral wave that grows toward the tail, plus a bend
when turning.

Proportions follow the living animal: 26.5 mm from snout to tail base and
32.6 mm overall at size 1.0, a body about 27% as deep as that standard length and
half as wide as it is deep, a head a quarter of it with an eye 2.6 mm across.

How the skin is lit matters more than its colours. A neon's stripe is not
pigment: it is stacks of guanine platelets that lie tilted in the skin and
throw the light coming down from the surface out sideways. The shader does the
same thing, by tilting the stripe's shading normal toward the fish's back (and
allowing it the extra brightness of a reflector that aims its light, see
_GAIN), so the stripe is bright because it faces the lamp, goes dark in
shadow, and changes hue with the angle it is seen from. Nothing on the fish
emits light.
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from bl_common import MM, new_material

BODY_MM = 26.5        # standard length: snout to the base of the tail fin
TAIL_MM = 6.1         # how far the tail fin reaches beyond that
TOTAL_MM = BODY_MM + TAIL_MM

# station (fraction of standard length), top, bottom, half-width (mm)
_PROFILE = np.array([
    (0.000, 0.75, -0.82, 0.60),
    (0.012, 1.22, -1.24, 0.86),
    (0.035, 1.76, -1.68, 1.10),
    (0.075, 2.36, -2.18, 1.32),
    (0.115, 2.78, -2.62, 1.52),
    (0.170, 3.12, -3.04, 1.74),
    (0.240, 3.40, -3.48, 1.93),
    (0.340, 3.62, -3.84, 2.00),
    (0.450, 3.52, -3.70, 1.88),
    (0.560, 3.12, -3.18, 1.62),
    (0.670, 2.56, -2.50, 1.28),
    (0.780, 1.95, -1.88, 0.92),
    (0.880, 1.45, -1.42, 0.60),
    (0.950, 1.30, -1.28, 0.42),
    (1.000, 1.36, -1.34, 0.30),
    (1.030, 1.22, -1.20, 0.10),
])
_S_END = float(_PROFILE[-1, 0])
_PZ, _PY = 0.94, 0.88                    # cross-section: superellipse exponents (height, width)


def _profile(s):
    """top, bottom, half-width at station(s) s: a smooth curve through the table."""
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    x = _PROFILE[:, 0]
    i = np.clip(np.searchsorted(x, s, side="right") - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    t = np.clip((s - x[i]) / h, 0.0, 1.0)
    h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
    h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    out = []
    for k in (1, 2, 3):
        y = _PROFILE[:, k]
        d = np.gradient(y, x)
        out.append(h00 * y[i] + h10 * h * d[i] + h01 * y[i + 1] + h11 * h * d[i + 1])
    return tuple(out)


def _keel(s):
    """How much narrower the underside is than the back: most at the head, so
    the face is a wedge with the eyes at its widest, not a capsule."""
    return 0.10 + 0.12 * (1.0 - min(max(s / 0.25, 0.0), 1.0))


def _surface_y(s, z):
    """Half-width of the body at station s and height z (mm): where a fin base
    or an eye must sit to lie on the skin."""
    top, bot, wid = (float(v[0]) for v in _profile(s))
    mid, half = (top + bot) / 2, (top - bot) / 2
    u = min(max((z - mid) / half, -1.0), 1.0)
    c = math.copysign(abs(u) ** (1.0 / _PZ), u)
    return wid * (1.0 + _keel(s) * c) * max(1.0 - c * c, 0.0) ** (0.5 * _PY)


class _Builder:
    """Collects quads and triangles; UVs are kept per face corner so a closed
    tube can carry an angle coordinate without a smeared seam."""

    def __init__(self):
        self.verts, self.faces, self.mats = [], [], []
        self.uv, self.uv2 = [], []

    def grid(self, pts, uv, mat, uv2=None, close_v=False):
        """pts indexed [u][v]; adds a quad sheet (or a tube when close_v, in
        which case uv and uv2 carry one more column than pts)."""
        nu, nv = len(pts), len(pts[0])
        off = len(self.verts)
        for row in pts:
            self.verts.extend(tuple(float(c) for c in p) for p in row)
        for i in range(nu - 1):
            for j in range(nv if close_v else nv - 1):
                j2 = (j + 1) % nv
                self.faces.append((off + i * nv + j, off + i * nv + j2, off + (i + 1) * nv + j2, off + (i + 1) * nv + j))
                self.mats.append(mat)
                for a, c in ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)):
                    self.uv.append(tuple(uv[a][c]))
                    self.uv2.append(tuple(uv2[a][c]) if uv2 is not None else (0.0, 0.0))
        return off

    def fan(self, centre, ring, uv_c, uv_ring, uv2_c, mat, flip=False):
        """Close the end of a tube: ring is a list of existing vertex indices."""
        ci = len(self.verts)
        self.verts.append(tuple(float(c) for c in centre))
        n = len(ring)
        for j in range(n):
            a, b = ring[j], ring[(j + 1) % n]
            self.faces.append((ci, b, a) if flip else (ci, a, b))
            self.mats.append(mat)
            ua, ub = uv_ring[j], uv_ring[(j + 1) % n]
            for u in ((uv_c, ub, ua) if flip else (uv_c, ua, ub)):
                self.uv.append(tuple(u))
                self.uv2.append(tuple(uv2_c))


def _fin(b, base_pts, dirs, lengths, mat, rays, nu=6, caudal=False, sweep=0.0):
    """A fin: a ray leaves each base point along its direction for its length.
    UV is (distance along the ray 0..1, ray number), so the shader can draw the
    rays; the tail fin's first coordinate is shifted by 2 to mark it.
    sweep bends the rays back toward their tips (mm at the tip)."""
    nv = len(base_pts)
    pts, uv = [], []
    for i in range(nu):
        t = i / (nu - 1)
        row, urow = [], []
        for j in range(nv):
            p = np.array(base_pts[j], dtype=np.float64) + np.array(dirs[j]) * lengths[j] * t
            p[0] -= sweep * t * t
            row.append(p)
            urow.append((t + (2.0 if caudal else 0.0), rays * j / (nv - 1) if rays else 0.5))
        pts.append(row)
        uv.append(urow)
    b.grid(pts, uv, mat)


def _unit(v):
    v = np.array(v, dtype=np.float64)
    return v / np.linalg.norm(v)


EYE_S, EYE_Z, EYE_R = 0.108, 0.45, 1.30      # eye centre (station, height) and radius, mm


def build_tetra_mesh(name="NeonTetra"):
    b = _Builder()
    L = BODY_MM

    # --- body: lofted rings, closed at both ends -------------------------------
    K, M = 48, 24
    ss = _S_END * np.linspace(0, 1, K) ** 1.15            # rings a little closer on the head
    top, bot, wid = _profile(ss)
    pts, uv, uv2 = [], [], []
    for k in range(K):
        mid, half = (top[k] + bot[k]) / 2, (top[k] - bot[k]) / 2
        keel = _keel(ss[k])
        row, urow, u2row = [], [], []
        for j in range(M + 1):
            th = 2 * math.pi * j / M                       # 0 on the dorsal ridge
            c, s = math.cos(th), math.sin(th)
            y = wid[k] * (1.0 + keel * c) * math.copysign(abs(s) ** _PY, s)
            z = mid + half * math.copysign(abs(c) ** _PZ, c)
            if j < M:
                row.append((-ss[k] * L, y, z))
            urow.append((float(ss[k]), z / 10.0 + 0.5))    # station, height
            u2row.append((0.5 + 0.5 * c, j / M))           # belly 0 .. ridge 1, angle round the body
        pts.append(row)
        uv.append(urow)
        uv2.append(u2row)
    off = b.grid(pts, uv, 0, uv2=uv2, close_v=True)
    mid0 = (top[0] + bot[0]) / 2
    b.fan((0.30, 0.0, mid0), [off + j for j in range(M)], (0.0, mid0 / 10 + 0.5), [uv[0][j] for j in range(M)],
          (0.5, 0.0), 0, flip=True)
    last = off + (K - 1) * M
    b.fan((-ss[-1] * L - 0.05, 0.0, 0.0), [last + j for j in range(M)], (float(ss[-1]), 0.5),
          [uv[-1][j] for j in range(M)], (0.5, 0.0), 0)

    # --- eyes: shallow domes that follow the head's surface ----------------------
    radii = (0.0, 0.26, 0.50, 0.56, 0.74, 0.90, 1.0)       # the pupil is the inner half
    nv_e = 16
    for side in (1, -1):
        pts, uv = [], []
        for r in radii:
            row, urow = [], []
            for j in range(nv_e + 1):
                lon = side * 2 * math.pi * j / nv_e
                ex, ez = r * math.cos(lon), r * math.sin(lon)
                x, z = -EYE_S * L + EYE_R * ex, EYE_Z + EYE_R * ez
                y = _surface_y(-x / L, z) - 0.07 + 0.60 * math.sqrt(max(1.0 - r * r, 0.0))
                if j < nv_e:
                    row.append((x, side * y, z))
                urow.append((0.5 + 0.5 * ex, 0.5 + 0.5 * ez))
            pts.append(row)
            uv.append(urow)
        b.grid(pts, uv, 2, close_v=True)
        first_face = len(b.faces) - (len(radii) - 1) * nv_e
        for f in range(first_face, first_face + 2 * nv_e):
            b.mats[f] = 3

    # --- fins ----------------------------------------------------------------
    def top_at(s):
        return float(_profile(s)[0][0])

    def bot_at(s):
        return float(_profile(s)[1][0])

    # caudal: forked, nineteen rays
    vs = np.linspace(-1, 1, 19)
    base = [(-L + 0.35, 0.0, 1.30 * v) for v in vs]
    dirs = [_unit((-math.cos(0.50 * v), 0.0, math.sin(0.50 * v))) for v in vs]
    lens = [(2.5 + 5.3 * abs(v) ** 1.5) * (1.0 - 0.10 * max(abs(v) - 0.85, 0.0) / 0.15) for v in vs]
    _fin(b, base, dirs, lens, 1, rays=18, nu=8, caudal=True)

    # dorsal
    n = 9
    sv = np.linspace(0.470, 0.600, n)
    base = [(-s * L, 0.0, top_at(s) - 0.30) for s in sv]
    ang = [math.radians(26 + 30 * i / (n - 1)) for i in range(n)]
    dirs = [_unit((-math.sin(a), 0.0, math.cos(a))) for a in ang]
    lens = [3.9, 5.0, 5.2, 4.9, 4.3, 3.6, 2.9, 2.3, 1.8]
    _fin(b, base, dirs, lens, 1, rays=10, nu=7, sweep=0.5)

    # adipose: a small fleshy lobe, no rays
    sv = np.linspace(0.772, 0.806, 4)
    base = [(-s * L, 0.0, top_at(s) - 0.20) for s in sv]
    dirs = [_unit((-0.78, 0.0, 0.62))] * 4
    _fin(b, base, dirs, [0.9, 1.35, 1.45, 1.0], 1, rays=0, nu=3, sweep=0.2)

    # anal: long-based, the front rays longest
    n = 11
    sv = np.linspace(0.590, 0.800, n)
    base = [(-s * L, 0.0, bot_at(s) + 0.30) for s in sv]
    ang = [math.radians(30 + 30 * i / (n - 1)) for i in range(n)]
    dirs = [_unit((-math.sin(a), 0.0, -math.cos(a))) for a in ang]
    lens = [3.3, 4.3, 4.4, 3.9, 3.2, 2.6, 2.2, 1.9, 1.7, 1.5, 1.2]
    _fin(b, base, dirs, lens, 1, rays=12, nu=7, sweep=0.4)

    # pelvic pair
    for side in (1, -1):
        sv = np.linspace(0.440, 0.482, 4)
        base = [(-s * L, side * 0.45, bot_at(s) + 0.35) for s in sv]
        dirs = [_unit((-0.72, side * 0.24, -0.65))] * 4
        _fin(b, base, dirs, [3.4, 3.2, 2.6, 1.8], 1, rays=7, nu=4)

    # pectoral pair: low on the side, just behind the gill cover
    for side in (1, -1):
        zs = np.linspace(-1.55, -2.45, 5)
        s0 = 0.262
        base = []
        for i, z in enumerate(zs):
            s = s0 + 0.012 * i
            base.append((-s * L, side * (_surface_y(s, z) - 0.12), z))
        dirs = [_unit((-0.84, side * 0.40, -0.26 - 0.07 * i)) for i in range(5)]
        _fin(b, base, dirs, [3.2, 3.9, 3.9, 3.3, 2.4], 1, rays=10, nu=5)

    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(c * MM for c in v) for v in b.verts], [], b.faces)
    for layer, data in (("UVMap", b.uv), ("Ring", b.uv2)):
        uvl = me.uv_layers.new(name=layer)
        uvl.data.foreach_set("uv", np.asarray(data, dtype=np.float32).ravel())
    me.polygons.foreach_set("material_index", b.mats)
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()
    return me


# --- materials -----------------------------------------------------------------

def _set(nt, sock, v):
    if v is None:
        return
    if isinstance(v, (int, float)):
        sock.default_value = v
    elif isinstance(v, tuple):
        sock.default_value = (*v, 1.0) if len(sock.default_value) == 4 else v
    else:
        nt.links.new(v, sock)


def _smooth(nt, value, lo, hi, out_lo=0.0, out_hi=1.0):
    n = nt.nodes.new("ShaderNodeMapRange")
    n.interpolation_type = "SMOOTHSTEP"
    n.inputs["From Min"].default_value = lo
    n.inputs["From Max"].default_value = hi
    n.inputs["To Min"].default_value = out_lo
    n.inputs["To Max"].default_value = out_hi
    nt.links.new(value, n.inputs["Value"])
    return n.outputs["Result"]


def _math(nt, op, a, b=None, c=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    for sock, v in zip(n.inputs, (a, b, c)):
        _set(nt, sock, v)
    return n.outputs[0]


def _mul(nt, *vals):
    out = vals[0]
    for v in vals[1:]:
        out = _math(nt, "MULTIPLY", out, v)
    return out


def _max(nt, *vals):
    out = vals[0]
    for v in vals[1:]:
        out = _math(nt, "MAXIMUM", out, v)
    return out


def _mix(nt, fac, a, b):
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = "RGBA"
    for sock, v in ((n.inputs["Factor"], fac), (n.inputs["A"], a), (n.inputs["B"], b)):
        _set(nt, sock, v)
    return n.outputs["Result"]


def _vmath(nt, op, a, b=None, scale=None):
    n = nt.nodes.new("ShaderNodeVectorMath")
    n.operation = op
    _set(nt, n.inputs[0], a)
    _set(nt, n.inputs[1], b)
    if scale is not None:
        _set(nt, n.inputs["Scale"], scale)
    return n.outputs[0]


def _uv(nt, layer):
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = layer
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs[0])
    return uv.outputs["UV"], sep.outputs["X"], sep.outputs["Y"]


def _dorsal(nt):
    """The fish's own up direction, in world space."""
    n = nt.nodes.new("ShaderNodeVectorTransform")
    n.vector_type = "VECTOR"
    n.convert_from, n.convert_to = "OBJECT", "WORLD"
    n.inputs[0].default_value = (0.0, 0.0, 1.0)
    return _vmath(nt, "NORMALIZE", n.outputs[0])


# The stripe's two colours: seen square-on to the platelets it is blue-green,
# seen obliquely the reflected wavelength shortens to a deep electric blue.
_STRIPE_NEAR = (0.00, 0.64, 0.84)
_STRIPE_FAR = (0.00, 0.16, 1.00)
_TILT = 1.30          # how far the platelets lean toward the back (0 = flat in the skin)
# A matt surface spreads the light it receives over every direction; the
# platelets send it out in a band round the horizontal, so to a viewer beside
# the fish the stripe is brighter than paint of the same colour could be, and
# from other directions dimmer. It is still only reflected light: in shadow it
# is dark. _GAIN is that factor for the direct view; _GAIN_MIRRORED is for the
# fish's mirror images in the glass and under the surface, which look at the
# stripe from directions the platelets do not serve (without it the side of
# the fish that faces the lamp throws a pale streak onto the back glass).
_GAIN = 1.6
_GAIN_MIRRORED = 0.4


def _iridescent(nt, tilt_mask, normal=None):
    """Returns (colour, shading normal, gain) for skin whose reflecting platelets
    lean toward the fish's back by tilt_mask (0..1). Multiply the colour by the
    gain last, after any darkening."""
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    base = normal if normal is not None else geo.outputs["Normal"]
    lean = _vmath(nt, "SCALE", _dorsal(nt), scale=_math(nt, "MULTIPLY", tilt_mask, _TILT))
    tilted = _vmath(nt, "NORMALIZE", _vmath(nt, "ADD", base, lean))
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.42
    nt.links.new(tilted, lw.inputs["Normal"])
    col = _mix(nt, lw.outputs["Facing"], _STRIPE_NEAR, _STRIPE_FAR)
    lp = nt.nodes.new("ShaderNodeLightPath")
    gain = _math(nt, "MULTIPLY_ADD", lp.outputs["Is Reflection Ray"], _GAIN_MIRRORED - _GAIN, _GAIN)
    return col, tilted, gain


def tetra_materials():
    # body ------------------------------------------------------------------
    body, nt, out = new_material("TetraBody")
    N, L = nt.nodes, nt.links
    _, s, v = _uv(nt, "UVMap")
    _, n, ang = _uv(nt, "Ring")                                 # n: belly 0 .. dorsal ridge 1
    z = _math(nt, "MULTIPLY_ADD", v, 10.0, -5.0)                # mm above the snout's level

    # The stripe runs straight from the top of the eye to the adipose fin,
    # widest behind the head. dz is height above its centre line, hw its half-width.
    zc = _math(nt, "MULTIPLY_ADD", s, -0.35, 1.50)
    hw = _math(nt, "MINIMUM", _math(nt, "MULTIPLY_ADD", s, 2.0, 0.42), _math(nt, "MULTIPLY_ADD", s, -0.48, 0.90))
    dz = _math(nt, "SUBTRACT", z, zc)
    under = _math(nt, "SUBTRACT", _math(nt, "MULTIPLY", dz, -1.0), hw)      # mm below the stripe's lower edge
    over = _math(nt, "SUBTRACT", dz, hw)                                    # mm above its upper edge
    band = _smooth(nt, _math(nt, "SUBTRACT", _math(nt, "ABSOLUTE", dz), hw), -0.12, 0.10, 1.0, 0.0)
    along = _mul(nt, _smooth(nt, s, 0.050, 0.090), _smooth(nt, s, 0.775, 0.815, 1.0, 0.0))
    blue_m = _mul(nt, band, along, _smooth(nt, n, 0.86, 0.95, 1.0, 0.0))

    # dark olive back: above the stripe, along the ridge, the top of the tail
    # stalk behind the stripe's end, and the top of the snout in front of it
    below = _smooth(nt, under, -0.10, 0.12)
    tail_top = _mul(nt, _smooth(nt, dz, -0.45, -0.05), _smooth(nt, s, 0.775, 0.815))
    snout_top = _mul(nt, _smooth(nt, s, 0.045, 0.085, 1.0, 0.0), _smooth(nt, z, -0.2, 0.5))
    back_m = _max(nt, _smooth(nt, over, -0.10, 0.14), _smooth(nt, n, 0.86, 0.95), tail_top, snout_top)

    # red: below the stripe from mid-body to the tail, its front edge leaning
    # back toward the belly round the end of the body cavity; a thin pale keel
    # stays under it as far as the tail stalk
    red_front = _smooth(nt, _math(nt, "MULTIPLY_ADD", z, 0.04, s), 0.385, 0.560)
    red_keel = _math(nt, "MAXIMUM", _smooth(nt, n, 0.04, 0.28), _smooth(nt, s, 0.68, 0.82))
    red_m = _mul(nt, below, red_front, red_keel, _smooth(nt, s, 0.93, 1.0, 1.0, 0.55))

    # where the flesh is thin over the backbone, just under the stripe, the dark
    # of the spine and the kidney shows through; lower down the silver lining of
    # the body cavity hides everything
    spine_m = _mul(nt, below, _smooth(nt, under, 0.15, 1.5, 1.0, 0.0), _smooth(nt, s, 0.10, 0.22))
    flesh_m = _smooth(nt, z, -2.2, 0.3)

    # scales: one random value per scale varies the brightness, and the cell
    # edges give the faintest relief so a highlight breaks up instead of smearing
    sc = N.new("ShaderNodeCombineXYZ")
    L.new(_math(nt, "MULTIPLY", s, BODY_MM / 0.42), sc.inputs[0])
    L.new(_math(nt, "MULTIPLY", ang, 44.0), sc.inputs[1])
    vor = N.new("ShaderNodeTexVoronoi")
    vor.voronoi_dimensions = "2D"
    vor.inputs["Scale"].default_value = 1.0
    L.new(sc.outputs[0], vor.inputs["Vector"])
    sepc = N.new("ShaderNodeSeparateColor")
    L.new(vor.outputs["Color"], sepc.inputs[0])
    cell = sepc.outputs[0]
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.10
    bump.inputs["Distance"].default_value = 0.00008
    L.new(vor.outputs["Distance"], bump.inputs["Height"])

    stripe_col, normal, gain = _iridescent(nt, blue_m, bump.outputs["Normal"])

    silver = (0.76, 0.79, 0.78)
    col = _mix(nt, _math(nt, "MULTIPLY", flesh_m, 0.70), silver, (0.30, 0.34, 0.29))
    col = _mix(nt, red_m, col, (0.60, 0.012, 0.009))
    col = _mix(nt, _math(nt, "MULTIPLY", spine_m, 0.85), col, (0.040, 0.045, 0.032))
    col = _mix(nt, back_m, col, (0.075, 0.080, 0.040))
    # the rear edge of the gill cover: a faint curved line
    gill = _mul(nt, _smooth(nt, _math(nt, "ABSOLUTE", _math(nt, "MULTIPLY_ADD", _math(nt, "MULTIPLY", z, z), 0.003,
                                                           _math(nt, "SUBTRACT", s, 0.272))),
                            0.003, 0.011, 1.0, 0.0), _smooth(nt, z, -3.0, -2.2), _smooth(nt, z, 0.3, 0.9, 1.0, 0.0))
    mouth = _mul(nt, _smooth(nt, s, 0.014, 0.030, 1.0, 0.0),
                 _smooth(nt, _math(nt, "ABSOLUTE", _math(nt, "ADD", z, 0.10)), 0.05, 0.14, 1.0, 0.0))
    col = _mix(nt, _math(nt, "MAXIMUM", _math(nt, "MULTIPLY", gill, 0.30), mouth), col, (0.05, 0.035, 0.03))
    # the stripe is brightest along its middle and bluer toward the tail
    core = _smooth(nt, _math(nt, "DIVIDE", _math(nt, "ABSOLUTE", dz), hw), 0.25, 1.05, 1.0, 0.55)
    stripe_col = _mix(nt, _smooth(nt, s, 0.30, 0.80, 0.0, 0.45), stripe_col, _STRIPE_FAR)
    # and it is not even: patches of it flare and fade along the body, differently on every fish
    info = N.new("ShaderNodeObjectInfo")
    patch = N.new("ShaderNodeTexNoise")
    patch.noise_dimensions = "1D"
    patch.inputs["Scale"].default_value = 7.0
    patch.inputs["Detail"].default_value = 1.0
    L.new(_math(nt, "MULTIPLY_ADD", info.outputs["Random"], 37.0, s), patch.inputs["W"])
    stripe_col = _mix(nt, 1.0, stripe_col, _math(nt, "MULTIPLY", core, _smooth(nt, patch.outputs["Fac"], 0.25, 0.75, 0.62, 1.10)))
    N[-1].blend_type = "MULTIPLY"
    stripe_col = _vmath(nt, "SCALE", stripe_col, scale=gain)
    col = _mix(nt, blue_m, col, stripe_col)
    col = _mix(nt, 1.0, col, _math(nt, "MULTIPLY_ADD", cell, 0.14, 0.93))                 # +-7% per scale
    N[-1].blend_type = "MULTIPLY"

    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col, bsdf.inputs["Base Color"])
    L.new(normal, bsdf.inputs["Normal"])
    silver_m = _math(nt, "SUBTRACT", 1.0, _max(nt, red_m, back_m, blue_m, _math(nt, "MULTIPLY", flesh_m, 0.6)))
    # the mirror-like silver is on the flank; the underside is plain white
    metal = _math(nt, "ADD", _math(nt, "MULTIPLY", blue_m, 0.15),
                  _mul(nt, silver_m, _smooth(nt, n, 0.08, 0.40, 0.15, 0.60)))
    L.new(metal, bsdf.inputs["Metallic"])
    L.new(_math(nt, "MULTIPLY_ADD", back_m, 0.25, _math(nt, "MULTIPLY_ADD", blue_m, 0.10, 0.27)), bsdf.inputs["Roughness"])
    L.new(_math(nt, "MULTIPLY_ADD", back_m, -0.32, 0.50), bsdf.inputs["Specular IOR Level"])
    # A tetra is a small, thin, half-transparent animal: light from the lamp
    # behind it comes through the body, most of all through the tail stalk.
    L.new(_mul(nt, _math(nt, "SUBTRACT", 1.0, blue_m), _math(nt, "MULTIPLY_ADD", red_m, 0.20, 0.30)),
          bsdf.inputs["Subsurface Weight"])
    # the wet skin: a thin clear coat, for the glint along the back under the lamp
    L.new(_math(nt, "MULTIPLY_ADD", back_m, -0.17, 0.25), bsdf.inputs["Coat Weight"])
    bsdf.inputs["Coat Roughness"].default_value = 0.16
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.65, 0.50)
    bsdf.inputs["Subsurface Scale"].default_value = 0.0016
    L.new(bsdf.outputs[0], out.inputs["Surface"])

    # fins ------------------------------------------------------------------
    # Clear membrane between fine bony rays. The membrane is almost not there;
    # the rays are pale and translucent, so they show dusky against light water
    # and light up against dark, and the whole fin glints when it turns.
    fin, nt, out = new_material("TetraFin")
    N, L = nt.nodes, nt.links
    _, fu, fv = _uv(nt, "UVMap")
    caudal = _math(nt, "GREATER_THAN", fu, 1.5)
    t = _math(nt, "FRACT", _math(nt, "MULTIPLY", fu, 0.5))
    t = _math(nt, "MULTIPLY", t, 2.0)                                       # 0 base .. 1 tip, either kind of fin
    ray = _smooth(nt, _math(nt, "PINGPONG", fv, 0.5), 0.10, 0.24, 1.0, 0.0)
    lead = _mul(nt, _smooth(nt, fv, 0.15, 0.9, 1.0, 0.0), _math(nt, "SUBTRACT", 1.0, caudal))   # the stout first ray
    root = _smooth(nt, t, 0.0, 0.30, 1.0, 0.0)
    alpha = _math(nt, "MULTIPLY_ADD", ray, _math(nt, "MULTIPLY_ADD", t, -0.20, 0.55), _math(nt, "MULTIPLY_ADD", root, 0.24, 0.10))
    alpha = _math(nt, "MINIMUM", _math(nt, "MULTIPLY_ADD", lead, 0.30, alpha), 0.9)
    # the red of the tail stalk runs a little way onto the middle of the tail fin
    blush = _mul(nt, caudal, _smooth(nt, t, 0.0, 0.50, 1.0, 0.0),
                 _smooth(nt, _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", fv, 9.0)), 3.0, 8.0, 1.0, 0.0))
    alpha = _math(nt, "MAXIMUM", alpha, _math(nt, "MULTIPLY", blush, 0.70))
    fcol = _mix(nt, ray, (0.40, 0.44, 0.42), (0.44, 0.44, 0.36))
    fcol = _mix(nt, lead, fcol, (0.58, 0.60, 0.56))
    fcol = _mix(nt, blush, fcol, (0.62, 0.05, 0.03))
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(fcol, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.22
    thru = N.new("ShaderNodeBsdfTranslucent")
    L.new(fcol, thru.inputs["Color"])
    lit = N.new("ShaderNodeMixShader")
    lit.inputs[0].default_value = 0.05      # little of the lamp's light is scattered forward: most goes straight through
    L.new(bsdf.outputs[0], lit.inputs[1])
    L.new(thru.outputs[0], lit.inputs[2])
    clear = N.new("ShaderNodeBsdfTransparent")
    mixs = N.new("ShaderNodeMixShader")
    L.new(alpha, mixs.inputs[0])
    L.new(clear.outputs[0], mixs.inputs[1])
    L.new(lit.outputs[0], mixs.inputs[2])
    L.new(mixs.outputs[0], out.inputs["Surface"])

    # eye -------------------------------------------------------------------
    # Silver-gold iris; the stripe's colour crosses its upper third, as it does
    # on the living fish, and a dark rim sets the eye into the head.
    iris, nt, out = new_material("TetraIris")
    N, L = nt.nodes, nt.links
    _, eu, ev = _uv(nt, "UVMap")
    ex = _math(nt, "MULTIPLY_ADD", eu, 2.0, -1.0)
    ez = _math(nt, "MULTIPLY_ADD", ev, 2.0, -1.0)
    r = _math(nt, "SQRT", _math(nt, "ADD", _math(nt, "MULTIPLY", ex, ex), _math(nt, "MULTIPLY", ez, ez)))
    upper = _smooth(nt, ez, 0.22, 0.50)
    stripe_col, normal, gain = _iridescent(nt, upper)
    stripe_col = _vmath(nt, "SCALE", stripe_col, scale=_math(nt, "MULTIPLY", gain, 0.6))
    col = _mix(nt, _smooth(nt, ez, -0.6, 0.3), (0.64, 0.65, 0.60), (0.62, 0.56, 0.36))
    col = _mix(nt, upper, col, stripe_col)
    col = _mix(nt, _smooth(nt, r, 0.86, 1.0, 0.0, 0.75), col, (0.03, 0.035, 0.03))
    col = _mix(nt, _smooth(nt, r, 0.50, 0.60, 1.0, 0.0), col, (0.01, 0.01, 0.01))
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    L.new(col, bsdf.inputs["Base Color"])
    L.new(normal, bsdf.inputs["Normal"])
    bsdf.inputs["Metallic"].default_value = 0.55
    bsdf.inputs["Roughness"].default_value = 0.24
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.04
    L.new(bsdf.outputs[0], out.inputs["Surface"])

    pupil, nt, out = new_material("TetraPupil")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.004, 0.004, 0.005, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.30
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.04
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

    def pose(self, position, forward, phase=0.0, amp=0.5, bend=0.0, roll=0.0, grow=1.0):
        """position: metres. forward: direction of travel. phase: tail-beat phase
        in radians. amp: 0 still .. 1 hard swimming. bend: -1..1 turn (positive
        swings the tail toward local -Y, steering the head toward +Y)."""
        d = self.d
        Lm = TOTAL_MM * MM
        env = (0.018 - 0.085 * d + 0.235 * d * d) * amp
        wave = env * Lm * np.sin(phase - 2 * math.pi * d / 1.05)
        co = self.rest.copy()
        co[:, 1] += wave - bend * 0.10 * Lm * d * d
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
        self.obj.matrix_world = Matrix.Translation(Vector(position)) @ rot @ Matrix.Scale(max(self.size * grow, 1e-4), 4)


# Adult neons in a tank are 30 to 35 mm overall; the mesh is 32.6 mm at size 1.
SIZE_RANGE = (0.92, 1.06)


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
        t = Tetra(f"{name}_{i:02d}", size=rng.uniform(*SIZE_RANGE))
        t.pose((x * MM, y * MM, z * MM), fwd, phase=rng.uniform(0, 6.283), amp=rng.uniform(0.25, 0.7),
               bend=rng.gauss(0, 0.3), roll=rng.gauss(0, 0.09))
        out.append(t)
    return out

class School:
    """The film's one shoal: build once, then pose(frame) at will.

    Every scape calls `fish.School()` and gets the same fish on the same path
    (src/sim/school.py simulates the whole film once and caches it), so the
    shoal carries straight on through each change of scape. What a scape
    controls is the water the shoal may use while it is on screen: the SWIM
    literal at the top of its module.
    """

    def __init__(self, name="Tetra"):
        import random
        from sim import school as sim
        self.data = sim.film()
        self.frame_start = int(self.data["frame_start"])
        self.frames = self.data["pos"].shape[0]
        rng = random.Random(sim.SEED)
        # A real shoal is never one size: a couple of half-grown fish, most near
        # full size, one or two big ones (24 to 36 mm). Fixed set, shuffled once.
        sizes = [0.74, 0.80, 0.86, 0.90, 0.93, 0.96, 0.98, 1.0, 1.02, 1.04, 1.07, 1.10]
        rng.shuffle(sizes)
        self.fish = [Tetra(f"{name}_{i:02d}", size=sizes[i % len(sizes)]) for i in range(self.data["pos"].shape[1])]

    @property
    def objects(self):
        return [t.obj for t in self.fish]

    def pose(self, frame, grow=None):
        """grow: optional function index -> 0..1 size, for animals that appear over time."""
        f = min(max(frame - self.frame_start, 0), self.frames - 1)
        d = self.data
        for i, t in enumerate(self.fish):
            t.pose(tuple(float(c) * MM for c in d["pos"][f, i]), tuple(float(c) for c in d["fwd"][f, i]),
                   phase=float(d["phase"][f, i]), amp=float(d["amp"][f, i]),
                   bend=float(d["bend"][f, i]), roll=float(d["roll"][f, i]),
                   grow=1.0 if grow is None else grow(i))

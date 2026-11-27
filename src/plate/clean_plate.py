"""Room plate (part of task T1.3).

The lit plate with the tank and the lamp painted out, so it can be projected
onto the rebuilt room (floor, wall, tray) and seen correctly behind and through
a rendered tank. The wall, skirting and floor are all horizontal structures, so
the hole is filled row by row between the image margins either side of the tank,
after shearing out the camera's slight roll.

Writes to <work>/plate/:
  room_plate.png           16-bit display-referred sRGB
  room_plate_linear.exr    scene-linear (see src/colour.py), for Blender
  tank_matte.png           tank + lamp silhouette at the solved camera
  room_plate_preview.jpg

Run:  python src/plate/clean_plate.py
"""
import json
import os
import sys
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import colour  # noqa: E402
import paths  # noqa: E402
from camera import Camera, box_corners  # noqa: E402

MARGIN_L = (2, 58)        # columns sampled left of the tank
MARGIN_R = (1018, 1078)   # columns sampled right of the tank
FEATHER = 5
GLOW_REMOVED = 0.65   # share of the tank's glow taken off the wall behind it


def removal_mask(cam, room):
    """Everything that the rendered tank and lamp will replace."""
    tray_top = room["floor_z"] + room["tray"]["height"]
    tank = cam.hull_mask(box_corners((0, 0, tray_top), (300, 300, 300)), dilate=5)
    h = room["lamp"]["head"]
    z0, z1 = h["underside_z"], h["underside_z"] + h["thickness"]
    head_pts = [(p[0], p[1], z) for p in (h["rear_left"], h["rear_right"], h["front_left"], h["front_right"]) for z in (z0, z1)]
    head = cam.hull_mask(head_pts, dilate=14)            # generous: the lit head has a halo
    s = room["lamp"]["stem"]
    stem = cam.hull_mask(box_corners((s["x"] - s["width"], s["y"] - 5, 290), (s["x"] + s["width"], s["y"] + 5, s["z_top"])), dilate=6)
    c = room["lamp"]["clamp"]
    clamp = cam.hull_mask(box_corners((c["x_min"], 285, c["z_min"]), (c["x_max"], 315, c["z_max"])), dilate=6)
    return cv2.max(cv2.max(tank, head), cv2.max(stem, clamp))


def fill_rows(img, mask, slope, untrusted):
    """Replace masked pixels by interpolating each (de-rolled) row between its margins.
    `untrusted` marks pixels near the hole that are not bare wall or floor."""
    H, W = mask.shape
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    map_x = np.tile(xs, (H, 1))
    shift = slope * (xs - W / 2)
    fwd_y = ys[:, None] + shift[None, :]          # sheared image samples source at y + slope*x
    sheared = cv2.remap(img, map_x, fwd_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sheared_mask = cv2.remap(mask, map_x, fwd_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    def margin(cols):
        band = sheared[:, cols[0]:cols[1]]
        ok = sheared_mask[:, cols[0]:cols[1]] == 0
        band = np.where(ok[..., None], band, np.nan)
        return np.nanmedian(band, axis=1)             # median ignores the thin cable

    left, right = margin(MARGIN_L), margin(MARGIN_R)
    for m in (left, right):
        assert not np.isnan(m).any(), "a margin row is fully masked; widen the margins"
    # smooth a little down the rows so the fill is not streaky
    left = cv2.GaussianBlur(left[:, None, :], (1, 5), 0)[:, 0, :]
    right = cv2.GaussianBlur(right[:, None, :], (1, 5), 0)[:, 0, :]
    xl, xr = np.mean(MARGIN_L), np.mean(MARGIN_R)
    w = np.clip((xs - xl) / (xr - xl), 0, 1)[None, :, None]
    fill = left[:, None, :] * (1 - w) + right[:, None, :] * w
    back_y = ys[:, None] - shift[None, :]
    fill = cv2.remap(fill.astype(np.float32), map_x, back_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # The margins are darker than the wall right next to the lit tank (glow), so the
    # raw fill would show as a rectangle. Carry the mismatch seen on the rim of the
    # hole smoothly across it (a membrane fill of the difference), which keeps the
    # horizontal structure from the row fill and makes the edge seamless.
    # Only bare wall and floor on the rim are trusted: not the tray, not the cable.
    ring = (cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0) & (cv2.dilate(mask, np.ones((5, 5), np.uint8)) == 0)
    ring &= untrusted == 0
    diff = (img - fill) * ring[..., None]
    wgt = ring.astype(np.float32)
    corr = np.zeros_like(img)
    done = np.zeros((H, W), bool)
    for sigma in (8, 16, 32, 64, 128, 256):
        num = cv2.GaussianBlur(diff, (0, 0), sigma)
        den = cv2.GaussianBlur(wgt, (0, 0), sigma)
        take = (~done) & (den > 0.02)
        corr[take] = num[take] / den[take][:, None]
        done |= take
    corr = cv2.GaussianBlur(corr, (0, 0), 10)
    # The rim is brightened by glow from the lit tank (in the lens as much as on
    # the wall). Honour it fully at the edge so the join is invisible, but ease
    # most of it away deeper in, where the wall will be seen through the water.
    depth = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    ease = np.clip(depth / 110.0, 0, 1)
    ease = ease * ease * (3 - 2 * ease)
    corr *= (1.0 - GLOW_REMOVED * ease)[..., None]
    fill = fill + corr

    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), FEATHER)[..., None]
    alpha = np.clip(alpha * 1.6, 0, 1)
    return img * (1 - alpha) + fill * alpha


def main():
    paths.ensure_dirs()
    cam = Camera()
    room = json.loads((paths.CONFIG / "room.json").read_text(encoding="utf-8"))
    plate = cv2.imread(str(paths.PLATE / "lit_plate.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0

    mask = removal_mask(cam, room)
    wl, wr = room["wall"]["floor_junction_left"], room["wall"]["floor_junction_right"]
    a, b = cam.project([wl, wr])
    slope = float((b[1] - a[1]) / (b[0] - a[0]))
    print(f"roll slope {slope:.4f} ({np.degrees(np.arctan(slope)):.2f} deg); mask covers {mask.mean() / 255 * 100:.1f}% of the frame")

    # Not bare wall or floor: the tray and everything below its rim, and the cable
    # where it leaves the tank on the left.
    tr = room["tray"]
    tray_top = room["floor_z"] + tr["height"]
    untrusted = cam.hull_mask(box_corners((tr["x_min"] - 6, tr["front_y"] - 6, room["floor_z"]),
                                          (tr["x_max"] + 6, 320, tray_top + 4)), dilate=10)
    rim_row = int(cam.project([(150, 0, tray_top)])[0][1]) - 40
    untrusted[rim_row:, :] = 255
    cv2.rectangle(untrusted, (0, 960), (80, 1070), 255, -1)   # cable

    room_plate = np.clip(fill_rows(plate, mask, slope, untrusted), 0, 1)
    cv2.imwrite(str(paths.PLATE / "room_plate.png"), (room_plate * 65535 + 0.5).astype(np.uint16))
    cv2.imwrite(str(paths.PLATE / "tank_matte.png"), mask)
    cv2.imwrite(str(paths.PLATE / "room_plate_preview.jpg"), (room_plate * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])

    linear = colour.display_to_linear(room_plate.astype(np.float64)).astype(np.float32)
    ok = cv2.imwrite(str(paths.PLATE / "room_plate_linear.exr"), linear,
                     [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
    assert ok, "EXR write failed"
    lin_lit = colour.display_to_linear(plate.astype(np.float64)).astype(np.float32)
    cv2.imwrite(str(paths.PLATE / "lit_plate_linear.exr"), lin_lit, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
    back = colour.linear_to_display(cv2.imread(str(paths.PLATE / "room_plate_linear.exr"), cv2.IMREAD_UNCHANGED).astype(np.float64))
    err = float(np.abs(back - room_plate)[room_plate.max(axis=2) < 0.995].max())
    print(f"EXR round-trip max error {err:.4f} (display units)")
    assert err < 0.004, "linear EXR does not round-trip"
    print("OK")


if __name__ == "__main__":
    main()

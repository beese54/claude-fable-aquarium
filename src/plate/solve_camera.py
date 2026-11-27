"""Camera solve (task T1.2).

Solves the phone camera's pose and focal length from features of the 30 cm tank
read off the unlit plate (config/camera_picks.json). Point features contribute
reprojection error; line features contribute point-to-projected-line distance,
which lets edges whose ends are hidden (behind the tray) still constrain the fit.

Writes config/camera.json (OpenCV and Blender forms) and
review/01_camera_overlay.jpg (tank wireframe over the plate).

Run:  python src/plate/solve_camera.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

W, H = 1080, 1920
MAX_RMS_PX = 2.0


def project(params, pts_mm, cx=W / 2, cy=H / 2):
    rvec, tvec, f = params[0:3], params[3:6], params[6]
    R, _ = cv2.Rodrigues(rvec)
    cam = (R @ np.asarray(pts_mm, dtype=np.float64).T).T + tvec
    return np.stack([f * cam[:, 0] / cam[:, 2] + cx, f * cam[:, 1] / cam[:, 2] + cy], axis=1)


def residuals(params, picks):
    res = []
    p3 = np.array([p["mm"] for p in picks["points"]], dtype=np.float64)
    p2 = np.array([p["px"] for p in picks["points"]], dtype=np.float64)
    res.extend((project(params, p3) - p2).ravel())
    for ln in picks["lines"]:
        a, b = project(params, [ln["a_mm"], ln["b_mm"]])
        d = b - a
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        res.append(float(n @ (np.array(ln["px"]) - a)))
    return np.array(res)


def initial_guess(picks):
    p3 = np.array([p["mm"] for p in picks["points"]], dtype=np.float64)
    p2 = np.array([p["px"] for p in picks["points"]], dtype=np.float64)
    best = None
    for f in np.linspace(900, 2600, 18):
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(p3, p2, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        x0 = np.concatenate([rvec.ravel(), tvec.ravel(), [f]])
        cost = float(np.sum(residuals(x0, picks) ** 2))
        if best is None or cost < best[0]:
            best = (cost, x0)
    return best[1]


def tank_wireframe(t=5.0, s=300.0):
    """Outer box edges and inner (water side) box edges, in mm."""
    def box(x0, y0, z0, x1, y1, z1):
        c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        return [(c[a], c[b]) for a, b in e]
    return box(0, 0, 0, s, s, s), box(t, t, t, s - t, s - t, s)


def draw_overlay(params, picks, plate_path, out_path):
    img = cv2.imread(str(plate_path), cv2.IMREAD_UNCHANGED)
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    S = 2  # supersample for clean thin lines
    big = cv2.resize(img, (W * S, H * S), interpolation=cv2.INTER_CUBIC)
    outer, inner = tank_wireframe(picks["glass_mm"], picks["tank_mm"][0])
    for edges, col in ((outer, (0, 0, 255)), (inner, (255, 160, 0))):
        for a, b in edges:
            pa, pb = project(params, [a, b]) * S
            cv2.line(big, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), col, 1, cv2.LINE_AA)
    for p in picks["points"]:
        q = np.round(np.array(p["px"]) * S).astype(int)
        cv2.circle(big, tuple(q), 7, (0, 255, 0), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    picks = json.loads((paths.CONFIG / "camera_picks.json").read_text(encoding="utf-8"))
    x0 = initial_guess(picks)
    sol = least_squares(residuals, x0, args=(picks,), method="lm", xtol=1e-12, ftol=1e-12)
    p = sol.x
    r = residuals(p, picks)
    n_pt = 2 * len(picks["points"])
    pt_err = np.linalg.norm(r[:n_pt].reshape(-1, 2), axis=1)
    rms_pts = float(np.sqrt(np.mean(pt_err ** 2)))
    rms_lines = float(np.sqrt(np.mean(r[n_pt:] ** 2)))
    rms_all = float(np.sqrt((np.sum(pt_err ** 2) + np.sum(r[n_pt:] ** 2)) / (len(pt_err) + len(r[n_pt:]))))

    R, _ = cv2.Rodrigues(p[0:3])
    t = p[3:6]
    f = float(p[6])
    cam_pos_mm = -R.T @ t
    # Blender camera looks down -Z with +Y up; OpenCV looks down +Z with +Y down.
    R_blender = R.T @ np.diag([1.0, -1.0, -1.0])
    M = np.eye(4)
    M[:3, :3] = R_blender
    M[:3, 3] = cam_pos_mm / 1000.0  # metres

    for nm, e in zip([q["name"] for q in picks["points"]], pt_err):
        print(f"  {e:5.2f} px  {nm}")
    for ln, e in zip(picks["lines"], r[n_pt:]):
        print(f"  {e:+5.2f} px  line: {ln['name']}")
    print(f"focal {f:.1f} px  (vertical FOV {np.degrees(2 * np.arctan(H / 2 / f)):.2f} deg)")
    print(f"camera at {np.round(cam_pos_mm, 1)} mm in tank frame; {-cam_pos_mm[1]:.0f} mm in front of the glass, "
          f"{cam_pos_mm[2]:.0f} mm above the tank base")
    print(f"RMS points {rms_pts:.2f} px, lines {rms_lines:.2f} px, all {rms_all:.2f} px")

    out = {
        "image_size": [W, H],
        "focal_px": round(f, 3),
        "principal_point": [W / 2, H / 2],
        "vertical_fov_deg": round(float(np.degrees(2 * np.arctan(H / 2 / f))), 4),
        "opencv": {"rvec": [round(float(v), 8) for v in p[0:3]], "tvec_mm": [round(float(v), 4) for v in t]},
        "blender": {
            "units": "metres, tank frame: X right, Y away from camera, Z up, origin at outer front-left-bottom corner",
            "matrix_world": [[round(float(v), 8) for v in row] for row in M],
            "sensor_fit": "VERTICAL", "sensor_height_mm": 36.0,
            "lens_mm": round(f * 36.0 / H, 5),
        },
        "camera_position_mm": [round(float(v), 2) for v in cam_pos_mm],
        "fit": {"rms_points_px": round(rms_pts, 3), "rms_lines_px": round(rms_lines, 3), "rms_all_px": round(rms_all, 3),
                "max_point_px": round(float(pt_err.max()), 3)},
    }
    (paths.CONFIG / "camera.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    draw_overlay(p, picks, paths.PLATE / "unlit_plate.png", paths.REVIEW / "01_camera_overlay.jpg")
    draw_overlay(p, picks, paths.PLATE / "lit_plate.png", paths.REVIEW / "01_camera_overlay_lit.jpg")
    if rms_all > MAX_RMS_PX:
        sys.exit(f"FAIL: RMS {rms_all:.2f} px exceeds {MAX_RMS_PX} px")
    print("OK")


if __name__ == "__main__":
    main()

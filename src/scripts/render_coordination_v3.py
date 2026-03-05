"""
Render coordination decision scene v3 with planner-verified configs.

Three robots sharing the same base:
  Robot 1 (solid) → throws to B1 (red box, forward-right)
  Robot 2 (solid) → throws to B2 (blue box, forward-left), closest to R1
  Robot 3 (ghost, blue tint, alpha=0.2) → throws to B2, farthest from R1

Ball trajectories:
  Set 1: R1 EEF → B1, solid blue spheres
  Set 2: R2 EEF → B2, solid blue spheres
  Set 3: R3 EEF → B2, ghost blue spheres (alpha=0.2)

Multi-pass composited onto white background.
Output: figures/model/coordination_decision_v3.png
"""

import os
import sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

# ── Paths ────────────────────────────────────────────────────────────
ROBOT_XML = os.path.join(current_dir, '../description/iiwa7_allegro_throwing.xml')
OUTPUT_DIR = os.path.join(current_dir, '../../figures/model')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Resolution (render at 2x, downscale to 1600x1200) ───────────────
RENDER_W, RENDER_H = 3200, 2400
FINAL_W, FINAL_H = 1600, 1200

# ── Planner-verified robot configurations (iiwa7 joints) ────────────
Q_R1 = np.array([-1.4578, 0.6056, 1.8329, -1.1944, -0.5671, 0.9056, -3.0891])
Q_R2 = np.array([-2.5141, -1.4944, 1.8329, 0.9056, 1.5329, 0.6056, -2.5432])
Q_R3 = np.array([1.1678, 1.2056, -2.6671, 0.9056, -2.3671, -2.0944, 0.7691])

# Hand open (post-release) — from config.yaml hand_home_pose
HAND_OPEN = np.array([
    0.03063398, -0.15441399, 0.82104033, 0.82512679, 0.00664029,
    -0.2039733, 0.80409743, 0.0, -0.10271061, -0.06070025,
    0.90650362, 0.81377007, 0.86300686, 0.44554398, 0.08874171,
    0.77574977
])

# ── EEF positions (world frame) from planner ────────────────────────
KUKA_BASE_Z = 0.5  # table height
# AE (relative to kuka_base) from planner output
EEF_R1 = np.array([0.5567, -0.2028, KUKA_BASE_Z + 0.6458])
EEF_R2 = np.array([0.3851, 0.8081, KUKA_BASE_Z + 0.2900])
EEF_R3 = np.array([0.1500, 1.2500, KUKA_BASE_Z + 0.4200])  # 偏左且更远离 S1/S2

# ── Targets (render positions, z=0.01 for visibility) ───────────────
B1_POS = np.array([1.0, -0.6, 0.01])     # forward-right, red box
B2_POS = np.array([0.5, 1.1, 0.01])      # forward-left, blue box
B2_GHOST_POS = np.array([0.20, 1.35, 0.01])  # 半透明方案的蓝箱：更偏左、更远

# ── Camera ───────────────────────────────────────────────────────────
CAMERA_CFG = {
    'azimuth': 135,
    'elevation': -55,
    'distance': 4.0,
}

# ── Geoms to hide for clean render ──────────────────────────────────
HIDDEN_GEOMS = [
    'x_axis', 'y_axis', 'z_axis',
    'vel_arrow', 'vel_arrow_head',
    'floor', 'kuka_table',
    'sphere1_geom', 'sphere2_geom', 'sphere3_geom',
]

# ── Ball parameters ──────────────────────────────────────────────────
BALL_RADIUS = 0.04
N_BALLS = 5
BALL_B2_RGBA = np.array([0.3, 0.5, 1.0, 1.0])      # blue balls → blue box B2
BALL_B1_RGBA = np.array([0.9, 0.25, 0.2, 1.0])      # red balls → red box B1
BALL_GHOST_RGBA = np.array([0.3, 0.5, 1.0, 0.8])    # (will be ghosted in composite)
FLIGHT_TIME = 0.55
GRAVITY = 9.81

# ── Ghost rendering parameters ──────────────────────────────────────
GHOST_TINT_STRENGTH = 0.0    # 0 = keep original colors, no tint
GHOST_ALPHA = 0.35            # final opacity on white canvas


def hide_geoms(model, names):
    for name in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_rgba[gid, 3] = 0.0


def set_scene(model, data, q_iiwa, hand_pose, box_configs):
    """Set robot config, hand pose, box positions. Hide spheres."""
    data.qpos[:7] = q_iiwa
    data.qpos[7:23] = hand_pose
    for name, pos in box_configs:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        j = model.body_jntadr[bid]
        data.qpos[j:j + 3] = pos
    for name in ['sphere1', 'sphere2', 'sphere3']:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        j = model.body_jntadr[bid]
        data.qpos[j:j + 3] = [0, 0, -5]
    mujoco.mj_forward(model, data)


def ballistic_pos(start, target, t, T=FLIGHT_TIME):
    dx = target - start
    vx, vy = dx[0] / T, dx[1] / T
    vz = (dx[2] + 0.5 * GRAVITY * T**2) / T
    return np.array([
        start[0] + vx * t,
        start[1] + vy * t,
        start[2] + vz * t - 0.5 * GRAVITY * t**2,
    ])


def add_trajectory_balls(scene, start, target, rgba, n=N_BALLS, radius=BALL_RADIUS):
    """Add n+1 spheres: n evenly spaced along arc + 1 landing ball."""
    t_vals = np.linspace(0, FLIGHT_TIME, n + 2)[1:-1]  # n intermediate
    t_vals = np.append(t_vals, FLIGHT_TIME)              # + landing
    identity = np.eye(3).flatten()
    for t in t_vals:
        pos = ballistic_pos(start, target, t)
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0, 0]), pos, identity, rgba,
        )
        scene.ngeom += 1


def make_camera(cfg, lookat):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = cfg['azimuth']
    camera.elevation = cfg['elevation']
    camera.distance = cfg['distance']
    camera.lookat[:] = lookat
    return camera


def composite_layers(bg, solid_imgs, ghost_img):
    """Composite ghost (back) + solids (front) onto white canvas."""
    result = np.full_like(bg, 255, dtype=np.float32)

    # Ghost layer (same colors, just low alpha — no tint)
    diff = np.abs(ghost_img.astype(np.int16) - bg.astype(np.int16))
    mask = diff.max(axis=2) > 3
    if np.any(mask):
        ghost_pixels = ghost_img[mask].astype(np.float32)
        result[mask] = result[mask] * (1 - GHOST_ALPHA) + ghost_pixels * GHOST_ALPHA

    # Solid layers (in order, on top)
    for solid in solid_imgs:
        diff = np.abs(solid.astype(np.int16) - bg.astype(np.int16))
        mask = diff.max(axis=2) > 3
        if np.any(mask):
            result[mask] = solid[mask].astype(np.float32)

    return np.clip(result, 0, 255).astype(np.uint8)


def auto_crop_4_3(pixels):
    """Auto-crop to content with padding, enforce 4:3 aspect ratio."""
    h, w = pixels.shape[:2]
    gray = np.min(pixels, axis=2)
    mask = gray < 250
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return pixels, (0, 0)  # no content

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    # 8% padding
    pad_r = int(0.08 * (r_max - r_min))
    pad_c = int(0.08 * (c_max - c_min))
    r_min = max(0, r_min - pad_r)
    r_max = min(h - 1, r_max + pad_r)
    c_min = max(0, c_min - pad_c)
    c_max = min(w - 1, c_max + pad_c)

    # Enforce 4:3
    crop_h = r_max - r_min + 1
    crop_w = c_max - c_min + 1
    target_ratio = 4.0 / 3.0

    if crop_w / crop_h < target_ratio:
        new_w = int(crop_h * target_ratio)
        expand = new_w - crop_w
        c_min = max(0, c_min - expand // 2)
        c_max = min(w - 1, c_min + new_w - 1)
        if c_max >= w:
            c_max = w - 1
            c_min = max(0, c_max - new_w + 1)
    else:
        new_h = int(crop_w / target_ratio)
        expand = new_h - crop_h
        r_min = max(0, r_min - expand // 2)
        r_max = min(h - 1, r_min + new_h - 1)
        if r_max >= h:
            r_max = h - 1
            r_min = max(0, r_max - new_h + 1)

    return pixels[r_min:r_max + 1, c_min:c_max + 1], (c_min, r_min)


def world_to_pixel(points_3d, camera_cfg, lookat, render_w, render_h,
                   crop_origin, crop_size, final_w, final_h, fovy=45.0):
    """Project 3D world points to pixel coordinates in the final image."""
    az = np.radians(camera_cfg['azimuth'])
    el = np.radians(camera_cfg['elevation'])
    d = camera_cfg['distance']

    # Camera position in world frame
    cam_pos = lookat + d * np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        -np.sin(el)
    ])

    # Camera basis vectors
    forward = (lookat - cam_pos)
    forward = forward / np.linalg.norm(forward)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)   # +y in image = down

    # View matrix (world → camera)
    R = np.array([right, down, forward])
    t = R @ (-cam_pos)

    # Focal length from FOV
    f = render_h / (2 * np.tan(np.radians(fovy) / 2))
    cx, cy = render_w / 2.0, render_h / 2.0

    crop_x0, crop_y0 = crop_origin
    crop_w, crop_h = crop_size
    scale_x = final_w / crop_w
    scale_y = final_h / crop_h

    results = {}
    for name, pt in points_3d.items():
        p_cam = R @ pt + t
        if p_cam[2] <= 0:
            results[name] = (None, None)
            continue
        px_render = f * p_cam[0] / p_cam[2] + cx
        py_render = f * p_cam[1] / p_cam[2] + cy
        # Apply crop
        px_crop = px_render - crop_x0
        py_crop = py_render - crop_y0
        # Apply resize
        px_final = int(px_crop * scale_x)
        py_final = int(py_crop * scale_y)
        results[name] = (px_final, py_final)

    return results


def main():
    print(f"Loading model: {ROBOT_XML}")
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    model.vis.global_.offwidth = RENDER_W
    model.vis.global_.offheight = RENDER_H
    hide_geoms(model, HIDDEN_GEOMS)

    data = mujoco.MjData(model)

    # Camera lookat = centroid of (robot base, B1, B2)
    robot_base = np.array([0.0, 0.0, KUKA_BASE_Z])
    lookat = (robot_base + B1_POS + B2_POS) / 3.0
    camera = make_camera(CAMERA_CFG, lookat)
    print(f"  Camera lookat: {lookat}")
    print(f"  EEF R1: {EEF_R1}")
    print(f"  EEF R2: {EEF_R2}")
    print(f"  EEF R3: {EEF_R3}")

    renderer = mujoco.Renderer(model, RENDER_H, RENDER_W)

    # ── Pass 0: Background ───────────────────────────────────────────
    print("Pass 0: background")
    orig_alpha = model.geom_rgba[:, 3].copy()
    model.geom_rgba[:, 3] = 0.0
    renderer.update_scene(data, camera)
    bg = renderer.render().copy()
    model.geom_rgba[:, 3] = orig_alpha

    # ── Pass 1: Robot 1 + boxes + Set 1 trajectory ───────────────────
    # MuJoCo colors: box1=BLUE, box2=RED, box3=YELLOW
    print("Pass 1: Robot 1 (solid) + boxes + trajectory set 1")
    set_scene(model, data, Q_R1, HAND_OPEN, [
        ('box2', B1_POS), ('box1', B2_POS)
    ])
    renderer.update_scene(data, camera)
    add_trajectory_balls(renderer._scene, EEF_R1, B1_POS, BALL_B1_RGBA)
    r1_img = renderer.render().copy()

    # ── Pass 2: Robot 2 + Set 2 trajectory ───────────────────────────
    print("Pass 2: Robot 2 (solid) + trajectory set 2")
    set_scene(model, data, Q_R2, HAND_OPEN, [
        ('box1', [0, 0, -100]), ('box2', [0, 0, -100])
    ])
    renderer.update_scene(data, camera)
    add_trajectory_balls(renderer._scene, EEF_R2, B2_POS, BALL_B2_RGBA)
    r2_img = renderer.render().copy()

    # ── Pass 3: Robot 3 (ghost) + Set 3 trajectory ───────────────────
    print("Pass 3: Robot 3 (ghost) + trajectory set 3")
    # 在幽灵层中显示其目标蓝箱（偏左远离 S1/S2），以体现“另一个可行但较远的方案”
    set_scene(model, data, Q_R3, HAND_OPEN, [
        ('box1', B2_GHOST_POS),  # 蓝箱（ghost层）
        ('box2', [0, 0, -100])   # 红箱隐藏
    ])
    renderer.update_scene(data, camera)
    # 轨迹落点与幽灵蓝箱一致，保证“落点合理”
    add_trajectory_balls(renderer._scene, EEF_R3, B2_GHOST_POS, BALL_GHOST_RGBA)
    r3_img = renderer.render().copy()

    # ── Composite ────────────────────────────────────────────────────
    print("Compositing: ghost → solid R1 → solid R2")
    result = composite_layers(bg, [r1_img, r2_img], r3_img)

    # ── Auto-crop (4:3) and resize to 1600x1200 ─────────────────────
    cropped, crop_origin = auto_crop_4_3(result)
    crop_h, crop_w = cropped.shape[:2]
    print(f"  Crop origin: {crop_origin}, crop size: {crop_w}x{crop_h}")

    img = Image.fromarray(cropped)
    img = img.resize((FINAL_W, FINAL_H), Image.LANCZOS)

    # ── 计算像素坐标并叠加标注（S1/S2/S3） ───────────────────────────
    points = {
        'Robot 1 EEF': EEF_R1,
        'Robot 2 EEF': EEF_R2,
        'Robot 3 EEF': EEF_R3,
        'B1 center':   B1_POS,
        'B2 center':   B2_POS,
        'Robot base':  robot_base,
    }

    pixel_coords = world_to_pixel(
        points, CAMERA_CFG, lookat,
        RENDER_W, RENDER_H,
        crop_origin, (crop_w, crop_h),
        FINAL_W, FINAL_H
    )

    draw = ImageDraw.Draw(img)

    def _label(name, text, color=(20, 20, 20)):
        pxpy = pixel_coords.get(name)
        if pxpy and pxpy[0] is not None:
            x, y = pxpy
            draw.text((x + 8, y - 14), text, fill=color)

    # 标签：表达三种方案
    _label('Robot 1 EEF', 'S1: 红球→红箱')
    _label('Robot 2 EEF', 'S2: 蓝球→蓝箱（靠近S1）')
    _label('Robot 3 EEF', 'S3: 蓝球→蓝箱（较远，半透明）')

    # 用连线可视化 S2 与 S3 相对 S1 的“接近/远离”
    r1 = pixel_coords.get('Robot 1 EEF')
    r2 = pixel_coords.get('Robot 2 EEF')
    r3 = pixel_coords.get('Robot 3 EEF')
    if r1 and r2 and r1[0] is not None and r2[0] is not None:
        draw.line([r1, r2], fill=(40, 160, 40), width=3)     # 绿色：更近
    if r1 and r3 and r1[0] is not None and r3[0] is not None:
        draw.line([r1, r3], fill=(200, 140, 40), width=3)    # 橙色：更远

    draw.text((24, 24), 'Multi-Throw Coordination: 两者均有效，但选择更近的 S2', fill=(10, 10, 10))

    out_path = os.path.join(OUTPUT_DIR, 'coordination_decision_v3.png')
    img.save(out_path)
    print(f"  Saved: {out_path}")

    # ── Print pixel coordinates for PPT overlay ──────────────────────
    print("\n" + "=" * 60)
    print("  PIXEL COORDINATES (in 1600x1200 final image)")
    print("=" * 60)

    # 已在保存前计算了 pixel_coords，这里直接使用其结果打印

    for name, (px, py) in pixel_coords.items():
        if px is not None:
            print(f"  {name:20s}: ({px:4d}, {py:4d})")
        else:
            print(f"  {name:20s}: (behind camera)")

    del renderer
    print("\nDone.")


if __name__ == '__main__':
    main()

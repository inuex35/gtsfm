"""Browser-based 3D visualization of SfM results using Plotly.

Works on WSL2 and headless environments without X server.
Reads camera_poses.txt and point_cloud.ply from the output directory.

Usage:
    python -m gtsfm.visualization.view_scene_browser --output_dir ./output

Authors: Auto-generated for spherical SfM
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go


def read_camera_poses(filepath: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Read camera poses from camera_poses.txt.

    Returns:
        Dict mapping camera index to (translation, quaternion_wxyz).
    """
    cameras = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            idx = int(parts[0])
            t = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            cameras[idx] = (t, q)
    return cameras


def read_ply(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read ASCII PLY point cloud.

    Returns:
        points: (N, 3) array of xyz coordinates.
        colors: (N, 3) array of RGB values [0-255].
    """
    points = []
    colors = []
    in_data = False
    with open(filepath) as f:
        for line in f:
            if in_data:
                parts = line.strip().split()
                if len(parts) >= 6:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
            elif line.strip() == "end_header":
                in_data = True
    return np.array(points), np.array(colors)


def quat_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
    ])


def make_camera_traces(
    cameras: Dict[int, Tuple[np.ndarray, np.ndarray]],
    axis_len: float = 0.05,
) -> List[go.Scatter3d]:
    """Create Plotly traces for camera frustums.

    Each camera is drawn as 3 axis lines (R/G/B for X/Y/Z) from the camera center.
    """
    positions = np.array([t for t, _ in cameras.values()])

    # Camera positions
    traces = [
        go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="markers",
            marker=dict(size=4, color="red"),
            name="Cameras",
            hovertext=[f"cam {i}" for i in cameras],
        )
    ]

    # Camera axes
    axis_colors = ["red", "green", "blue"]
    axis_names = ["X", "Y", "Z"]
    for ax_idx in range(3):
        xs, ys, zs = [], [], []
        for _, (t, q) in cameras.items():
            R = quat_to_rotation(q[0], q[1], q[2], q[3])
            axis = R[:, ax_idx] * axis_len
            end = t + axis
            xs.extend([t[0], end[0], None])
            ys.extend([t[1], end[1], None])
            zs.extend([t[2], end[2], None])
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=axis_colors[ax_idx], width=2),
            name=f"Cam {axis_names[ax_idx]}-axis",
            showlegend=False,
        ))

    # Camera path (connect cameras in order)
    sorted_indices = sorted(cameras.keys())
    path_pts = np.array([cameras[i][0] for i in sorted_indices])
    traces.append(go.Scatter3d(
        x=path_pts[:, 0], y=path_pts[:, 1], z=path_pts[:, 2],
        mode="lines",
        line=dict(color="rgba(255,0,0,0.3)", width=1),
        name="Camera path",
        showlegend=False,
    ))

    return traces


def subsample_points(
    points: np.ndarray, colors: np.ndarray, max_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly subsample points if exceeding max_points."""
    if len(points) <= max_points:
        return points, colors
    indices = np.random.choice(len(points), max_points, replace=False)
    return points[indices], colors[indices]


def filter_outliers(
    points: np.ndarray, colors: np.ndarray, percentile: float = 95
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outlier points beyond the given percentile of distance from median."""
    median = np.median(points, axis=0)
    dists = np.linalg.norm(points - median, axis=1)
    threshold = np.percentile(dists, percentile)
    mask = dists < threshold
    return points[mask], colors[mask]


def view_scene_browser(args: argparse.Namespace) -> None:
    """Create and display an interactive 3D scene in the browser."""
    output_dir = args.output_dir

    # Read data
    poses_file = os.path.join(output_dir, "camera_poses.txt")
    ply_file = os.path.join(output_dir, "point_cloud.ply")

    if not os.path.exists(poses_file):
        raise FileNotFoundError(f"Camera poses not found: {poses_file}")

    cameras = read_camera_poses(poses_file)
    print(f"Loaded {len(cameras)} cameras")

    fig = go.Figure()

    # Point cloud
    if os.path.exists(ply_file):
        points, colors = read_ply(ply_file)
        print(f"Loaded {len(points)} points")

        points, colors = filter_outliers(points, colors, args.percentile)
        points, colors = subsample_points(points, colors, args.max_points)
        print(f"Displaying {len(points)} points (after filtering)")

        color_strings = [f"rgb({r},{g},{b})" for r, g, b in colors]
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers",
            marker=dict(size=args.point_size, color=color_strings, opacity=0.8),
            name=f"Points ({len(points)})",
        ))
    else:
        print(f"No point cloud found at {ply_file}, showing cameras only")

    # Cameras
    cam_traces = make_camera_traces(cameras, axis_len=args.axis_len)
    for trace in cam_traces:
        fig.add_trace(trace)

    # Layout
    fig.update_layout(
        title=f"SfM Result — {len(cameras)} cameras",
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Save and open
    html_path = os.path.join(output_dir, "scene.html")
    fig.write_html(html_path)
    print(f"Saved to {html_path}")

    if not args.no_open:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(html_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SfM results in the browser using Plotly.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory with camera_poses.txt and point_cloud.ply")
    parser.add_argument("--max_points", type=int, default=500000, help="Max points to display (subsampled if exceeded)")
    parser.add_argument("--point_size", type=float, default=1.5, help="Point size in the viewer")
    parser.add_argument("--axis_len", type=float, default=0.01, help="Camera axis length")
    parser.add_argument("--percentile", type=float, default=95, help="Outlier filter percentile")
    parser.add_argument("--no_open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()
    view_scene_browser(args)

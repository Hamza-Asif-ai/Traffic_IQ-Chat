"""
calibration_helper.py
Draws a labeled pixel grid over the first frame of your video, so you can
accurately read off the pixel distance between two known real-world points
(e.g. lane markings) instead of guessing -- this is what most people get
wrong, causing unrealistic speed estimates.

Usage:
    python calibration_helper.py --video sample_data/traffic_sample.mp4 --output calibration_grid.png

Then open calibration_grid.png, find two points whose real-world distance
you know, read their pixel coordinates off the grid, and compute:
    pixel_distance = distance between the two points (use the grid to measure)
    meters_per_pixel = real_distance_m / pixel_distance

Good reference distances to use:
    - Lane width: ~3.5-3.7m on most highways/motorways
    - UK-style dashed lane markings: each painted dash + gap repeats every 6m
    - A vehicle's own length: an average car is ~4.5m long (use a car already
      in the frame as a ruler if you don't trust the road markings)
"""

import argparse
import cv2


def save_calibration_grid(video_path: str, output_path: str, grid_spacing: int = 50):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise FileNotFoundError(f"Could not read a frame from: {video_path}")

    h, w = frame.shape[:2]
    grid_color = (0, 255, 255)
    text_color = (0, 0, 255)

    # Vertical grid lines + x-axis labels
    for x in range(0, w, grid_spacing):
        cv2.line(frame, (x, 0), (x, h), grid_color, 1)
        cv2.putText(frame, str(x), (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)

    # Horizontal grid lines + y-axis labels
    for y in range(0, h, grid_spacing):
        cv2.line(frame, (0, y), (w, y), grid_color, 1)
        cv2.putText(frame, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)

    cv2.imwrite(output_path, frame)
    print(f"Calibration grid saved to: {output_path}")
    print(f"Frame size: {w}x{h}, grid spacing: {grid_spacing}px")
    print("\nHow to use it:")
    print("1. Open the saved image.")
    print("2. Find two points with a known real-world distance (e.g. lane width).")
    print("3. Read their approximate (x, y) pixel coordinates from the grid.")
    print("4. pixel_distance = straight-line distance between those two points")
    print("   (use the Pythagorean theorem if they're not aligned on one axis).")
    print("5. Plug real_distance_m and pixel_distance into the app/script/notebook.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a pixel-grid overlay for speed calibration.")
    parser.add_argument("--video", required=True, help="Path to your video file")
    parser.add_argument("--output", default="calibration_grid.png", help="Where to save the grid image")
    parser.add_argument("--grid_spacing", type=int, default=50, help="Pixel spacing between grid lines")
    args = parser.parse_args()

    save_calibration_grid(args.video, args.output, args.grid_spacing)

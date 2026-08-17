"""
Regenerates oschersleben_map_wide.png/.yaml and oschersleben_centerline_wide.csv
from oschersleben_centerline.csv, at a chosen track half-width.

The original map's boundary lines are ~2px thin, so they can't be grown via
morphological dilation (it just erases them). Instead this redraws fresh
boundary lines offset from the centerline at HALF_WIDTH_M on each side.

Change HALF_WIDTH_M below and rerun. Known safe range: the tightest gap
between two different (non-adjacent) parts of this track is ~4.19m
center-to-center, so keep HALF_WIDTH_M comfortably under 2.0m (2.0m each side
= 4.0m total, leaving only ~0.19m of buffer) or the boundaries will start
overlapping and merge separate track passages. 1.6m was the validated value
used for config_oschersleben_wide.yaml.
"""
import os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import sys

# examples/suport/generate_wide_map.py -> examples/ -> examples/maps/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.dirname(SCRIPT_DIR)
MAPS_DIR = os.path.join(EXAMPLES_DIR, 'maps')

sys.path.insert(0, EXAMPLES_DIR)
from waypoint_follow import compute_track_boundaries

HALF_WIDTH_M = 1.5  # <-- change this (meters, each side of centerline)

# original map metadata (must match oschersleben_map.yaml)
RESOLUTION = 0.04295
ORIGIN_X, ORIGIN_Y = -55.07650228661655, -33.57884064395765


def main():
    orig_img = Image.open(os.path.join(MAPS_DIR, 'oschersleben_map.png'))
    w, h = orig_img.size

    centerline = np.loadtxt(os.path.join(MAPS_DIR, 'oschersleben_centerline.csv'), delimiter=',', skiprows=1)
    xy = centerline[:, 0:2]

    w_right_new = np.full(len(xy), HALF_WIDTH_M)
    w_left_new = np.full(len(xy), HALF_WIDTH_M)
    left_b, right_b = compute_track_boundaries(xy, w_right_new, w_left_new)

    def world_to_px(pts):
        px = (pts[:, 0] - ORIGIN_X) / RESOLUTION
        row = h - 1 - (pts[:, 1] - ORIGIN_Y) / RESOLUTION
        return np.stack([px, row], axis=1)

    left_px = world_to_px(left_b)
    right_px = world_to_px(right_b)

    canvas = Image.new('L', (w, h), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.line([tuple(p) for p in left_px] + [tuple(left_px[0])], fill=0, width=2, joint='curve')
    draw.line([tuple(p) for p in right_px] + [tuple(right_px[0])], fill=0, width=2, joint='curve')
    wide_map_path = os.path.join(MAPS_DIR, 'oschersleben_map_wide.png')
    canvas.save(wide_map_path)

    with open(os.path.join(MAPS_DIR, 'oschersleben_map_wide.yaml'), 'w') as f:
        f.write(f'image: oschersleben_map_wide.png\n')
        f.write(f'resolution: {RESOLUTION}\n')
        f.write(f'origin: [{ORIGIN_X},{ORIGIN_Y}, 0.000000]\n')
        f.write('negate: 0\n')
        f.write('occupied_thresh: 0.45\n')
        f.write('free_thresh: 0.196\n')

    out = np.column_stack([xy, w_right_new, w_left_new])
    np.savetxt(os.path.join(MAPS_DIR, 'oschersleben_centerline_wide.csv'), out, delimiter=',',
               header='x_m, y_m, w_tr_right_m, w_tr_left_m', comments='# ')

    print(f'Generated oschersleben_map_wide.png/.yaml and oschersleben_centerline_wide.csv '
          f'at half_width={HALF_WIDTH_M}m ({HALF_WIDTH_M * 2}m total width)')

    # sanity check: verify topology wasn't broken (should still be 3 free-space
    # components: outside / track corridor / interior island; and known-good
    # points should stay in the same corridor component)
    arr = np.array(Image.open(wide_map_path))
    free = arr >= 128
    labeled, num = ndimage.label(free)
    print(f'free-space components: {num} (expect 3 -- if this changed, HALF_WIDTH_M is too large '
          f'and two track passages merged)')


if __name__ == '__main__':
    main()

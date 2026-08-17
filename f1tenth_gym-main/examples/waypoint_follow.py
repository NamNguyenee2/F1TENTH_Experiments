import json
import os
import shutil
import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace

from numba import njit

from pyglet.gl import GL_POINTS

import matplotlib
matplotlib.use('Agg')  # headless: no display/GL context needed (SSH-safe)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# folder layout (examples/waypoint_follow.py, examples/maps/, examples/results/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(BASE_DIR, 'maps')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

"""
Planner Helpers
"""
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        #points = self.waypoints

        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]
        
    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle


class FlippyPlanner:
    """
    Planner designed to exploit integration methods and dynamics.
    For testing only. To observe this error, use single track dynamics for all velocities >0.1
    """
    def __init__(self, speed=1, flip_every=1, steer=2):
        self.speed = speed
        self.flip_every = flip_every
        self.counter = 0
        self.steer = steer
    
    def render_waypoints(self, *args, **kwargs):
        pass

    def plan(self, *args, **kwargs):
        if self.counter%self.flip_every == 0:
            self.counter = 0
            self.steer *= -1
        return self.speed, self.steer


def compute_track_boundaries(centerline_xy, w_right, w_left):
    """
    Offsets a closed centerline polyline perpendicular to its local tangent by
    the given per-point right/left track widths, returning (left, right) boundary
    polylines. w_right/w_left are 1D arrays, one value per centerline point.
    """
    tangents = np.empty_like(centerline_xy)
    tangents[1:-1] = centerline_xy[2:] - centerline_xy[:-2]
    tangents[0] = centerline_xy[1] - centerline_xy[-1]
    tangents[-1] = centerline_xy[0] - centerline_xy[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents /= norms

    left_normal = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    left_bound = centerline_xy + left_normal * w_left[:, None]
    right_bound = centerline_xy - left_normal * w_right[:, None]
    return left_bound, right_bound


def render_log_to_video(log, waypoints_xy, video_path, timestep, stride=1, playback_speed=1.0,
                         left_bound=None, right_bound=None):
    """
    Renders a logged run (x, y, yaw per step) to an MP4 by animating the car
    over the waypoint track. Headless (Agg backend) so it works over SSH.
    """
    xs = log['x'][::stride]
    ys = log['y'][::stride]
    yaws = log['yaw'][::stride]

    fig, ax = plt.subplots(figsize=(8, 8))
    if left_bound is not None and right_bound is not None:
        ax.plot(left_bound[:, 0], left_bound[:, 1], '-', color='black', linewidth=1, zorder=1)
        ax.plot(right_bound[:, 0], right_bound[:, 1], '-', color='black', linewidth=1, zorder=1)
    ax.plot(waypoints_xy[:, 0], waypoints_xy[:, 1], '--', color='lightgray', linewidth=1, zorder=1)
    ax.set_aspect('equal')
    margin = 5.0
    ax.set_xlim(waypoints_xy[:, 0].min() - margin, waypoints_xy[:, 0].max() + margin)
    ax.set_ylim(waypoints_xy[:, 1].min() - margin, waypoints_xy[:, 1].max() + margin)
    ax.set_title('Pure Pursuit')

    trail, = ax.plot([], [], '-', color='red', linewidth=1, alpha=0.5, zorder=2)
    car_dot, = ax.plot([], [], 'o', color='red', markersize=8, zorder=3)
    heading_line, = ax.plot([], [], '-', color='blue', linewidth=2, zorder=4)

    def update(i):
        x, y, yaw = xs[i], ys[i], yaws[i]
        trail.set_data(xs[:i + 1], ys[:i + 1])
        car_dot.set_data([x], [y])
        heading_line.set_data([x, x + 1.5 * np.cos(yaw)], [y, y + 1.5 * np.sin(yaw)])
        return trail, car_dot, heading_line

    anim = FuncAnimation(fig, update, frames=len(xs), blit=True)
    fps = max(1.0, playback_speed / (timestep * stride))

    if shutil.which('ffmpeg') is not None:
        anim.save(video_path, writer='ffmpeg', fps=fps)
    else:
        gif_path = os.path.splitext(video_path)[0] + '.gif'
        print('ffmpeg not found on PATH; saving a GIF instead (install ffmpeg, or `pip install '
              'imageio-ffmpeg`, for real MP4 output).')
        anim.save(gif_path, writer='pillow', fps=fps)
        video_path = gif_path

    plt.close(fig)
    return video_path


def main():
    """
    main entry point
    """

    # tuned for oschersleben_map_wide (track boundaries redrawn at 1.6m/side instead of the
    # original 1.1m/side -- at the original width, no tlad/vgain combo we tried could clear
    # the ~3m-radius hairpin around s=32-38m without running off-track).
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 2.0, 'vgain': 0.7}

    enable_render = False  # live pyglet window needs a display/GL context -- unsafe over SSH
    save_video = True      # render the driven trajectory to an MP4 after the run instead
    video_stride = 20       # keep every Nth logged step (speeds up rendering, NOT playback length)
    video_playback_speed = 1.0  # >1 = shorter/faster video, <1 = slow motion
    num_laps = 3           # env's own `done` hardcodes a 2-lap stop, so we track laps ourselves

    with open(os.path.join(MAPS_DIR, 'config_oschersleben_wide.yaml')) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # config files store bare/relative filenames (e.g. './oschersleben_map_wide');
    # resolve them against maps/ regardless of where this script is run from
    conf.map_path = os.path.join(MAPS_DIR, os.path.basename(conf.map_path))
    conf.wpt_path = os.path.join(MAPS_DIR, os.path.basename(conf.wpt_path))
    if hasattr(conf, 'centerline_path'):
        conf.centerline_path = os.path.join(MAPS_DIR, os.path.basename(conf.centerline_path))

    planner = PurePursuitPlanner(conf, (0.17145+0.15875)) #FlippyPlanner(speed=0.2, flip_every=1, steer=10)

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_waypoints(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    env.add_render_callback(render_callback)
    
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    if enable_render:
        env.render()

    laptime = 0.0
    start = time.time()

    log = {'x': [], 'y': [], 'yaw': [], 'lap_counts': [], 'collisions': [], 'speed': [], 'steer': []}

    while not obs['collisions'][0] and obs['lap_counts'][0] < num_laps:
    # while obs['lap_counts'][0] < num_laps:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward

        log['x'].append(obs['poses_x'][0])
        log['y'].append(obs['poses_y'][0])
        log['yaw'].append(obs['poses_theta'][0])
        log['lap_counts'].append(obs['lap_counts'][0])
        log['collisions'].append(obs['collisions'][0])
        log['speed'].append(speed)
        log['steer'].append(steer)


        if enable_render:
            env.render(mode='human')

    if obs['collisions'][0]:
        print('Collided after %.1f laps' % obs['lap_counts'][0])
    else:
        print('Completed %d laps' % num_laps)
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if save_video:
        for key in log:
            log[key] = np.array(log[key])
        waypoints_xy = np.vstack((planner.waypoints[:, conf.wpt_xind], planner.waypoints[:, conf.wpt_yind])).T

        left_bound = right_bound = None
        if hasattr(conf, 'centerline_path'):
            centerline = np.loadtxt(conf.centerline_path, delimiter=conf.centerline_delim,
                                     skiprows=conf.centerline_rowskip)
            left_bound, right_bound = compute_track_boundaries(
                centerline[:, 0:2], w_right=centerline[:, 2], w_left=centerline[:, 3])

        video_path = render_log_to_video(log, waypoints_xy,
                                          os.path.join(RESULTS_DIR, conf.run_name + '_video.mp4'),
                                          timestep=env.timestep, stride=video_stride,
                                          playback_speed=video_playback_speed,
                                          left_bound=left_bound, right_bound=right_bound)
        print('Video saved to', video_path)
    json_path = os.path.join(RESULTS_DIR, f'map_{conf.run_name}_lap{num_laps}.json')
    with open(json_path, 'w') as f:
        json.dump({key: np.asarray(val).tolist() for key, val in log.items()}, f)
if __name__ == '__main__':
    main()

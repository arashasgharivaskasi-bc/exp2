"""run_frontiers_in_gym.py
Exploration code using paths provided by bc_exploration and simulation environment provided by bc_gym_planning_env
"""
from __future__ import print_function, absolute_import, division

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from bc_exploration.algorithms.frontier_based_exploration import create_frontier_agent_from_params
from bc_exploration.mapping.costmap import Costmap as exp_CostMap
from bc_exploration.utilities.util import xy_to_rc, which_coords_in_bounds
from bc_exploration.mapping.log_odds_mapper import LogOddsMapper
from bc_exploration.utilities.paths import get_maps_dir, get_exploration_dir

from bc_gym_planning_env.utilities.pixel_lidar import VirtualLidar
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D as gym_CostMap
from bc_gym_planning_env.envs.base.params import EnvParams
from bc_gym_planning_env.envs.base.env import PlanEnv


def to_brain_costmap(exp_costmap):
    """
    Convert current exp_costmap to a CostMap2D object that corresponds to brain's values.
    Note the origin is just the resolution scaled of the current costmap
    :param exp_costmap Costmap: map used by bc_exploration
    :return CostMap2D: brain formatted costmap
    """
    costmap_data = exp_costmap.data.copy()
    costmap_data[exp_costmap.data == exp_CostMap.FREE] = gym_CostMap.FREE_SPACE
    costmap_data[exp_costmap.data == exp_CostMap.OCCUPIED] = gym_CostMap.LETHAL_OBSTACLE
    costmap_data[exp_costmap.data == exp_CostMap.UNEXPLORED] = gym_CostMap.NO_INFORMATION

    return gym_CostMap(data=np.flipud(costmap_data), resolution=exp_costmap.resolution,
                       origin=exp_costmap.origin.astype(np.float64))

def run_frontiers_in_gym(map_filename, params_filename, start_state, sensor_range,
                         map_resolution, render=True, render_interval=10,
                         max_exploration_iterations=None):
    """
    Interface for running frontier exploration in the bc gym environment that is initialized via map_filename.. etc
    :param map_filename str: path of the map to load into the grid world environment, needs to be a uint8 png with
                         values 127 for unexplored, 255 for free, 0 for occupied.
    :param params_filename str: path of the params file for setting up the frontier agent etc.
                            See bc_exploration/params/ for examples
    :param start_state array(3)[float]: starting state of the robot in the map (in meters) [x, y, theta],
                        if None the starting state is random
    :param sensor_range float: range of the sensor (LiDAR) in meters
    :param map_resolution float: resolution of the map desired
    :param render bool: whether or not to visualize
    :param render_interval int: visualize every render_interval iterations
    :param max_exploration_iterations int: number of exploration cycles to run before exiting
    :return Costmap: occupancy_map, final map from exploration, percentage explored, time taken to explore
    """

    # some parameters
    frontier_agent = create_frontier_agent_from_params(params_filename)
    footprint = frontier_agent.get_footprint()

    # setup costmap
    map = exp_CostMap(data=cv2.imread(map_filename),
                      resolution=map_resolution,
                      origin=np.array([0., 0.]))
    map_data = to_brain_costmap(map).get_data()
    costmap = gym_CostMap(data=map_data[:, :, 0],
                          resolution=map_resolution,
                          origin=np.array([0., 0.]))

    padding = 0.
    map_shape = np.array(map_data.shape[:2]) + int(2. * padding // map_resolution)
    exp_initial_map = exp_CostMap(data=exp_CostMap.UNEXPLORED * np.ones(map_shape, dtype=np.uint8),
                                  resolution=map_resolution,
                                  origin=[-padding - 0., -padding - 0.])

    footprint_coords = footprint.get_ego_points(start_state[2], map_resolution) + start_state[:2]
    footprint_coords = xy_to_rc(footprint_coords, exp_initial_map).astype(np.int)
    footprint_coords = footprint_coords[which_coords_in_bounds(footprint_coords, exp_initial_map.get_shape())]
    exp_initial_map.data[footprint_coords[:, 0], footprint_coords[:, 1]] = exp_CostMap.FREE

    gym_initial_map = to_brain_costmap(exp_initial_map)

    # pick a sensor
    sensor = VirtualLidar(range_max=sensor_range,
                          range_angular=250 * np.pi / 180,
                          costmap=costmap,
                          resolution_angular=1.0 * np.pi / 180)

    # define log-odds mapper
    mapper = LogOddsMapper(initial_map=exp_initial_map,
                           sensor_range=sensor.get_range_max(),
                           measurement_certainty=0.8,
                           max_log_odd=8,
                           min_log_odd=-8,
                           threshold_occupied=.5,
                           threshold_free=.5)

    # define planning environment parameters
    env_params = EnvParams(
        iteration_timeout=1200,
        pose_delay=0,
        control_delay=0,
        state_delay=0,
        goal_spat_dist=1.0,
        goal_ang_dist=np.pi / 2,
        dt=0.05,  # 20 Hz
        path_limiter_max_dist=5.0,
        sensor=sensor
    )

    env = PlanEnv(
        costmap=gym_initial_map,
        path=start_state[None,:],
        params=env_params
    )

    obs = env.reset()

    if render:
        env.render()

    pose = obs.pose
    scan_angles, scan_ranges = obs.get_lidar_scan()
    occupancy_map = mapper.update(state=pose, scan_angles=scan_angles, scan_ranges=scan_ranges)

    iteration = 0
    is_last_plan = False
    was_successful = True
    j = 0
    while True:
        if max_exploration_iterations is not None and iteration > max_exploration_iterations:
            was_successful = False
            is_last_plan = True

        # using the current map, make an action plan
        path = frontier_agent.plan(state=pose, occupancy_map=occupancy_map, is_last_plan=is_last_plan)

        # if we get empty lists for policy/path, that means that the agent was
        # not able to find a path/frontier to plan to.
        if not path.shape[0]:
            print("No more frontiers! Either the map is 100% explored, or bad start state, or there is a bug!")
            break

        # if we have a policy, follow it until the end. update the map sparsely (speeds it up)
        env = PlanEnv(
            costmap=to_brain_costmap(occupancy_map),
            path=path,
            params=env_params
        )

        path = list(path)
        done = False
        while len(path) != 0 or not done:
            desired_pose = path.pop(0)

            if footprint.check_for_collision(desired_pose, occupancy_map, unexplored_is_occupied=True):
                footprint_coords = footprint.get_ego_points(desired_pose[2], map_resolution) + desired_pose[:2]
                footprint_coords = xy_to_rc(footprint_coords, occupancy_map).astype(np.int)
                footprint_coords = footprint_coords[which_coords_in_bounds(footprint_coords, occupancy_map.get_shape())]
                occupancy_map.data[footprint_coords[:, 0], footprint_coords[:, 1]] = exp_CostMap.FREE

            obs, reward, done, info = env.simple_step(desired_pose)
            pose = obs.pose
            scan_angles, scan_ranges = obs.get_lidar_scan()
            occupancy_map = mapper.update(state=pose, scan_angles=scan_angles, scan_ranges=scan_ranges)

            # put the current laserscan on the map before planning
            occupied_coords, _ = sensor.get_scan_points(pose)
            occupied_coords = xy_to_rc(occupied_coords, occupancy_map).astype(np.int)
            occupied_coords = occupied_coords[which_coords_in_bounds(occupied_coords, occupancy_map.get_shape())]
            occupancy_map.data[occupied_coords[:, 0], occupied_coords[:, 1]] = exp_CostMap.OCCUPIED

            if render and j % render_interval == 0:
                state = env.get_state()
                state.costmap = to_brain_costmap(occupancy_map)
                env.set_state(state)
                env.render()

            j += 1

        if is_last_plan:
            break

        iteration += 1

    if render:
        cv2.waitKey(0)

    return occupancy_map, iteration, was_successful

def main():
    """
    Main Function
    """
    np.random.seed(3)
    occupancy_map, iterations_taken, _ = \
        run_frontiers_in_gym(map_filename=os.path.join(get_maps_dir(), "brain/vw_ground_truth_full_edited.png"),
                             params_filename=os.path.join(get_exploration_dir(), "params/params.yaml"),
                             map_resolution=0.03,
                             start_state=np.array([15., 10., 0.]),
                             sensor_range=10.0,
                             max_exploration_iterations=25,
                             render_interval=10)

    print("This is " + str(iterations_taken) + " iterations!")
    plt.figure()
    plt.imshow(occupancy_map.data.astype(np.uint8), cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
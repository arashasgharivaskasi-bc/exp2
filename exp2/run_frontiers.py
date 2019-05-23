"""run_frontiers.py
Interface for integrating bc_exploration functions in bc_gym_planning_env
Contains code for exploration and rendering in gym environment
"""
from __future__ import print_function, absolute_import, division

import os
import cv2
import numpy as np

# from bc_gym_planning_env.envs.mini_env import MiniEnv, RandomMiniEnv, AisleTurnEnv
from bc_gym_planning_env.envs.synth_turn_env import RandomAisleTurnEnv,\
    ColoredCostmapRandomAisleTurnEnv,\
    ColoredEgoCostmapRandomAisleTurnEnv
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D


from algorithms.frontier_based_exploration import create_frontier_agent_from_params, visualize
from envs.grid_world import GridWorld
from mapping.costmap import Costmap
from mapping.log_odds_mapper import LogOddsMapper
from sensors.sensors import Lidar
from utilities.paths import get_maps_dir, get_exploration_dir
from utilities.util import xy_to_rc, which_coords_in_bounds, scan_to_points


def create_gym_environment_from_parameters(env_mode):
    """
    Creates a random gym environment and stores it in the "maps" directory
    :param env_mode str: indicates the type of environment in which we want to run the exploration algorithm
    :return map_filename string: map filename
    """
    if env_mode == "MiniEnv":
        raise NotImplementedError
        # occupancy_grid = MiniEnv().get_state().costmap.get_data()
    elif env_mode == "RandomMiniEnv":
        raise NotImplementedError
        # occupancy_grid = RandomMiniEnv().get_state().costmap.get_data()
    elif env_mode == "AisleTurnEnv":
        raise NotImplementedError
        # occupancy_grid = AisleTurnEnv().get_state().costmap.get_data()
    elif env_mode == "RandomAisleTurnEnv":
        occupancy_grid = RandomAisleTurnEnv().get_state().costmap.get_data()
    elif env_mode == "ColoredCostmapRandomAisleTurnEnv":
        occupancy_grid = ColoredCostmapRandomAisleTurnEnv().get_state().costmap.get_data()
    elif env_mode == "ColoredEgoCostmapRandomAisleTurnEnv":
        occupancy_grid = ColoredEgoCostmapRandomAisleTurnEnv().get_state().costmap.get_data()
    elif env_mode == "ComplexCostmapRandomAisleTurnEnv":
        complexity = np.random.randint(2, 5)
        map_bundle = np.zeros([500, 500, complexity])
        for m in range(complexity):
            map_bundle[:, :, m] = \
                cv2.resize(ColoredCostmapRandomAisleTurnEnv().get_state().costmap.get_data(), (500, 500))
        occupancy_grid = np.sum(map_bundle, axis=2)
    else:
        raise NameError("Invalid gym enviroment type!\nPlease choose one of the following gym environments:"
                        "\nMiniEnv \\ RandomMiniEnv \\ AisleTurnEnv \\ "
                        "RandomAisleTurnEnv \\ ColoredCostmapRandomAisleTurnEnv \\ "
                        "ColoredEgoCostmapRandomAisleTurnEnv \\ ComplexCostmapRandomAisleTurnEnv")

    occupancy_grid[occupancy_grid != 0] = 255
    occupancy_grid = -1 * occupancy_grid + 255

    path = os.path.join(get_maps_dir(), 'gym')
    if not os.path.exists(path):
        os.mkdir(path)

    map_filename = os.path.join(path, 'random_gym_map.png')
    cv2.imwrite(map_filename, occupancy_grid)

    return map_filename


def visualize_in_gym(gym_env, exp_map, path, pose):
    """
    Alternative visualization method using gym rendering function
    :param gym_env RandomAisleTurnEnv: object storing the robot and environment state in bc_gym_planning_env standard
    :param exp_map Costmap: occupancy map in bc_exploration standard
    :param path array(N, 3)[float]: containing an array of poses for the robot to follow
    :param pose array(3)[float]: corresponds to the pose of the robot [x, y, theta]
    """

    gym_compat_map = np.zeros(exp_map.data.shape, dtype=np.uint8)
    gym_compat_map[exp_map.data == Costmap.OCCUPIED] = CostMap2D.LETHAL_OBSTACLE
    gym_compat_map[exp_map.data == Costmap.FREE] = CostMap2D.FREE_SPACE
    gym_compat_map[exp_map.data == Costmap.UNEXPLORED] = CostMap2D.NO_INFORMATION

    gym_costmap = CostMap2D(np.flipud(gym_compat_map), exp_map.resolution, exp_map.origin.copy())

    gym_state = gym_env.get_state()

    if np.array(path).shape[0]:
        gym_state.path = path
        gym_state.original_path = path
    else:
        gym_state.path = pose[None, :]
        gym_state.original_path = pose[None, :]

    gym_state.costmap = gym_costmap

    gym_state.pose = pose

    gym_state.robot_state.x = pose[0]
    gym_state.robot_state.y = pose[1]
    gym_state.robot_state.angle = pose[2]

    gym_env.set_state(gym_state)

    gym_env.render(mode='human')


def run_frontier_exploration(map_filename, params_filename, start_state, sensor_range, map_resolution,
                             completion_percentage, render=True, render_mode='exp', render_interval=1, render_size_scale=1.7,
                             completion_check_interval=1, render_wait_for_key=True, max_exploration_iterations=None):
    """
    Interface for running frontier exploration on the grid world environment that is initialized via map_filename.. etc
    :param map_filename str: path of the map to load into the grid world environment, needs to be a uint8 png with
                         values 127 for unexplored, 255 for free, 0 for occupied.
    :param params_filename str: path of the params file for setting up the frontier agent etc.
                            See exploration/params/ for examples
    :param start_state array(3)[float]: starting state of the robot in the map (in meters) [x, y, theta], if None the starting
                        state is random
    :param sensor_range float: range of the sensor (lidar) in meters
    :param map_resolution float: resolution of the map desired
    :param completion_percentage float: decimal of the completion percentage desired i.e (.97), how much of the ground
                                  truth environment to explore, note that 1.0 is not always reachable because of
                                  footprint.
    :param render bool: whether or not to visualize
    :param render_mode str: specify rendering function (bc_exploration or bc_gym_planning_env)
    :param render_interval int: visualize every render_interval iterations
    :param render_size_scale Tuple(int): (h, w), the size of the render window in pixels
    :param completion_check_interval int: check for exploration completion every completion_check_interval iterations
    :param render_wait_for_key bool: if render is enabled, if render_wait_for_key is True then the exploration algorithm
                                will wait for key press to start exploration. Timing is not affected.
    :param max_exploration_iterations int: number of exploration cycles to run before exiting
    :return Costmap: occupancy_map, final map from exploration), percentage explored, time taken to explore
    """

    # some parameters
    frontier_agent = create_frontier_agent_from_params(params_filename)
    footprint = frontier_agent.get_footprint()

    # pick a sensor
    sensor = Lidar(sensor_range=sensor_range,
                   angular_range=250 * np.pi / 180,
                   angular_resolution=1.0 * np.pi / 180,
                   map_resolution=map_resolution)

    # setup grid world environment
    env = GridWorld(map_filename=map_filename,
                    map_resolution=map_resolution,
                    sensor=sensor,
                    footprint=footprint,
                    start_state=start_state)

    # setup corresponding gym environment
    env_instance = RandomAisleTurnEnv()

    render_size = (np.array(env.get_map_shape()[::-1]) * render_size_scale).astype(np.int)

    # setup log-odds mapper, we assume the measurements are very accurate,
    # thus one scan should be enough to fill the map
    padding = 1.
    map_shape = np.array(env.get_map_shape()) + int(2. * padding // map_resolution)
    initial_map = Costmap(data=Costmap.UNEXPLORED * np.ones(map_shape, dtype=np.uint8),
                          resolution=env.get_map_resolution(),
                          origin=[-padding - env.start_state[0], -padding - env.start_state[1]])

    clearing_footprint_points = footprint.get_clearing_points(map_resolution)
    clearing_footprint_coords = xy_to_rc(clearing_footprint_points, initial_map).astype(np.int)
    initial_map.data[clearing_footprint_coords[:, 0], clearing_footprint_coords[:, 1]] = Costmap.FREE

    mapper = LogOddsMapper(initial_map=initial_map,
                           sensor_range=sensor.get_sensor_range(),
                           measurement_certainty=0.8,
                           max_log_odd=8,
                           min_log_odd=-8,
                           threshold_occupied=.5,
                           threshold_free=.5)

    # reset the environment to the start state, map the first observations
    pose, [scan_angles, scan_ranges] = env.reset()

    occupancy_map = mapper.update(state=pose, scan_angles=scan_angles, scan_ranges=scan_ranges)

    if render:
        if render_mode == 'exp':
            visualize(occupancy_map=occupancy_map, state=pose, footprint=footprint,
                      start_state=start_state, scan_angles=scan_angles, scan_ranges=scan_ranges, path=[],
                      render_size=render_size, frontiers=[], wait_key=0 if render_wait_for_key else 1)
        elif render_mode == 'gym':
            visualize_in_gym(gym_env=env_instance, exp_map=occupancy_map, path=[], pose=pose)
        else:
            raise NameError("Invalid rendering method.\nPlease choose one of \"exp\" or \"gym\" methods.")

    iteration = 0
    is_last_plan = False
    was_successful = True
    percentage_explored = 0.0
    while True:
        if iteration % completion_check_interval == 0:
            percentage_explored = env.compare_maps(occupancy_map)
            if percentage_explored >= completion_percentage:
                is_last_plan = True

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
        for j, desired_state in enumerate(path):
            if footprint.check_for_collision(desired_state, occupancy_map, unexplored_is_occupied=True):
                footprint_coords = footprint.get_ego_points(desired_state[2], map_resolution) + desired_state[:2]
                footprint_coords = xy_to_rc(footprint_coords, occupancy_map).astype(np.int)
                footprint_coords = footprint_coords[which_coords_in_bounds(footprint_coords, occupancy_map.get_shape())]
                occupancy_map.data[footprint_coords[:, 0], footprint_coords[:, 1]] = Costmap.FREE

            pose, [scan_angles, scan_ranges] = env.step(desired_state)
            occupancy_map = mapper.update(state=pose, scan_angles=scan_angles, scan_ranges=scan_ranges)

            # put the current laserscan on the map before planning
            occupied_coords = scan_to_points(scan_angles + pose[2], scan_ranges) + pose[:2]
            occupied_coords = xy_to_rc(occupied_coords, occupancy_map).astype(np.int)
            occupied_coords = occupied_coords[which_coords_in_bounds(occupied_coords, occupancy_map.get_shape())]
            occupancy_map.data[occupied_coords[:, 0], occupied_coords[:, 1]] = Costmap.OCCUPIED

            # shows a live visualization of the exploration process if render is set to true
            if render and j % render_interval == 0:
                if render_mode == 'exp':
                    visualize(occupancy_map=occupancy_map, state=pose, footprint=footprint,
                              start_state=start_state, scan_angles=scan_angles, scan_ranges=scan_ranges,
                              path=path, render_size=render_size,
                              frontiers=frontier_agent.get_frontiers(compute=True, occupancy_map=occupancy_map),
                              wait_key=1,
                              path_idx=j)
                elif render_mode == 'gym':
                    visualize_in_gym(gym_env=env_instance, exp_map=occupancy_map, path=path, pose=pose)
                else:
                    raise NameError("Invalid rendering method.\nPlease choose one of \"exp\" or \"gym\" methods.")

        if is_last_plan:
            break

        iteration += 1

    if render:
        cv2.waitKey(0)

    return occupancy_map, percentage_explored, iteration, was_successful


def main():
    """
    Main Function
    """
    np.random.seed(3)
    _, percent_explored, iterations_taken, _ = \
        run_frontier_exploration(map_filename=create_gym_environment_from_parameters("RandomAisleTurnEnv"),
                                 params_filename=os.path.join(get_exploration_dir(), "params/params.yaml"),
                                 map_resolution=0.03,
                                 start_state=None,
                                 sensor_range=10.0,
                                 completion_percentage=10,
                                 max_exploration_iterations=None,
                                 render_mode='gym',
                                 render_size_scale=50.0,
                                 render_interval=5)

    print("Map", "{:.2f}".format(percent_explored * 100), "\b% explored!",
          "This is " + str(iterations_taken) + " iterations!")


if __name__ == '__main__':
    main()

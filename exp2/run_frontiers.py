from __future__ import print_function, absolute_import, division

import os
import cv2
import numpy as np

from bc_gym_planning_env.envs.mini_env import MiniEnv, RandomMiniEnv
from bc_gym_planning_env.envs.synth_turn_env import AisleTurnEnv, RandomAisleTurnEnv, ColoredCostmapRandomAisleTurnEnv, ColoredEgoCostmapRandomAisleTurnEnv

from algorithms.frontier_based_exploration import run_frontier_exploration

# from agents.frontier_agent import FrontierAgent
# from footprints.footprint_points import get_tricky_circular_footprint, get_tricky_oval_footprint, get_jackal_footprint
# from footprints.footprints import CustomFootprint
# from sensors.sensors import Lidar
# from envs.grid_world import GridWorld
# from mapping.costmap import Costmap
from utilities.paths import get_maps_dir, get_exploration_dir

def create_gym_environment_from_parameters(env_mode,env_params=None):
    """
    Creates a random gym environment and stores it in the "maps" directory
    :param env_mode: indicates the type of environment in which we want to run the exploration algorithm
    :return map_filename: map filename
    """
    if env_mode == "MiniEnv":
        occupancy_grid = MiniEnv().get_state().costmap.get_data()
    elif env_mode == "RandomMiniEnv":
        occupancy_grid = RandomMiniEnv().get_state().costmap.get_data()
    elif env_mode == "AisleTurnEnv":
        occupancy_grid = AisleTurnEnv().get_state().costmap.get_data()
    elif env_mode == "RandomAisleTurnEnv":
        occupancy_grid = RandomAisleTurnEnv().get_state().costmap.get_data()
    elif env_mode == "ColoredCostmapRandomAisleTurnEnv":
        occupancy_grid = ColoredCostmapRandomAisleTurnEnv().get_state().costmap.get_data()
    elif env_mode == "ColoredEgoCostmapRandomAisleTurnEnv":
        occupancy_grid = ColoredEgoCostmapRandomAisleTurnEnv().get_state().costmap.get_data()
    elif env_mode == "ComplexCostmapRandomAisleTurnEnv":
        complexity = np.random.randint(2,5)
        map_bundle = np.zeros([500,500,complexity])
        for m in range(complexity):
            map_bundle[:,:,m] = cv2.resize(ColoredCostmapRandomAisleTurnEnv().get_state().costmap.get_data(),(500,500))
        occupancy_grid = np.sum(map_bundle,axis = 2)
    else:
        raise NameError("Invalid gym envirment type!\nPlease choose one of the following gym environments:"
            "\nMiniEnv \\ RandomMiniEnv \\ AisleTurnEnv \\ "
            "RandomAisleTurnEnv \\ ColoredCostmapRandomAisleTurnEnv \\ "
            "ColoredEgoCostmapRandomAisleTurnEnv \\ ComplexCostmapRandomAisleTurnEnv")

    occupancy_grid[occupancy_grid != 0] = 255
    occupancy_grid = -1 * occupancy_grid + 255

    # map_filename = os.path.join(get_maps_dir(), 'myMap.png')
    map_filename = os.path.join(get_maps_dir(), 'gym/myMap.png')
    cv2.imwrite(map_filename, occupancy_grid)

    return map_filename


def main():
    """
    Main Function
    """
    np.random.seed(3)
    _, percent_explored, iterations_taken, _ = \
        run_frontier_exploration(map_filename=create_gym_environment_from_parameters("ComplexCostmapRandomAisleTurnEnv"),
                                 params_filename=os.path.join(get_exploration_dir(),"params/params.yaml"),
                                 map_resolution=0.03,
                                 start_state=None,
                                 sensor_range=10.0,
                                 completion_percentage=10,
                                 max_exploration_iterations=None,
                                 render_size_scale=2.0,
                                 render_interval=5)

    print("Map", "{:.2f}".format(percent_explored * 100), "\b% explored!",
          "This is " + str(iterations_taken) + " iterations!")


if __name__ == '__main__':
    main()
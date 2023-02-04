"""
This file holds all URLs.
"""

DATASET_URLS = {
    'maze2d-open-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-sparse.hdf5',
    'maze2d-umaze-v1': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5',
    'maze2d-medium-v1': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5',
    'maze2d-large-v1': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5',
    'maze2d-eval-umaze-v1': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-umaze-sparse-v1.hdf5',
    'maze2d-eval-medium-v1': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-medium-sparse-v1.hdf5',
    'maze2d-eval-large-v1': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-large-sparse-v1.hdf5',
    'maze2d-open-dense-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-dense.hdf5',
    'maze2d-umaze-dense-v1': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-dense-v1.hdf5',
    'maze2d-medium-dense-v1': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-dense-v1.hdf5',
    'maze2d-large-dense-v1': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-dense-v1.hdf5',
    'maze2d-eval-umaze-dense-v1': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-umaze-dense-v1.hdf5',
    'maze2d-eval-medium-dense-v1': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-medium-dense-v1.hdf5',
    'maze2d-eval-large-dense-v1': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-large-dense-v1.hdf5',
    'minigrid-fourrooms-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/minigrid/minigrid4rooms.hdf5',
    'minigrid-fourrooms-random-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/minigrid/minigrid4rooms_random.hdf5',
    'pen-human-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-v0_demos_clipped.hdf5',
    'pen-cloned-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-demos-v0-bc-combined.hdf5',
    'pen-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-v0_expert_clipped.hdf5',
    'hammer-human-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-v0_demos_clipped.hdf5',
    'hammer-cloned-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-demos-v0-bc-combined.hdf5',
    'hammer-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-v0_expert_clipped.hdf5',
    'relocate-human-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-v0_demos_clipped.hdf5',
    'relocate-cloned-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-demos-v0-bc-combined.hdf5',
    'relocate-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-v0_expert_clipped.hdf5',
    'door-human-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-v0_demos_clipped.hdf5',
    'door-cloned-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-demos-v0-bc-combined.hdf5',
    'door-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-v0_expert_clipped.hdf5',
    'halfcheetah-random-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_random.hdf5',
    'halfcheetah-medium-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium.hdf5',
    'halfcheetah-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_expert.hdf5',
    'halfcheetah-medium-replay-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_mixed.hdf5',
    'halfcheetah-medium-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium_expert.hdf5',
    'walker2d-random-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_random.hdf5',
    'walker2d-medium-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium.hdf5',
    'walker2d-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_expert.hdf5',
    'walker2d-medium-replay-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker_mixed.hdf5',
    'walker2d-medium-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium_expert.hdf5',
    'hopper-random-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_random.hdf5',
    'hopper-medium-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium.hdf5',
    'hopper-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_expert.hdf5',
    'hopper-medium-replay-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_mixed.hdf5',
    'hopper-medium-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium_expert.hdf5',
    'ant-random-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_random.hdf5',
    'ant-medium-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_medium.hdf5',
    'ant-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_expert.hdf5',
    'ant-medium-replay-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_mixed.hdf5',
    'ant-medium-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_medium_expert.hdf5',
    'ant-random-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_random_expert.hdf5',
    'antmaze-umaze-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
    'antmaze-umaze-diverse-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
    'antmaze-medium-play-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
    'antmaze-medium-diverse-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
    'antmaze-large-play-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
    'antmaze-large-diverse-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
    'antmaze-umaze-v2': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5',
    'antmaze-umaze-diverse-v2': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5',
    'antmaze-medium-play-v2': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5',
    'antmaze-medium-diverse-v2': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5',
    'antmaze-large-play-v2': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5',
    'antmaze-large-diverse-v2': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5',
    'flow-ring-random-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-ring-v0-random.hdf5',
    'flow-ring-controller-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-ring-v0-idm.hdf5',
    'flow-merge-random-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-merge-v0-random.hdf5',
    'flow-merge-controller-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-merge-v0-idm.hdf5',
    'kitchen-complete-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
    'kitchen-partial-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_light_slider-v0.hdf5',
    'kitchen-mixed-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    'carla-lane-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_lane_follow_flat-v0.hdf5',
    'carla-town-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_town_subsamp_flat-v0.hdf5',
    'carla-town-full-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_town_flat-v0.hdf5',
    'bullet-halfcheetah-random-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_random.hdf5',
    'bullet-halfcheetah-medium-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_medium.hdf5',
    'bullet-halfcheetah-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_expert.hdf5',
    'bullet-halfcheetah-medium-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_medium_expert.hdf5',
    'bullet-halfcheetah-medium-replay-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_medium_replay.hdf5',
    'bullet-hopper-random-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_random.hdf5',
    'bullet-hopper-medium-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_medium.hdf5',
    'bullet-hopper-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_expert.hdf5',
    'bullet-hopper-medium-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_medium_expert.hdf5',
    'bullet-hopper-medium-replay-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_medium_replay.hdf5',
    'bullet-ant-random-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_random.hdf5',
    'bullet-ant-medium-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_medium.hdf5',
    'bullet-ant-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_expert.hdf5',
    'bullet-ant-medium-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_medium_expert.hdf5',
    'bullet-ant-medium-replay-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_medium_replay.hdf5',
    'bullet-walker2d-random-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_random.hdf5',
    'bullet-walker2d-medium-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_medium.hdf5',
    'bullet-walker2d-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_expert.hdf5',
    'bullet-walker2d-medium-expert-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_medium_expert.hdf5',
    'bullet-walker2d-medium-replay-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_medium_replay.hdf5',
    'bullet-maze2d-open-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-maze2d-open-sparse.hdf5',
    'bullet-maze2d-umaze-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-maze2d-umaze-sparse.hdf5',
    'bullet-maze2d-medium-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-maze2d-medium-sparse.hdf5',
    'bullet-maze2d-large-v0': 'https://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-maze2d-large-sparse.hdf5',
}

# Gym-MuJoCo V1/V2 envs
for env in ['halfcheetah', 'hopper', 'walker2d', 'ant']:
    for dset in ['random', 'medium', 'expert', 'medium-replay', 'full-replay', 'medium-expert']:
        # v1 envs
        dset_name = env + '_' + dset.replace('-', '_') + '-v1'
        env_name = dset_name.replace('_', '-')
        DATASET_URLS[env_name] = 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v1/%s.hdf5' % dset_name

        # v2 envs
        dset_name = env + '_' + dset.replace('-', '_') + '-v2'
        env_name = dset_name.replace('_', '-')
        DATASET_URLS[env_name] = 'https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/%s.hdf5' % dset_name

# Adroit v1 envs
for env in ['hammer', 'pen', 'relocate', 'door']:
    for dset in ['human', 'expert', 'cloned']:
        env_name = env + '-' + dset + '-v1'
        DATASET_URLS[env_name] = 'https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/%s.hdf5' % env_name


DATASETS_MUJOCO = [
    'halfcheetah-random-v0',
    'halfcheetah-medium-v0',
    'halfcheetah-expert-v0',
    'halfcheetah-medium-replay-v0',
    'halfcheetah-medium-expert-v0',
    'walker2d-random-v0',
    'walker2d-medium-v0',
    'walker2d-expert-v0',
    'walker2d-medium-replay-v0',
    'walker2d-medium-expert-v0',
    'hopper-random-v0',
    'hopper-medium-v0',
    'hopper-expert-v0',
    'hopper-medium-replay-v0',
    'hopper-medium-expert-v0',
    'ant-random-v0',
    'ant-medium-v0',
    'ant-expert-v0',
    'ant-medium-replay-v0',
    'ant-medium-expert-v0',
    'ant-random-expert-v0']

DATASETS_MAZES = [
    'maze2d-open-v0', 
    'maze2d-umaze-v1',
    'maze2d-medium-v1',
    'maze2d-large-v1',
    'maze2d-eval-umaze-v1',
    'maze2d-eval-medium-v1',
    'maze2d-eval-large-v1',
    'maze2d-open-dense-v0',
    'maze2d-umaze-dense-v1',
    'maze2d-medium-dense-v1',
    'maze2d-large-dense-v1',
    'maze2d-eval-umaze-dense-v1',
    'maze2d-eval-medium-dense-v1',
    'maze2d-eval-large-dense-v1',
    'antmaze-umaze-v0',
    'antmaze-umaze-diverse-v0',
    'antmaze-medium-play-v0',
    'antmaze-medium-diverse-v0',
    'antmaze-large-play-v0',
    'antmaze-large-diverse-v0',
    'antmaze-umaze-v2',
    'antmaze-umaze-diverse-v2',
    'antmaze-medium-play-v2',
    'antmaze-medium-diverse-v2',
    'antmaze-large-play-v2',
    'antmaze-large-diverse-v2']

DATASETS_ADROIT = [
    'flow-ring-random-v0',
    'flow-ring-controller-v0',
    'flow-merge-random-v0',
    'flow-merge-controller-v0',    
    'kitchen-complete-v0',
    'kitchen-partial-v0',
    'kitchen-mixed-v0',
    'carla-lane-v0',
    'carla-town-v0',
    'carla-town-full-v0']

DATASETS_BULLET = [
    'bullet-halfcheetah-random-v0',
    'bullet-halfcheetah-medium-v0',
    'bullet-halfcheetah-expert-v0',
    'bullet-halfcheetah-medium-expert-v0',
    'bullet-halfcheetah-medium-replay-v0',
    'bullet-hopper-random-v0',
    'bullet-hopper-medium-v0',
    'bullet-hopper-expert-v0',
    'bullet-hopper-medium-expert-v0',
    'bullet-hopper-medium-replay-v0',
    'bullet-ant-random-v0',
    'bullet-ant-medium-v0',
    'bullet-ant-expert-v0',
    'bullet-ant-medium-expert-v0',
    'bullet-ant-medium-replay-v0',
    'bullet-walker2d-random-v0',
    'bullet-walker2d-medium-v0',
    'bullet-walker2d-expert-v0',
    'bullet-walker2d-medium-expert-v0',
    'bullet-walker2d-medium-replay-v0']
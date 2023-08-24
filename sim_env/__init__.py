from gym.envs.registration import register

register(
    id='TabletopEnv1-v0',
    entry_point='sim_env.tabletop:Tabletop',
    kwargs={
        'xml': 'env1',
    }
)

register(
    id='CloseDrawerEnv1-v0',
    entry_point='sim_env.tabletop_task:TabletopTask',
    kwargs={
        'xml': 'env1',
        'rewMode': 'close_drawer',
        'max_path_length': 60,
    }
)

register(
    id='CupForwardEnv1-v0',
    entry_point='sim_env.tabletop_task:TabletopTask',
    kwargs={
        'xml': 'env1',
        'rewMode': 'cup_forward',
        'max_path_length': 60,
    }
)

register(
    id='FaucetRightEnv1-v0',
    entry_point='sim_env.tabletop_task:TabletopTask',
    kwargs={
        'xml': 'env1',
        'rewMode': 'faucet_right',
        'max_path_length': 60,
    }
)

register(
    id='CloseDrawerEnv1-v1',
    entry_point='sim_env.tabletop_task:TabletopTask',
    kwargs={
        'xml': 'env1_heavy_mug',
        'rewMode': 'close_drawer',
        'max_path_length': 60,
    }
)

register(
    id='CloseDrawerEnv1Dense-v1',
    entry_point='sim_env.tabletop_task:TabletopTask',
    kwargs={
        'xml': 'env1_heavy_mug',
        'rewMode': 'close_drawer',
        'max_path_length': 60,
        'reward_type': 'dense',
    }
)

register(
    id='CupForwardEnv1-v1',
    entry_point='sim_env.tabletop_task:TabletopTask',
    kwargs={
        'xml': 'env1_heavy_mug',
        'rewMode': 'cup_forward',
        'max_path_length': 60,
    }
)

register(
    id='CupForwardEnv1Dense-v1',
    entry_point='sim_env.tabletop_task:TabletopTask',
    kwargs={
        'xml': 'env1_heavy_mug',
        'rewMode': 'cup_forward',
        'max_path_length': 60,
        'reward_type': 'dense',
    }
)

register(
    id='FaucetRightEnv1-v1',
    entry_point='sim_env.tabletop_task:TabletopTask',
    kwargs={
        'xml': 'env1_heavy_mug',
        'rewMode': 'faucet_right',
        'max_path_length': 60,
    }
)

register(
    id='FaucetRightEnv1Dense-v1',
    entry_point='sim_env.tabletop_task:TabletopTask',
    kwargs={
        'xml': 'env1_heavy_mug',
        'rewMode': 'faucet_right',
        'max_path_length': 60,
        'reward_type': 'dense',
    }
)

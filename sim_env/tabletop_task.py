import numpy as np

from sim_env import tabletop

class TabletopTask(tabletop.Tabletop):
    """Subclass of Tabletop with a pre-defined task and reward function."""

    def __init__(self, reward_type='sparse', **kwargs):
        self.quick_init(locals())
        super().__init__(**kwargs)
        drawer_pos = None
        faucet_pos = None
        if 'env3' in self.xml:
            target = [-0.05, .7, 0]
        elif 'env4' in self.xml:
            target = [0.17, .65, 0]
        else:
            target = [0, .7, 0]
            # Roughly the position of the front middle of the drawer when pushed
            # in. 
            drawer_pos = [0.25, 0.62, 0.06]
            # Roughly the position of the middle of the faucet when turned 
            # right.
            faucet_pos = [-0.27, 0.53, 0.11]
        self.target = np.array(target)
        self.drawer_pos = drawer_pos
        self.faucet_pos = faucet_pos
        self.reward_type = reward_type

    def reset(self):
        obs, unused_env_info = super().reset()
        return obs.flatten()

    def step(self, action):
        new_obs, reward, done, info = super().step(action)
        return new_obs.flatten(), reward, done, info

    def _get_low_dim_info(self):
        endeff_pos = self.get_endeff_pos()
        env_info =  {'mug_x': self.data.qpos[9],
                    'mug_y': self.data.qpos[10],
                    'mug_z': self.data.qpos[11],
                    'mug_quat_x': self.data.qpos[12],
                    'mug_quat_y': self.data.qpos[13],
                    'mug_quat_z': self.data.qpos[14],
                    'mug_quat_w': self.data.qpos[15],
                    'hand_x': endeff_pos[0],
                    'hand_y': endeff_pos[1],
                    'hand_z': endeff_pos[2],
                    'drawer': self.data.qpos[16],
                    'coffee_machine': self.data.qpos[17],
                    'faucet': self.data.qpos[18],
                    'dist': -0.0,
                    'image_shape':(self.imsize, self.imsize_x, 3)}
        return env_info

    def compute_reward(self):
        env_info = self._get_low_dim_info()
        reward_info = {}
        if self.rewMode == 'close_drawer':  # task 5: close drawer
            reward = int(env_info['drawer'] > -0.05)
            reward_info['success'] = reward
            if self.reward_type == 'dense':
                hand_pos = np.array(
                    [env_info[k] for k in ['hand_x', 'hand_y', 'hand_z']])
                dist_hand_drawer = np.sqrt(
                    np.sum((hand_pos - self.drawer_pos)**2))
                reward += env_info['drawer']
                reward += -dist_hand_drawer
                reward_info['drawer_reward'] = env_info['drawer']
                reward_info['dist_hand_drawer'] = dist_hand_drawer
        elif self.rewMode == 'cup_forward':  # task 41: push cup forward
            mug_pos = np.array(
                [env_info[k] for k in ['mug_x', 'mug_y', 'mug_z']])
            dist_cup_target = np.sqrt(np.sum((mug_pos - self.target)**2))
            reward = int(dist_cup_target < 0.075)
            reward_info['success'] = reward
            if self.reward_type == 'dense':
                hand_pos = np.array(
                    [env_info[k] for k in ['hand_x', 'hand_y', 'hand_z']])
                dist_hand_cup = np.sqrt(np.sum((hand_pos - self.target)**2))
                reward += -dist_cup_target - dist_hand_cup
                reward_info['dist_cup_target'] = dist_cup_target
                reward_info['dist_hand_cup'] = dist_hand_cup
        elif self.rewMode == 'faucet_right':  # task 93: turn faucet right
            reward = env_info['faucet'] < -0.01
            reward_info['success'] = reward
            if self.reward_type == 'dense':
                hand_pos = np.array(
                    [env_info[k] for k in ['hand_x', 'hand_y', 'hand_z']])
                dist_hand_faucet = np.sqrt(
                    np.sum((hand_pos - self.faucet_pos)**2))
                reward += -env_info['faucet']
                reward += -dist_hand_faucet
                reward_info['faucet_reward'] = -env_info['faucet']
                reward_info['dist_hand_faucet'] = dist_hand_faucet
        else:
            reward = 0.0
        return reward, reward_info


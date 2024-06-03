from tianshou.env import DummyVectorEnv
import warnings
import pdb
from typing import Any, Callable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import packaging

from tianshou.env.utils import ENV_TYPE, gym_new_venv_step_type
from tianshou.env.worker import (
    DummyEnvWorker,
    EnvWorker,
    RayEnvWorker,
    SubprocEnvWorker,
)

try:
    import gym as old_gym

    has_old_gym = True
except ImportError:
    has_old_gym = False

GYM_RESERVED_KEYS = [
    "metadata", "reward_range", "spec", "action_space", "observation_space"
]

class DummyCrowdVectorEnv(DummyVectorEnv):
    def __init__(self, env_fns: List[Callable[[], ENV_TYPE]], **kwargs: Any) -> None:
        super().__init__(env_fns, **kwargs)
        self.update_holes_for_each_agent()

    def update_holes_for_each_agent(self):
        for env_id in range(len(self._env_fns)):
            other_id = [i for i in range(len(self._env_fns))]
            other_id.remove(env_id)
            global_map = self.get_env_attr('bbox', other_id)
            self.set_env_attr('holes', global_map, env_id)
    
    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> gym_new_venv_step_type:
        """Run one timestep of some environments' dynamics.

        If id is None, run one timestep of all the environments’ dynamics;
        otherwise run one timestep for some environments with given id,  either
        an int or a list. When the end of episode is reached, you are
        responsible for calling reset(id) to reset this environment’s state.

        Accept a batch of action and return a tuple (batch_obs, batch_rew,
        batch_done, batch_info) in numpy format.

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple consisting of either:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``terminated`` a numpy.ndarray, whether these episodes have been \
                terminated
            * ``truncated`` a numpy.ndarray, whether these episodes have been truncated
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

        For the async simulation:

        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        None, fetch unfinished step() calls instead.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if not self.is_async:
            assert len(action) == len(id)
            for i, j in enumerate(id):
                # update global map before taking an action
                self.update_holes_for_each_agent()
                self.workers[j].send(action[i])
            result = []
            for j in id:
                env_return = self.workers[j].recv()
                env_return[-1]["env_id"] = j
                result.append(env_return)
        else:
            if action is not None:
                self._assert_id(id)
                assert len(action) == len(id)
                for act, env_id in zip(action, id):
                    # update global map before taking an action
                    self.update_holes_for_each_agent()
                    self.workers[env_id].send(act)
                    self.waiting_conn.append(self.workers[env_id])
                    self.waiting_id.append(env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
            ready_conns: List[EnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(
                    self.waiting_conn, self.wait_num, self.timeout
                )
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                # env_return can be (obs, reward, done, info) or
                # (obs, reward, terminated, truncated, info)
                env_return = conn.recv()
                env_return[-1]["env_id"] = env_id  # Add `env_id` to info
                result.append(env_return)
                self.ready_id.append(env_id)
        obs_list, rew_list, term_list, trunc_list, info_list = tuple(zip(*result))
        try:
            obs_stack = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_list, dtype=object)
        return (
            obs_stack,
            np.stack(rew_list),
            np.stack(term_list),
            np.stack(trunc_list),
            np.stack(info_list),
        )

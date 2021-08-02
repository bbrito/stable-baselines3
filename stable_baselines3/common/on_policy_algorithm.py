import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
import datetime
import uuid
import os
import logging
import dataclasses

from stable_baselines3.common import logger, evaluation_old
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv
from imitation.data import rollout,  types
from stable_baselines3.common import utils
from config import ImitationConfig


def make_unique_timestamp() -> str:
    """Timestamp, with random uuid added to avoid collisions."""
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    random_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{random_uuid}"

def _save_trajectory(
    npz_path: str,
    trajectory: types.Trajectory,
) -> None:
    """Save a trajectory as a compressed Numpy file."""
    save_dir = os.path.dirname(npz_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    assert isinstance(trajectory, types.Trajectory)
    np.savez_compressed(npz_path, **dataclasses.asdict(trajectory))


def evaluate_policy(policy, env, seeds, log_dir=None, basename=None):
    """
    Evaluates the `policy` on a given `env` for a set of `seeds` and returns a list of rewards of all
    rollouts. If `log_dir` and `basename` are given then a visualization of the rollout (as VegaLite
    html) will be stored to the `log_dir`.
    """
    all_rewards = []

    for seed in seeds:
        env.seed(seed)
        this_rewards, _ = evaluation_old.evaluate_policy(
            policy, env, return_episode_rewards=True, n_eval_episodes=2
        )
        if log_dir is not None and basename is not None:
            rollout_path = path.join(log_dir, "eval")
            Path(rollout_path).mkdir(parents=True, exist_ok=True)
            viz = env.render()
            viz.properties(width=500, height=500).save(
                path.join(rollout_path, f"{basename}-{seed}.html")
            )
        all_rewards += this_rewards

    return all_rewards

class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        save_path: str = None,
    ):

        super(OnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.rampdown_rounds = 15
        self.traj_accum = None
        self.save_path = save_path


        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            net_arch=ImitationConfig.net_arch,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int, global_step: int
    ) -> bool:
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()

        #self.traj_accum = self.env.venv.venv.venv._traj_accum#rollout.TrajectoryAccumulator()

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        expert_pol = env.expert_policy

        round_num = global_step
        self.round_num = round_num
        self.save_path_round = os.path.join(self.save_path , "demos", f"round-{round_num:03d}")
        col_obs = np.zeros((1,8))
        col_rews = np.zeros((1,))

        if self.dagger:
            self.traj_accum = rollout.TrajectoryAccumulator()
            self.traj_accum.add_step({"obs": self._last_obs})

        self._last_obs = env.reset()
        self.traj_accum = rollout.TrajectoryAccumulator()
        self.traj_accum.add_step({"obs": self._last_obs})

        while n_steps < n_rollout_steps:


            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            # Query expert action
            (expert_action,), _ = expert_pol.predict(
                self._last_obs,
                deterministic=True,
            )
            # Query policy action
            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                policy_action_th, values, log_probs = self.policy.forward(obs_tensor)

            policy_action = policy_action_th.cpu().numpy()

            if self.dagger:
                # Replace the given action with a robot action 100*(1-beta)% of the time.
                self.beta = min(1, max(0, (self.rampdown_rounds - self.round_num) / self.rampdown_rounds))
                if np.random.uniform(0, 1) > self.beta:
                    actions = policy_action
                else:
                    actions = np.expand_dims(expert_action,axis=0)

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
                clipped_expert_actions = np.clip(np.expand_dims(expert_action,axis=0), self.action_space.low, self.action_space.high)

            next_obs, reward, done, info = env.step(clipped_actions)
            self._last_obs = info[0]['original_ob']

            if self.dagger and done:
                self.traj_accum.add_step(
                    {"acts": clipped_expert_actions, "obs": np.expand_dims(info[0]['terminal_observation'],0), "rews": info[0]['original_rw'],
                     "infos": info}
                )
                trajectory = self.traj_accum.finish_trajectory()
                timestamp = make_unique_timestamp()
                trajectory_path = os.path.join(
                    self.save_path_round, "dagger-gail-demo-" + timestamp + ".npz"
                )
                logging.info(f"Saving demo at '{trajectory_path}'")
                _save_trajectory(trajectory_path, trajectory)
                self.traj_accum = rollout.TrajectoryAccumulator()
                self.traj_accum.add_step({"obs": info[0]['original_ob']})
            elif not done:
                self.traj_accum.add_step(
                    {"acts": clipped_expert_actions, "obs": info[0]['original_ob'], "rews": info[0]['original_rw'],
                     "infos": info}
                )


            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(info)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, reward, self._last_dones, values, log_probs)
            #self._last_obs = next_obs
            self._last_dones = done

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(next_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=done)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        save_path: str = None,
        global_step: int = 0,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            #self.reset()
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, global_step=global_step)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                #if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

            # load expert data
            self._last_loaded_round = -1
            self.DEMO_SUFFIX = ".npz"
            self._all_demos = []
            self.expert_data_loader: Optional[Iterable[Mapping]] = None
            self._try_load_demos()

            self.train(n_epochs = 1)
            #self.round_num += 1
            print("training round: ", self.round_num)



        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

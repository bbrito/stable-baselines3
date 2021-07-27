from typing import Any, Callable, Dict, Optional, Type, Union

import contextlib
import numpy as np
import torch as th
import os
import logging
from gym import spaces
from torch.nn import functional as F
import tqdm.autonotebook as tqdm

from imitation.data import rollout, types
from stable_baselines3.common import logger
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
#from imitation.algorithms.bc_old import EpochOrBatchIteratorWithProgress
#from imitation.policies import base
#from imitation.data import types

import itertools
import torch.utils.data as th_data
from imitation.data import buffer, types, wrappers
from typing import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

T = TypeVar("T")


def endless_iter(iterable: Iterable[T]) -> Iterator[T]:
    """Generator that endlessly yields elements from iterable.

    If any call to `iter(iterable)` has no elements, then this function raises
    ValueError.

    >>> x = range(2)
    >>> it = endless_iter(x)
    >>> next(it)
    0
    >>> next(it)
    1
    >>> next(it)
    0

    """
    try:
        next(iter(iterable))
    except StopIteration:
        err = ValueError(f"iterable {iterable} had no elements to iterate over.")
        raise err

    return itertools.chain.from_iterable(itertools.repeat(iterable))

class EpochOrBatchIteratorWithProgress:
    def __init__(
        self,
        data_loader: Iterable[dict],
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
        on_batch_end: Optional[Callable[[], None]] = None,
    ):
        """Wraps DataLoader so that all BC batches can be processed in a one for-loop.
        Also uses `tqdm` to show progress in stdout.
        Args:
            data_loader: An iterable over data dicts, as used in `BC`.
            n_epochs: The number of epochs to iterate through in one call to
                __iter__. Exactly one of `n_epochs` and `n_batches` should be provided.
            n_batches: The number of batches to iterate through in one call to
                __iter__. Exactly one of `n_epochs` and `n_batches` should be provided.
            on_epoch_end: A callback function without parameters to be called at the
                end of every epoch.
            on_batch_end: A callback function without parameters to be called at the
                end of every batch.
        """
        if n_epochs is not None and n_batches is None:
            self.use_epochs = True
        elif n_epochs is None and n_batches is not None:
            self.use_epochs = False
        else:
            raise ValueError(
                "Must provide exactly one of `n_epochs` and `n_batches` arguments."
            )

        self.data_loader = data_loader
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.on_epoch_end = on_epoch_end
        self.on_batch_end = on_batch_end

    def __iter__(self) -> Iterable[Tuple[dict, dict]]:
        """Yields batches while updating tqdm display to display progress."""

        samples_so_far = 0
        epoch_num = 0
        batch_num = 0
        batch_suffix = epoch_suffix = ""
        if self.use_epochs:
            display = tqdm.tqdm(total=self.n_epochs)
            epoch_suffix = f"/{self.n_epochs}"
        else:  # Use batches.
            display = tqdm.tqdm(total=self.n_batches)
            batch_suffix = f"/{self.n_batches}"

        def update_desc():
            display.set_description(
                f"batch: {batch_num}{batch_suffix}  epoch: {epoch_num}{epoch_suffix}"
            )

        with contextlib.closing(display):
            while True:
                update_desc()
                got_data_on_epoch = False
                for batch in self.data_loader:
                    got_data_on_epoch = True
                    batch_num += 1
                    batch_size = len(batch["obs"])
                    assert batch_size > 0
                    samples_so_far += batch_size
                    stats = dict(
                        epoch_num=epoch_num,
                        batch_num=batch_num,
                        samples_so_far=samples_so_far,
                    )
                    yield batch, stats
                    if self.on_batch_end is not None:
                        self.on_batch_end()
                    if not self.use_epochs:
                        update_desc()
                        display.update(1)
                        if batch_num >= self.n_batches:
                            return
                if not got_data_on_epoch:
                    raise AssertionError(
                        f"Data loader returned no data after "
                        f"{batch_num} batches, during epoch "
                        f"{epoch_num} -- did it reset correctly?"
                    )
                epoch_num += 1
                if self.on_epoch_end is not None:
                    self.on_epoch_end()

                if self.use_epochs:
                    update_desc()
                    display.update(1)
                    if epoch_num >= self.n_epochs:
                        return


def _load_trajectory(npz_path: str) -> types.Trajectory:
    """Load a single trajectory from a compressed Numpy file."""
    np_data = np.load(npz_path, allow_pickle=True)
    has_rew = "rews" in np_data
    cls = types.TrajectoryWithRew if has_rew else types.Trajectory
    return cls(**dict(np_data.items()))



class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        expert_data,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 32,
        n_epochs: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.05,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        alpha: int = 1.0,
        decay: int = 0.99,
        save_path: str = None,
    ):

        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.alpha = alpha
        self.decay = decay
        self.dagger = True
        self.bc = False
        self.ppo = False
        self.save_path = save_path

        if expert_data is not None:
            if isinstance(expert_data, types.Transitions):
                if len(expert_data) < batch_size:
                    raise ValueError(
                        "Provided Transitions instance as `expert_data` argument but "
                        "len(expert_data) < expert_batch_size. "
                        f"({len(expert_data)} < {batch_size})."
                    )

                self.expert_data_loader = th_data.DataLoader(
                    expert_data,
                    batch_size=batch_size,
                    collate_fn=types.transitions_collate_fn,
                    shuffle=True,
                    drop_last=True,
                )
            else:
                self.expert_data_loader = expert_data
            self._endless_expert_iterator = endless_iter(self.expert_data_loader)

        if _init_setup_model:
            self._setup_model()

    def _next_expert_batch(self) -> Mapping:
        return next(self._endless_expert_iterator)

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def set_expert_data_loader(
            self,
            expert_data: Union[Iterable[Mapping], types.TransitionsMinimal],
    ) -> None:
        """Set the expert data loader, which yields batches of obs-act pairs.
        Changing the expert data loader on-demand is useful for DAgger and other
        interactive algorithms.
        Args:
            expert_data: Either a Torch `DataLoader`, any other iterator that
                yields dictionaries containing "obs" and "acts" Tensors or Numpy arrays,
                or a `TransitionsMinimal` instance.
                If this is a `TransitionsMinimal` instance, then it is automatically
                converted into a shuffled `DataLoader` with batch size
                `BC.DEFAULT_BATCH_SIZE`.
        """
        if isinstance(expert_data, types.TransitionsMinimal):
            self.expert_data_loader = th_data.DataLoader(
                expert_data,
                shuffle=True,
                batch_size=BC.DEFAULT_BATCH_SIZE,
                collate_fn=types.transitions_collate_fn,
            )
        else:
            self.expert_data_loader = expert_data

    def _load_all_demos(self):
        num_demos_by_round = []
        for round_num in range(self._last_loaded_round + 1, self.round_num + 1):
            round_dir = self._demo_dir_path_for_round(round_num)
            demo_paths = self._get_demo_paths(round_dir)
            self._all_demos.extend(_load_trajectory(p) for p in demo_paths)
            num_demos_by_round.append(len(demo_paths))
        logging.info(f"Loaded {len(self._all_demos)} total")
        demo_transitions = rollout.flatten_trajectories(self._all_demos)
        return demo_transitions, num_demos_by_round



    def _get_demo_paths(self, round_dir):
        return [
            os.path.join(round_dir, p)
            for p in os.listdir(round_dir)
            if p.endswith(self.DEMO_SUFFIX)
        ]

    def _demo_dir_path_for_round(self, round_num=None):
        if round_num is None:
            round_num = self.round_num
        return os.path.join(self.save_path, "demos", f"round-{round_num:03d}")

    def _check_has_latest_demos(self):
        demo_dir = self._demo_dir_path_for_round()
        demo_paths = self._get_demo_paths(demo_dir) if os.path.isdir(demo_dir) else []
        if len(demo_paths) == 0:
            raise NeedsDemosException(
                f"No demos found for round {self.round_num} in dir '{demo_dir}'. "
                f"Maybe you need to collect some demos? See "
                f".get_trajectory_collector()"
            )

    def _try_load_demos(self):
        self._check_has_latest_demos()
        if self._last_loaded_round < self.round_num:
            transitions, num_demos = self._load_all_demos()
            logging.info(
                f"Loaded {sum(num_demos)} new demos from {len(num_demos)} rounds"
            )
            data_loader = th_data.DataLoader(
                transitions,
                self.batch_size,
                drop_last=True,
                shuffle=True,
                collate_fn=types.transitions_collate_fn,
            )
            self.set_expert_data_loader(data_loader)
            self._last_loaded_round = self.round_num

    def train(self,
        *,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Callable[[], None] = None,
        on_batch_end: Callable[[], None] = None,
        log_interval: int = 100,
    ) -> None:
        """
        Update policy using the currently gathered
        rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)





        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        mse = []
        clip_fractions = []
        batch_num = 0
        # train for gradient_steps epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                loss = 0
                actions = rollout_data.actions
                if self.ppo:
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    # TODO: investigate why there is no issue with the gradient
                    # if that line is commented (as in SAC)
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    loss_ppo = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    loss += loss_ppo

                if self.bc:
                    # Get expert batch
                    expert_samples = self._next_expert_batch()
                    with th.no_grad():
                        # Compute value for the last timestep
                        obs_tensor = th.as_tensor(expert_samples['obs']).to(self.device)
                        actions_tensor = th.as_tensor(expert_samples['acts']).to(self.device)

                    # BC-GAIL
                    _, alogprobs, _ = self.policy.evaluate_actions(obs_tensor, actions_tensor)
                    bcloss = -alogprobs.mean()
                    loss = self.alpha * bcloss + (1 - self.alpha) * loss

                if self.ppo or self.bc:
                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

                if self.dagger:
                    #batch_size = 32
                    #transitions = {}
                    #transitions['acts'] = actions.detach().cpu().numpy()
                    #transitions['obs'] = rollout_data.observations.detach().cpu().numpy()

                    it = EpochOrBatchIteratorWithProgress(
                        self.expert_data_loader,
                        n_epochs=self.n_epochs,
                        n_batches=n_batches,
                        on_epoch_end=on_epoch_end,
                        on_batch_end=on_batch_end,
                    )

                    batch_num = 0
                    for batch, stats_dict_it in it:

                        obs = th.as_tensor(batch["obs"], device=self.device).detach()
                        acts = th.as_tensor(batch["acts"], device=self.device).detach()

                        values, log_prob, entropy = self.policy.evaluate_actions(obs,acts)

                        prob_true_act = th.exp(log_prob).mean()
                        log_prob = log_prob.mean()
                        entropy = entropy.mean()

                        l2_norms = [th.sum(th.square(w)) for w in self.policy.parameters()]
                        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square

                        #TODO: MOVE THIS
                        self.ent_weight = 1e-3
                        self.l2_weight = 0

                        ent_loss = -self.ent_weight * entropy
                        neglogp = -log_prob
                        l2_loss = self.l2_weight * l2_norm
                        loss = neglogp + ent_loss + l2_loss

                        self.policy.optimizer.zero_grad()
                        loss.backward()
                        self.policy.optimizer.step()




                        print(batch_num)

                        stats_dict_loss = dict(
                            neglogp=neglogp.item(),
                            loss=loss.item(),
                            entropy=entropy.item(),
                            ent_loss=ent_loss.item(),
                            prob_true_act=prob_true_act.item(),
                            l2_norm=l2_norm.item(),
                            l2_loss=l2_loss.item(),
                        )

                        if batch_num % log_interval == 0:
                            for stats in [stats_dict_it, stats_dict_loss]:
                                for k, v in stats.items():
                                    logger.record(k, v)
                            logger.dump(batch_num)
                        batch_num += 1

                        print(batch_num)


            if self.ppo:
                all_kl_divs.append(np.mean(approx_kl_divs))

                if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                    break

        self._n_updates += self.n_epochs
        self.alpha *= self.decay
        explained_var = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())

        # Logs
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", loss.item())
        logger.record("train/explained_variance", explained_var)
        logger.record("train/gamma", self.alpha)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        save_path: str = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO":

        return super(PPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            save_path = save_path,
        )

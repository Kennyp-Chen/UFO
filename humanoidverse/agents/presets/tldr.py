"""Internal implementation of the TeCH agent.

Historical TLDR names are kept for checkpoint/config compatibility.
"""

from __future__ import annotations

from humanoidverse.agents.gcr_rl_dist_aux.model import GcrRlDistAuxModelArchiConfig, GcrRlDistAuxModelConfig
from humanoidverse.agents.tldr_dist_aux.agent import TldrDistAuxAgentConfig, TldrDistAuxAgentTrainConfig
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
    RewardNormalizerConfig,
)
from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig


TRAIN_RUNTIME = {
    "log_every_updates": 384000,
    "update_agent_every": 1024,
    "num_seed_steps": 10240,
    "num_agent_updates": 128,
    "checkpoint_buffer": True,
    "use_trajectory_buffer": True,
    "buffer_size": 5120000,
    "eval_every_steps": 3200000,
}


def build_tldr_agent(
    device: str = "cuda",
    compile: bool = True,
    update_z_every_step: int = 10,
    low_memory: bool = False,
) -> TldrDistAuxAgentConfig:
    if low_memory:
        z_dim = 128
        actor_hidden_dim = 1024
        actor_hidden_layers = 4
        critic_hidden_dim = 1024
        critic_hidden_layers = 4
        aux_critic_hidden_dim = 1024
        aux_critic_hidden_layers = 4
        goal_encoder_hidden_dim = 128
        discriminator_hidden_dim = 512
        discriminator_hidden_layers = 2
        batch_size = 512
        inference_batch_size = 256000
    else:
        z_dim = 256
        actor_hidden_dim = 2048
        actor_hidden_layers = 6
        critic_hidden_dim = 2048
        critic_hidden_layers = 6
        aux_critic_hidden_dim = 2048
        aux_critic_hidden_layers = 6
        goal_encoder_hidden_dim = 256
        discriminator_hidden_dim = 1024
        discriminator_hidden_layers = 3
        batch_size = 1024
        inference_batch_size = 500000

    return TldrDistAuxAgentConfig(
        name="TldrDistAuxAgent",
        model=GcrRlDistAuxModelConfig(
            name="GcrRlDistAuxModel",
            device=device,
            archi=GcrRlDistAuxModelArchiConfig(
                name="GcrRlDistAuxModelArchiConfig",
                z_dim=z_dim,
                norm_z=True,
                goal_encoder=BackwardArchiConfig(
                    name="BackwardArchi",
                    hidden_dim=goal_encoder_hidden_dim,
                    hidden_layers=1,
                    norm=True,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"]),
                ),
                actor=ActorArchiConfig(
                    name="actor",
                    model="residual",
                    hidden_dim=actor_hidden_dim,
                    hidden_layers=actor_hidden_layers,
                    embedding_layers=2,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "last_action", "history_actor"]),
                ),
                critic=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=critic_hidden_dim,
                    model="residual",
                    hidden_layers=critic_hidden_layers,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"]
                    ),
                ),
                discriminator=DiscriminatorArchiConfig(
                    name="DiscriminatorArchi",
                    hidden_dim=discriminator_hidden_dim,
                    hidden_layers=discriminator_hidden_layers,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"]),
                ),
                aux_critic=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=aux_critic_hidden_dim,
                    model="residual",
                    hidden_layers=aux_critic_hidden_layers,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"]
                    ),
                ),
            ),
            obs_normalizer=ObsNormalizerConfig(
                name="ObsNormalizerConfig",
                normalizers={
                    "state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "privileged_state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "last_action": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "history_actor": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                },
                allow_mismatching_keys=True,
            ),
            inference_batch_size=inference_batch_size,
            seq_length=8,
            actor_std=0.05,
            amp=False,
            norm_aux_reward=RewardNormalizerConfig(name="RewardNormalizer", translate=False, scale=True),
        ),
        train=TldrDistAuxAgentTrainConfig(
            name="TldrDistAuxAgentTrainConfig",
            train_goal_ratio=0.2,
            expert_asm_ratio=0.6,
            relabel_ratio=0.4,
            lr_goal_encoder=8e-7,
            lr_actor=5e-5,
            lr_critic=5e-5,
            lr_discriminator=2e-6,
            lr_aux_critic=2e-5,
            lr_dual_lam=4e-5,
            weight_decay=0.0,
            weight_decay_discriminator=0.0,
            clip_grad_norm=0.0,
            batch_size=batch_size,
            discount=0.98,
            stddev_clip=0.3,
            actor_pessimism_penalty=0.5,
            critic_pessimism_penalty=0.5,
            aux_critic_pessimism_penalty=0.5,
            critic_target_tau=0.005,
            use_mix_rollout=True,
            update_z_every_step=int(update_z_every_step),
            z_buffer_size=8192,
            rollout_expert_trajectories=True,
            rollout_expert_trajectories_length=250,
            rollout_expert_trajectories_percentage=0.5,
            grad_penalty_discriminator=10.0,
            use_tldr_pretrain=True,
            tldr_pretrain_env_steps=200000,
            tldr_te_during_rl=True,
            freeze_goal_encoder_after_pretrain=False,
            dual_reg=True,
            dual_lam_init=3000.0,
            dual_slack=1.0,
            tldr_softplus_scale=500.0,
            tldr_softplus_beta=0.01,
            tldr_reward_scale=1.0,
            goal_encoder_lr_schedule="none",
            goal_encoder_lr_schedule_steps=0,
            goal_encoder_lr_min=1e-6,
            reg_coeff_disc=0.03,
            reg_coeff_aux=0.01,
            disc_reward_coef=0.5,
            scale_reg=True,
        ),
        aux_rewards=[
            "penalty_torques",
            "penalty_action_rate",
            "limits_dof_pos",
            "limits_torque",
            "penalty_undesired_contact",
            "penalty_feet_ori",
            "penalty_ankle_roll",
            "penalty_slippage",
        ],
        aux_rewards_scaling={
            "penalty_action_rate": -0.1,
            "penalty_feet_ori": -0.4,
            "penalty_ankle_roll": -4.0,
            "limits_dof_pos": -10.0,
            "penalty_slippage": -2.0,
            "penalty_undesired_contact": -1.0,
            "penalty_torques": 0.0,
            "limits_torque": 0.0,
        },
        cudagraphs=False,
        compile=compile,
    )

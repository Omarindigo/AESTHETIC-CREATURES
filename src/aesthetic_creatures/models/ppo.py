from __future__ import annotations

from stable_baselines3 import PPO


def build_model(config, env) -> PPO:
    policy_kwargs = dict(net_arch=list(config.policy_net))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=config.device,
    )
    return model


def load_model(model_path: str) -> PPO:
    return PPO.load(model_path)

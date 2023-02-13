from gymnasium.envs.registration import register

register(
     id="am_gym_env/DualArmAM-v0",
     entry_point="am_gym_env.envs:DualArmAM",
     max_episode_steps=300,
)
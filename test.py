import gymnasium as gym
import numpy as np
import am_gym_env
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation

env = gym.make('am_gym_env/DualArmAM',param_file='./am_gym_env/config/dual_arm_env.yaml', render_mode="human")
env.reset()

for i in range(3000):
    action = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    state, reward, terminated, truncated, info = env.step(action)

env.close()



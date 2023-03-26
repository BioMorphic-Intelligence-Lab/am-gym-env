import gymnasium as gym
import numpy as np
from dual_arm_am import DualArmAM

# Heurisic: suboptimal, have no notion of balance.
env = DualArmAM(render_mode='human')
env._max_episode_steps = 600
env.reset()
steps = 0
total_reward = 0
a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
SPEED = 0.29  # Will fall forward on higher speed
state = STAY_ON_ONE_LEG
moving_leg = 0
supporting_leg = 1 - moving_leg
SUPPORT_KNEE_ANGLE = +0.1
supporting_knee_angle = SUPPORT_KNEE_ANGLE
while True:
    s, r, terminated, truncated, info = env.step(a)
    total_reward += r
    if steps % 20 == 0 or terminated or truncated:
        print("\naction " + str([f"{x:+0.2f}" for x in a]))
        print(f"step {steps} total_reward {total_reward:+0.2f}")
        print("hull " + str([f"{x:+0.2f}" for x in s[0:4]]))
        print("leg0 " + str([f"{x:+0.2f}" for x in s[4:9]]))
        print("leg1 " + str([f"{x:+0.2f}" for x in s[9:14]]))
    steps += 1

    a[0] = 0.45
    a[1] = 0.45
    a[2] = 0.07
    a[3] = -0.01
    a[4] = -0.07
    a[5] = 0.01
    
    if terminated or truncated:
        break
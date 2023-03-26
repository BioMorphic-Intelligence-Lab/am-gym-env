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


    ###################### Simple Heuristic: ###########################
    ####### The quadrotor slowly descends while constant torque   ######
    ####### on the joint motors                                   ######
    ####################################################################
    
    theta = s[0]
    thetadot = s[1]
    xdot = s[2]
    ydot = s[3]
    j0 = s[4]
    j0dot = s[5]
    j1 = s[6]
    j1dot = s[7]
    j2 = s[9]
    j2dot = s[10]
    j3 = s[11]
    j3dot = s[12]

    # First the descending control of the quadrotor:
    # Simple PD control on attitude while maintaining thrust
    # That is a bit less then the gravity
    k_att = 1
    d_att = 0.1
    a[0] = -k_att *theta + d_att * thetadot
    a[1] = (k_att *theta + d_att * thetadot)

    # Add thrust to have total thrust that (almost) matches gravity
    addon = np.cos(theta) * 0.50
    a[0] += addon
    a[1] += addon

    a[2] = 2.0
    a[3] = 0.002
    a[4] = -2.0
    a[5] = -0.002
    
    if terminated or truncated:
        break
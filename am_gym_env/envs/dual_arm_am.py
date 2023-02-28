import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import gymnasium as gym

from gymnasium import spaces
from scipy.integrate import odeint
from matplotlib.patches import Rectangle

class DualArmAM(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 24}

    def __init__(self,  param_file, render_mode=None):
        
        # Read all the system parameters
        self._read_params(param_file)
        # Define the home configuration of the robotic arms
        self.home = np.pi / 180.0 * np.array([-45, 45, 45, -45])

        # Observations are dictionaries with the base and the arms state, i.e. position, velocity
        # and optionally the acceleration.
        self.observation_space = spaces.Dict(
            {
                "base_position": spaces.Box(-10, 10, shape=(3,), dtype=float),
                "base_velocity": spaces.Box(-100, 100, shape=(3,), dtype=float),
                "base_acceleration": spaces.Box(-100, 100, shape=(3,), dtype=float),
                "arm_position": spaces.Box(-np.pi, np.pi, shape=(4,), dtype=float),
                "arm_velocity": spaces.Box(-100, 100, shape=(4,), dtype=float),
                "arm_acceleration": spaces.Box(-100, 100, shape=(4,), dtype=float),
            }
        )

        # We have 6 motors that can be actuated: Two propellers and one at each joint of the 
        # robotic arms, i.e. there are six continuous control actions to be determined
        self.action_space = spaces.Box(-np.array([10.0,10.0,5.0,5.0,5.0,5.0]),
                                        np.array([10.0,10.0,5.0,5.0,5.0,5.0]),
                                       shape=(6,), dtype=float)


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.index = 0
        self.t_i = 0
        self.log = [{
            "base_position": np.zeros([3, 3000]),
            "base_velocity": np.zeros([3, 3000]),
            "base_acceleration": np.zeros([3, 3000]),
            "arm_position": np.zeros([4, 3000]),
            "arm_velocity": np.zeros([4, 3000]),
            "arm_acceleration": np.zeros([4, 3000]),
        } for i in range(2)]
    
    def _read_params(self, param_file):

        # Open the file
        f = open(param_file)
        # Load the data from the parameterfile as a dictionary
        data = yaml.load(f, Loader=yaml.FullLoader)
        # We're done with the file. Close it again
        f.close()

        # Store the values in class variables
        self.m_base = data["Base"]["m"]    # Mass of the base
        self.l_base = data["Base"]["l"]    # Length of the base
        self.off = data["Base"]["offset"]  # Mounting offset of the arms from the center
        self.m_link = data["Link"]["m"]    # Mass of one link
        self.l_link = data["Link"]["l"]    # Length of one link

        self.ts = data["General"]["ts"]    # Timestep of one period
        self.g = data["General"]["g"]      # Gravity Constant
         
        
    def _get_obs(self):
        return {
            "base_position": self._base_position,
            "base_velocity": self._base_velocity,
            "base_acceleration": self._base_acceleration,
            "arm_position": self._arm_position,
            "arm_velocity": self._arm_velocity,
            "arm_acceleration": self._arm_velocity,
            }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._base_position, ord=2
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize the system in a certain region at random with zero velocity 
        self._base_position = np.append(self.np_random.uniform(np.array([-2.0, 1.0]),
                                                                     np.array([2.0, 4.0]), size=2),
                                             0)
        self._base_velocity = np.zeros(3)
        self._base_acceleration = np.zeros(3)

        # The arms are in the 'home' configuration and also at zero velocity
        self._arm_position = self.home
        self._arm_velocity = np.zeros(4)
        self._arm_acceleration = np.zeros(4)

        if self.render_mode == "human":
            self.log_state()

        # Reset the timestamp index
        self.t_i = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        
        # Accumulate the current state into the state vector x
        x_0 = np.concatenate((self._base_position, self._arm_position, self._base_velocity, self._arm_velocity))
        # Assume control input constant for the time interval
        f = lambda x, t: self.dynamic_model(x, action)
        x_dot = f(x_0, 0)
        # Initiate the time vector upto the next time step
        t = np.linspace(0, self.ts, num=2)
        # Numerically integrate to find the next state
        x_next = odeint(f, x_0, t)
        
        # Extract the actual state
        self._base_position = x_next[-1,:3]
        self._arm_position = x_next[-1,3:7]
        self._base_velocity = x_next[-1,7:10]
        self._arm_velocity = x_next[-1,10:14]
        self._base_acceleration = x_dot[7:10]
        self._arm_acceleration = x_dot[10:14]
        
        # An episode is done iff the agent has reached the target
        terminated = np.linalg.norm(self._base_position, ord=2) < 0.1
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.log_state()

        self.t_i = self.t_i + 1
        return observation, reward, terminated, False, info


    def dynamic_model(self, x, u):
        
        ctrl_contribution = self.get_control_contribution(x,u)
        grvty_contribution = self.get_gravity_contribution(x)
        dmpng_contribution = self.get_damping_contribution(x)
        crls_contribution = self.get_coriolis_contribution(x)
        ext_force = self.get_external_force_contribution(x)


        x_dot = np.concatenate((x[7:14],
                                ctrl_contribution 
                              + grvty_contribution
                              + dmpng_contribution
                              + crls_contribution
                              + ext_force))

        #TODO comput the actual dynamics

        return x_dot

    def get_control_contribution(self, x, u):
        
        sT = np.sin(x[2])
        cT = np.cos(x[2])
        return np.array([-sT * (u[0] + u[1]),
                          cT * (u[0] + u[1]),
                          0.5 * self.l_base * (u[1] - u[0]),
                          u[2],
                          u[3],
                          u[4],
                          u[5]])      
    
    def get_gravity_contribution(self, x):
        return np.array([0, -(self.m_base + 4*self.m_link) * self.g, 0, 0, 0, 0, 0])
        
    def get_damping_contribution(self, x):
        # TODO
        return np.zeros(7)
    
    def get_coriolis_contribution(self, x):
        # TODO
        return np.zeros(7)

    def get_external_force_contribution(self, x):
        #TODO
        return np.zeros(7)

    def log_state(self):
        self.log[self.index]["base_position"][:,self.t_i] = self._base_position
        self.log[self.index]["base_velocity"][:,self.t_i] = self._base_velocity
        self.log[self.index]["base_acceleration"][:,self.t_i] = self._base_acceleration
        self.log[self.index]["arm_position"][:,self.t_i] = self._arm_position
        self.log[self.index]["arm_velocity"][:,self.t_i] = self._arm_velocity
        self.log[self.index]["arm_acceleration"][:,self.t_i] = self._arm_acceleration 

    def render(self):
        fig, ax = plt.subplots()

        # Draw the initial position of the base
        self.base = np.array([[-0.5 * self.l_base - 0.1 , -0.5 * self.l_base + 0.1, -0.5 * self.l_base,
                          -0.5 * self.l_base, 0.5 * self.l_base,
                          0.5 * self.l_base, 0.5 * self.l_base - 0.1, 0.5 * self.l_base + 0.1],
                         [0.1, 0.1, 0.1, 0.0 ,0.0 ,0.1, 0.1, 0.1]])

        base_line, = ax.plot(self.log[0]["base_position"][0,0] + self.base[0,:],
                             self.log[0]["base_position"][1,0] + self.base[1,:], color="black")

        # Draw the initial positions of the robotic arms
        self.link = np.array([[0, 0], [0, -self.l_link]])
        r1l1_rot = np.transpose(np.matmul(self.rot(self.log[0]["arm_position"][0,0]), self.link)) \
                - np.array([self.off,0])\
                + self.log[0]["base_position"][0:2,0]
        r1l1_line, = ax.plot(r1l1_rot[:,0], r1l1_rot[:,1], color="black")

        r1l2_rot = np.transpose(np.matmul(self.rot(self.log[0]["arm_position"][1,0]),
                                          np.matmul(self.rot(self.log[0]["arm_position"][0,0]),
                                                    self.link))
                   )\
                + r1l1_rot[1,:]
        r1l2_line, = ax.plot(r1l2_rot[:,0], r1l2_rot[:,1], color="black")

        r2l1_rot = np.transpose(np.matmul(self.rot(self.log[0]["arm_position"][2,0]), self.link)) \
                + np.array([self.off,0])\
                + self.log[0]["base_position"][0:2,0]
        r2l1_line, = ax.plot(r2l1_rot[:,0], r2l1_rot[:,1], color="black")

        r2l2_rot = np.transpose(np.matmul(self.rot(self.log[0]["arm_position"][3,0]),
                                          np.matmul(self.rot(self.log[0]["arm_position"][2,0]),
                                                    self.link))
                   )\
                + r2l1_rot[1,:]
        r2l2_line, = ax.plot(r2l2_rot[:,0], r2l2_rot[:,1], color="black")


        # Draw the robotic arm joints
        joints = np.array([r1l1_rot[0,:], r1l1_rot[1,:],
                           r2l1_rot[0,:], r2l1_rot[1,:]])
        joints_line, = ax.plot(joints[:,0], joints[:,1], markersize=3, marker="o", linewidth=0,color='black')

        # Draw the inital configuration of the robotic arm
        ax.set_xlim([-4,4])
        ax.set_ylim([-1,5])
        ax.set_aspect('equal')

        ax.add_patch(Rectangle((-4,-1),8,1,color='gray'))

        func = lambda i: self.animate(i, base_line, r1l1_line, r1l2_line, r2l1_line, r2l2_line, joints_line)
        vid = ani.FuncAnimation(fig, func, frames=150, interval=self.ts * 1000,
                                blit=True)

        plt.show()



    def animate(self, i, 
                base_line, r1l1_line, r1l2_line, r2l1_line, r2l2_line, joints_line):
        
        base_rot = self.rot(self.log[0]["base_position"][2,i])
        base = base_rot @ self.base

        base_line.set_data(self.log[0]["base_position"][0,i] + base[0,:],
                      self.log[0]["base_position"][1,i] + base[1,:])

        r1l1_rot = np.transpose(self.rot(self.log[0]["arm_position"][0,i]) @ base_rot @ self.link) \
                - self.rot(self.log[0]["base_position"][2,i]) @ np.array([self.off,0])\
                + self.log[0]["base_position"][0:2,i]
        r1l1_line.set_data(r1l1_rot[:,0], r1l1_rot[:,1])

        r1l2_rot = np.transpose(self.rot(self.log[0]["arm_position"][1,i]) @ (
                                         self.rot(self.log[0]["arm_position"][0,i]) @
                                         base_rot @ self.link))\
                + r1l1_rot[1,:]
        r1l2_line.set_data(r1l2_rot[:,0], r1l2_rot[:,1])

        r2l1_rot = np.transpose(self.rot(self.log[0]["arm_position"][2,i]) @ base_rot @ self.link) \
                + self.rot(self.log[0]["base_position"][2,i]) @ np.array([self.off,0])\
                + self.log[0]["base_position"][0:2,i]
        r2l1_line.set_data(r2l1_rot[:,0], r2l1_rot[:,1])

        r2l2_rot = np.transpose(self.rot(self.log[0]["arm_position"][3,i]) @
                                          (self.rot(self.log[0]["arm_position"][2,i]) @
                                                    base_rot @ self.link)
                   )\
                + r2l1_rot[1,:]
        r2l2_line.set_data(r2l2_rot[:,0], r2l2_rot[:,1])


        # Draw the robotic arm joints
        joints = np.array([r1l1_rot[0,:], r1l1_rot[1,:],
                           r2l1_rot[0,:], r2l1_rot[1,:]])
        joints_line.set_data(joints[:,0], joints[:,1])

        return base_line, r1l1_line, r1l2_line, r2l1_line, r2l2_line, joints_line


    def close(self):
        if self.render_mode == "human":
            self.render()

    def rot(self, theta):
        st = np.sin(theta)
        ct = np.cos(theta)
        return np.array([[ct, -st],[st, ct]])


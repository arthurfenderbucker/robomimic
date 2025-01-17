"""
A collection of useful environment wrappers.
"""
from copy import deepcopy
import textwrap
import numpy as np
from collections import deque

import robomimic.envs.env_base as EB


class EnvWrapper(object):
    """
    Base class for all environment wrappers in robomimic.
    """
    def __init__(self, env):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
        """
        assert isinstance(env, EB.EnvBase) or isinstance(env, EnvWrapper)
        self.env = env

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        """
        Utility function that checks if we're accidentally trying to double wrap an env
        Raises:
            Exception: [Double wrapping env]
        """
        env = self.env
        while True:
            if isinstance(env, EnvWrapper):
                if env.class_name() == self.class_name():
                    raise Exception(
                        "Attempted to double wrap with Wrapper: {}".format(
                            self.__class__.__name__
                        )
                    )
                env = env.env
            else:
                break

    @property
    def unwrapped(self):
        """
        Grabs unwrapped environment

        Returns:
            env (EnvBase instance): Unwrapped environment
        """
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

    def _to_string(self):
        """
        Subclasses should override this method to print out info about the 
        wrapper (such as arguments passed to it).
        """
        return ''

    def __repr__(self):
        """Pretty print environment."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\nenv={}".format(self.env), indent)
        msg = header + '(' + msg + '\n)'
        return msg

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if id(result) == id(self.env):
                    return self
                return result

            return hooked
        else:
            return orig_attr


class FrameStackWrapper(EnvWrapper):
    """
    Wrapper for frame stacking observations during rollouts. The agent
    receives a sequence of past observations instead of a single observation
    when it calls @env.reset, @env.reset_to, or @env.step in the rollout loop.
    """
    def __init__(self, env, num_frames):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
            num_frames (int): number of past observations (including current observation)
                to stack together. Must be greater than 1 (otherwise this wrapper would
                be a no-op).
        """
        assert num_frames > 1, "error: FrameStackWrapper must have num_frames > 1 but got num_frames of {}".format(num_frames)

        super(FrameStackWrapper, self).__init__(env=env)
        self.num_frames = num_frames

        # keep track of last @num_frames observations for each obs key
        self.obs_history = None

    def _get_initial_obs_history(self, init_obs):
        """
        Helper method to get observation history from the initial observation, by
        repeating it.

        Returns:
            obs_history (dict): a deque for each observation key, with an extra
                leading dimension of 1 for each key (for easy concatenation later)
        """
        obs_history = {}
        for k in init_obs:
            obs_history[k] = deque(
                [init_obs[k][None] for _ in range(self.num_frames)], 
                maxlen=self.num_frames,
            )
        return obs_history

    def _get_stacked_obs_from_history(self):
        """
        Helper method to convert internal variable @self.obs_history to a 
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        return { k : np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history }

    def cache_obs_history(self):
        self.obs_history_cache = deepcopy(self.obs_history)

    def uncache_obs_history(self):
        self.obs_history = self.obs_history_cache
        self.obs_history_cache = None

    def reset(self):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of 
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        obs = self.env.reset()
        self.timestep = 0  # always zero regardless of timestep type
        self.update_obs(obs, reset=True)
        self.obs_history = self._get_initial_obs_history(init_obs=obs)
        return self._get_stacked_obs_from_history()

    def reset_to(self, state):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of 
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        obs = self.env.reset_to(state)
        self.timestep = 0  # always zero regardless of timestep type
        self.update_obs(obs, reset=True)
        self.obs_history = self._get_initial_obs_history(init_obs=obs)
        return self._get_stacked_obs_from_history()

    def step(self, action):
        """
        Modify to update the internal frame history and return frame stacked observation,
        which will have leading dimension @self.num_frames for each key.

        Args:
            action (np.array): action to take

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        self.update_obs(obs, action=action, reset=False)
        # update frame history
        for k in obs:
            # make sure to have leading dim of 1 for easy concatenation
            self.obs_history[k].append(obs[k][None])
        obs_ret = self._get_stacked_obs_from_history()
        return obs_ret, r, done, info

    def update_obs(self, obs, action=None, reset=False):
        obs["timesteps"] = np.array([self.timestep])
        
        if reset:
            obs["actions"] = np.zeros(self.env.action_dimension)
        else:
            self.timestep += 1
            obs["actions"] = action[: self.env.action_dimension]

    def _to_string(self):
        """Info to pretty print."""
        return "num_frames={}".format(self.num_frames)
    





from motor_cortex.common.guidance_wrapper import GuidanceWrapper, GuidanceArguments
import matplotlib.pyplot as plt

class RedisWrapper(EnvWrapper):
    """
    Wrapper for relaying observations during rollouts to a redis server. The agent
    waits for the server to send back acknoledgement before acting in the environment.
    """
    def __init__(self, env, wait_ak = False):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
            wait_ak (bool): whether to wait for acknoledgement from the server before acting
        """
        
        super(RedisWrapper, self).__init__(env=env)

        print("======================= initializing redis wrapper =======================")
        self.wait_ak = wait_ak
        guidance_args = GuidanceArguments().parse_args(known_only=True)
        self.guidance_wrapper = GuidanceWrapper(guidance_args)
        self.rollouts_per_demo = self.guidance_wrapper.rollouts_per_demo

        self.main_camera_name = "robot0_agentview_right"
        self.main_camera_index = 0
        # if self.guidance_wrapper.pub_interval > 0:
            # self.action_mode.arm_action_mode.set_callable_each_step(
            #     self.guidance_wrapper.get_obs_relay_func(self.get_obs_action))

    def _visualize_pc(self, pc, rgb_image, depth_map):
        """
        Helper function to visualize the point cloud with colors.
        """
        colors = rgb_image.reshape(-1, 3)
        if not hasattr(self, 'fig'):
            self.fig = plt.figure(figsize=(15, 5))
            self.ax = self.fig.add_subplot(131, projection='3d')
            self.scatter = self.ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_box_aspect([1, 1, 1])  # Set the aspect ratio to be equal

            # Initialize quivers for frame of reference
            self.quiver_x = self.ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
            self.quiver_y = self.ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
            self.quiver_z = self.ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')
            self.ax.legend()

            self.azim = 0
            # Set initial view
            self.ax.view_init(elev=0, azim=self.azim)

            # Add RGB image subplot
            self.ax_rgb = self.fig.add_subplot(132)
            self.rgb_image_plot = self.ax_rgb.imshow(rgb_image)
            self.ax_rgb.set_title('RGB Image')
            self.ax_rgb.axis('off')

            # Add Depth image subplot
            self.ax_depth = self.fig.add_subplot(133)
            self.depth_image_plot = self.ax_depth.imshow(depth_map, cmap='gray')
            self.ax_depth.set_title('Depth Image')
            self.ax_depth.axis('off')

            plt.ion()
            plt.show()
        else:
            self.scatter._offsets3d = (pc[:, 0], pc[:, 1], pc[:, 2])
            self.scatter.set_color(colors)
            self.ax.set_xlim([pc[:, 0].min(), pc[:, 0].max()])
            self.ax.set_ylim([pc[:, 1].min(), pc[:, 1].max()])
            self.ax.set_zlim([pc[:, 2].min(), pc[:, 2].max()])
            self.ax.set_box_aspect([1, 1, 1])  # Set the aspect ratio to be equal

            # Update quivers for frame of reference
            max_range = np.array([pc[:, 0].max() - pc[:, 0].min(), pc[:, 1].max() - pc[:, 1].min(), pc[:, 2].max() - pc[:, 2].min()]).max() / 2.0
            mid_x = (pc[:, 0].max() + pc[:, 0].min()) * 0.5
            mid_y = (pc[:, 1].max() + pc[:, 1].min()) * 0.5
            mid_z = (pc[:, 2].max() + pc[:, 2].min()) * 0.5
            self.quiver_x.remove()
            self.quiver_y.remove()
            self.quiver_z.remove()
            self.quiver_x = self.ax.quiver(mid_x, mid_y, mid_z, max_range, 0, 0, color='r', label='X')
            self.quiver_y = self.ax.quiver(mid_x, mid_y, mid_z, 0, max_range, 0, color='g', label='Y')
            self.quiver_z = self.ax.quiver(mid_x, mid_y, mid_z, 0, 0, max_range, color='b', label='Z')

            self.azim -= 2
            self.ax.view_init(elev=0, azim=self.azim)

            # Update RGB image
            self.rgb_image_plot.set_data(rgb_image)

            # Update Depth image
            self.depth_image_plot.set_data(depth_map)

            plt.draw()
            plt.pause(0.001)



    def _get_point_cloud(self, obs, depth_map):
        """
        Helper function to compute the point cloud from the observation.
        """
        height=self.base_env.camera_heights[self.main_camera_index]
        width=self.base_env.camera_widths[self.main_camera_index]
        
        
        # get camera matrices
        intrinsic_matrix = self.get_camera_intrinsic_matrix(
            camera_name=f"{self.main_camera_name}",
            camera_height=height,
            camera_width=width
        )
        extrinsic_matrix = self.get_camera_extrinsic_matrix(
            camera_name=f"{self.main_camera_name}",
        )
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Flatten and stack pixel coordinates
        u_flat = u.flatten()
        v_flat = v.flatten()
        z_flat = depth_map.flatten()

        # Discard points with zero or invalid depth
        valid = z_flat > 0
        u_flat = u_flat[valid]
        v_flat = v_flat[valid]
        z_flat = z_flat[valid]

        # Intrinsic matrix inverse
        intrinsic_inv = np.linalg.inv(intrinsic_matrix)

        # Convert (u, v, 1) to camera coordinates
        pixel_coords = np.stack([u_flat, v_flat, np.ones_like(u_flat)], axis=1)
        camera_coords = (pixel_coords * z_flat[:, None]) @ intrinsic_inv.T

        # Add homogeneous coordinate for transformation
        camera_coords_hom = np.hstack([camera_coords, np.ones((camera_coords.shape[0], 1))])

        # Transform to world coordinates using the extrinsic matrix
        points_world = (camera_coords_hom @ extrinsic_matrix.T)[:,:3]
        
        return points_world


    def reset(self):
        obs = self.env.reset()
        print(dir(self.env))
        self.ep_meta = self.base_env.get_ep_meta()
        task_str = self.base_env.__class__.__name__
        task_instr = self.ep_meta['task']
        print("="*50)
        print(task_str)
        print(task_instr)
        print("="*50)
        variation = self.ep_meta['layout_id']
        demo_id = self.ep_meta['style_id']
        rollout = 0 #TODO: chage for multiple interaction in the same env
        
        # self.guidance_wrapper.reset_seeds(self.seed)
        self.guidance_wrapper.set_experiment(task_str, variation, demo_id, rollout)
        self.guidance_wrapper.set_task_description(task_instr)
        self.relay_obs(obs)
        return obs

    def reset_to(self, state):
        obs = self.env.reset_to(state)
        self.relay_obs(obs)
        return obs

    def step(self, action):
        
        obs_ret, r, done, info = self.env.step(action)
        self.relay_obs(obs_ret)
        return obs_ret, r, done, info

    def relay_obs(self, obs, action=None, reset=False):
        """overwriting the update_obs method to relay the observations to the redis server"""

        meta = self.guidance_wrapper.get_obs_meta(obs)

        depth_map = obs["{}_depth".format(self.main_camera_name)]
        try:
            # unnormalized depth map
            depth_map = self.get_real_depth_map(depth_map=depth_map)
        except:
            pass
        
        if len(depth_map.shape) == 4:
            depth_map = depth_map[-1,:,:,:]

        depth_map = np.transpose(depth_map, (1,2,0))

        pc = self._get_point_cloud(obs, depth_map)
        
        print("OBSERVATION KEYS")
        print(obs.keys())
        # rgb = obs["camera_rgb"]
        cam = "front"
        rgb = obs["{}_image".format(self.main_camera_name)]
        if len(rgb.shape) == 4:
            rgb = rgb[-1,:,:,:]
        rgb = np.transpose(rgb, (1,2,0))
        self._visualize_pc(pc, rgb, depth_map)

        self.guidance_wrapper.transmit(rgb,f"{cam}_rgb", meta=meta)
        self.guidance_wrapper.transmit(depth_map,f"{cam}_depth", meta=meta)
        self.guidance_wrapper.transmit(pc,f"{cam}_point_cloud", meta=meta)  

    def close(self):
        env = self.env
        while True:
            if isinstance(env, EnvWrapper) and hasattr(env, "env"):
                env = env.env
                if hasattr(env, "close"):
                    env.close()
                    break
            else:
                break

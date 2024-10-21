import numpy as np
import os
import torch
import math
from isaacgym import gymtorch
from isaacgym import gymapi
from pynput import keyboard

from isaacgymenvs.utils.torch_jit_utils import quat_to_euler, to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse, get_euler_xyz
from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict

class Centaur(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.wheel_action_scale = self.cfg["env"]["control"]["wheelActionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["height"] = self.cfg["env"]["learn"]["heightRewardScale"]
        self.rew_scales["pitch"] = self.cfg["env"]["learn"]["pitchRewardScale"]
        self.rew_scales["sync"] = self.cfg["env"]["learn"]["syncRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state
        
        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 47
        self.cfg["env"]["numActions"] = 12

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        # self.Kp = self.cfg["env"]["control"]["stiffness"]
        # self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.commands = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_x = self.commands.view(self.num_envs, 2)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 2)[..., 1]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle
        
        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        if cfg["env"]["test"]:
            def on_press(key):
                try:
                    if key.char == 'z':  # Quit simulation on 'q' key press
                        exit()
                    elif key.char == 'w':
                        self.command_x_range = [.9,1]
                        self.command_yaw_range = [0,0]
                        print('w')
                    elif key.char == 's':
                        self.command_x_range = [-1,-.9]
                        self.command_yaw_range = [0,0]
                        print('s')
                    elif key.char == 'd':
                        self.command_x_range = [0,0]
                        self.command_yaw_range = [-1,-0.9]
                        print('d')
                    elif key.char == 'a':
                        self.command_x_range = [0,0]
                        self.command_yaw_range = [0.9,1]
                        print('a')
                    elif key.char == 'q':
                        self.command_x_range = [0.9,1]
                        self.command_yaw_range = [0.3,0.4]
                        print('q')
                    elif key.char == 'e':
                        self.command_x_range = [0.9,1]
                        self.command_yaw_range = [-0.4,-0.3]
                        print('e')
                except AttributeError:
                    # Handle special keys here if needed
                    pass
                self.reset_idx(torch.arange(0, self.num_envs, device=self.device))

            # Start keyboard listener
            listener = keyboard.Listener(on_press=on_press)
            listener.start()

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
            self.up_axis_idx = 2 # Z
            self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
            self._create_ground_plane()
            self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

            # If randomizing, apply once immediately on startup before the fist sim step
            if self.randomize:
                self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/centaur/urdf/centaur.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        # asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        centaur_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(centaur_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(centaur_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(centaur_asset)
        self.dof_names = self.gym.get_asset_dof_names(centaur_asset)
        extremity_name = "wheel" if asset_options.collapse_fixed_joints else "FOOT"
        feet_names = [s for s in body_names if extremity_name in s]
        print(feet_names)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = ['base_link', 'head_pan_Link', 'head_tilt_Link', 'l_sho_pitch_Link',
                       'l_sho_roll_Link', 'l_el_Link', 'l_wrist_Link', 'l_index_base_Link',
                         'l_index_middle_Link', 'l_index_tip_Link', 'l_mid_base_Link',
                           'l_mid_middle_Link', 'l_mid_tip_Link', 'l_thumb_base_Link',
                             'l_thumb_middle_Link', 'l_thumb_tip_Link',
                               'motor_left_link_1', 'leg_left_link_1', 'leg2_left_link_1',
                                  'motor_right_link_1', 'leg_right_link_1',
                                   'leg2_right_link_1', 'r_sho_pitch_Link',
                                     'r_sho_roll_Link', 'r_el_Link', 'r_wrist_Link', 'r_index_base_Link',
                                       'r_index_middle_Link', 'r_index_tip_Link', 'r_mid_base_Link',
                                         'r_mid_middle_Link', 'r_mid_tip_Link', 'r_thumb_base_Link',
                                           'r_thumb_middle_Link', 'r_thumb_tip_Link']
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0
        
        dof_props = self.gym.get_asset_dof_properties(centaur_asset)
        dof_dict = self.gym.get_asset_dof_dict(centaur_asset)

        self.shoPitchIdx = []
        self.shoRollIdx = []
        self.wheelIdx = []
        self.kneeIdx = []
        self.hipIdx = []
        self.elIdx = []
        
        for name,idx in dof_dict.items():
            dof_props['driveMode'][idx] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][idx] = self.cfg["env"]["control"]["stiffness"][name]
            dof_props['damping'][idx] = self.cfg["env"]["control"]["damping"][name]

            if name == 'l4' or name == 'r4':
                dof_props['driveMode'][idx] = gymapi.DOF_MODE_VEL
                self.wheelIdx.append(idx)

            elif name == 'l3' or name == 'r3':
                self.kneeIdx.append(idx)

            elif name == 'l1' or name == 'r1':
                self.hipIdx.append(idx)

            elif name == 'r_sho_roll' or name == 'l_sho_roll':
                self.shoRollIdx.append(idx)

            elif name == 'r_el' or name == 'l_el':
                self.elIdx.append(idx)
            
            elif name == 'r_sho_pitch' or name == 'l_sho_pitch':
                self.shoPitchIdx.append(idx)

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.centaur_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            centaur_handle = self.gym.create_actor(env_ptr, centaur_asset, start_pose, "centaur", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, centaur_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, centaur_handle)
            self.envs.append(env_ptr)
            self.centaur_handles.append(centaur_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.centaur_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.centaur_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.centaur_handles[0], "lower_base_link")

    # [[[[[[[[[[[[[[[   ACTIONS  ]]]]]]]]]]]]]]]
    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        # print(self.actions)
        # Wheel action Scale
        for i in self.wheelIdx:
            self.actions[:,i] *= self.wheel_action_scale

        # Fixed Knee Joint 
        for i in self.kneeIdx:
            self.actions[:,i] = 0

        # Sync Hip Joints
        self.actions[:,self.hipIdx[0]] = 0
        self.actions[:,self.hipIdx[1]] = 0

        # Arm movements
        for i in self.elIdx:
            self.actions[:,i] = 0
        
        self.actions[:,self.shoRollIdx[0]] = 0
        self.actions[:,self.shoRollIdx[1]] = 0
        self.actions[:,self.shoPitchIdx[0]] = 0
        self.actions[:,self.shoPitchIdx[1]] = 0

        targets = (torch.tensor(self.action_scale, device=self.device) * torch.tensor(self.actions, device=self.device)) + torch.tensor(self.default_dof_pos, device=self.device)
        
        # SET POS and VEL targets
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor((targets)))
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor((targets)))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_centaur_reward(
            # tensors
            self.root_states,
            self.commands,
            self.torques,
            self.actions,
            self.contact_forces,
            self.knee_indices,
            self.progress_buf,
            # Dict
            self.rew_scales,
            # other
            self.base_index,
            self.max_episode_length,
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_centaur_observations(  # tensors
                                                        self.root_states,
                                                        self.commands,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.actions,
                                                        # scales
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale
        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
 
        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # print(env_ids)




#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_centaur_observations(root_states,
                                commands,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale
                                ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> Tensor
    # Base Roll pitch Yaw
    base_quat = root_states[:, 3:7]
    base_euler = quat_to_euler(base_quat)

    # Base Coordinates
    base_pos = root_states[:, :3]
    base_height = base_pos[:, 2]
    # print(f'EULER {(base_euler[0,2])}')
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    
    # print('--------------')
    # (base_ang_vel[0,1]) PITCH
    # print((base_ang_vel[0,1])*180/3.14)
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    commands_scaled = commands*torch.tensor([lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     base_euler,
                     commands_scaled,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     actions
                     ), dim=-1)
    
    # obs = torch.cat((base_lin_vel,
    #                  base_ang_vel,
    #                  projected_gravity,
    #                  commands_scaled,
    #                  dof_pos_scaled,
    #                  dof_vel*dof_vel_scale,
    #                  actions
    #                  ), dim=-1)
    # print(f'OBSERVATION" {obs}')
    return obs

# [[[[[[[[[[[[[[[Here the enviornments to reset are decided]]]]]]]]]]]]]]]
# [[[[[[[[[[[[[[[{{{{{{{{{{{{{{{{{{REWARDs}}}}}}}}}}}}}}}}}}]]]]]]]]]]]]]]]
@torch.jit.script
def compute_centaur_reward(
    # tensors
    root_states,
    commands,
    torques,
    actions,
    contact_forces,
    knee_indices,
    episode_lengths,
    # Dict
    rew_scales,
    # other
    base_index,
    max_episode_length
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int) -> Tuple[Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    
    # NEW Base Coordinates
    base_pos = root_states[:, :3]
    base_height = base_pos[:, 2]

    base_quat = root_states[:, 3:7]
    # NEW
    base_euler = quat_to_euler(base_quat)

    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :1] - base_lin_vel[:, :1]), dim=1)
    ang_vel_error = torch.square(commands[:, 1] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]

    # print(f'base_lin_vel[0, :]: {base_ang_vel[0, 2]}')
    # print(f'base_lin_vel[0, :2]: {base_lin_vel[0, :2]}')
    # print(f'lin_vel_error[0]: {commands[0, 1] - base_ang_vel[0, 2]}')
    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]
    # print(f'Reward: T {rew_torque[0]}, L {rew_lin_vel_xy[0]}, A {rew_ang_vel_z[0]}')

    # height penalty
    # rew_height = base_height * rew_scales["height"]

    # Pitch Reward
    rew_pitch = torch.square(base_euler[:,1])
    rew_pitch[rew_pitch < 0.99] = 0
    rew_pitch = rew_pitch * rew_scales["pitch"]

    # Non Sync legs penalty
    # sync_error = torch.square(actions[:, 0] - actions[:, 3])
    # sync_error += torch.square(actions[:, 1] - actions[:, 4])
    # rew_sync = torch.exp(-sync_error) * rew_scales["sync"]
    # print(rew_sync)

    # Total Reward
    total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque + rew_pitch
    total_reward = torch.clip(total_reward, 0., None)

    # reset agents
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out
    
    return total_reward.detach(), reset

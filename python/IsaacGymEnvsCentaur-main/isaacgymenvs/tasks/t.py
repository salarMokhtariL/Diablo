def _create_envs(self, num_envs, spacing, num_per_row):
    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
    asset_file = "urdf/centaur/urdf/centaur.urdf"

    # Load the centaur asset
    centaur_asset_options = gymapi.AssetOptions()
    centaur_asset = self.gym.load_asset(self.sim, asset_root, asset_file, centaur_asset_options)

    # Create environments in a loop
    for i in range(num_envs):
        # Create an environment
        env_ptr = self.gym.create_env(self.sim, gymapi.Vec3(-spacing, -spacing, 0), gymapi.Vec3(spacing, spacing, 0), num_per_row)
        
        # Create the centaur actor in the environment
        start_pose = gymapi.Transform()  # You might need to set a specific starting position
        centaur_handle = self.gym.create_actor(env_ptr, centaur_asset, start_pose, "centaur", i, 1, 0)

        # Create the cube in the environment
        self._create_cube(env_ptr)  # This adds the cube to the environment

        # Set properties for the centaur actor
        dof_props = self.gym.get_actor_dof_properties(env_ptr, centaur_handle)
        self.gym.set_actor_dof_properties(env_ptr, centaur_handle, dof_props)

        # Store references for later use
        self.envs.append(env_ptr)
        self.centaur_handles.append(centaur_handle)
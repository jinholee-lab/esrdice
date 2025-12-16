import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import gym
from gym import spaces
from ant_env import AntEnv
try:
    import mujoco_py
except ImportError as e:
    print("Error: mujoco-py not found. Please install it (e.g., 'pip install mujoco-py').")
    raise e

# Maze Cell Types
WW = 'WW' # Wall
EM = 'EM' # Empty
SP = 'SP' # Start Position

# Objective 1
O1 = 'O1'
D1 = 'D1'
# Objective 2
O2 = 'O2'
D2 = 'D2'
# Objective 3
O3 = 'O3'
D3 = 'D3'
# Objective 4
O4 = 'O4'
D4 = 'D4'

# Colors for goals
COLORS = [
    (1, 0, 0, 0.7), # Red
    (0, 1, 0, 0.7), # Green
    (0, 0, 1, 0.7), # Blue
    (1, 1, 0, 0.7), # Yellow
]

MAX_GOALS = len(COLORS)

MAP_V1 = [
    [WW, WW, WW, WW, WW, WW],
    [WW, SP, EM, WW, O1, WW],
    [WW, EM, EM, EM, EM, WW],
    [WW, WW, EM, EM, D1, WW],
    [WW, O2, EM, D2, WW, WW],
    [WW, WW, WW, WW, WW, WW],
]

class FairMazeEnv(gym.Env):
    """
    A new, standalone maze environment (based on D4RL's maze_env.py)
    that supports multi-objective, stateful goals defined directly
    in the maze_map (e.g., 'O1', 'D1', 'O2', 'D2').
    
    This class must be inherited by a specific locomotion env, e.g.:
    class FairAntMaze(FairMazeEnv, AntEnv):
        LOCOMOTION_ENV = AntEnv
    """
    
    LOCOMOTION_ENV = None # Must be specified by child class.

    def __init__(
        self,
        maze_map,
        maze_size_scaling,
        maze_height=0.5,
        manual_collision=False,
        non_zero_reset=False,
        agent_base_color=(0.2, 0.2, 0.8, 1.0),
        # agent_base_color=(1, 0, 0, 0.7),
        touch_threshold=0.5,
        *args,
        **kwargs):

        if self.LOCOMOTION_ENV is None:
            raise ValueError('LOCOMOTION_ENV is unspecified.')

        self.base_agent_color = np.array(agent_base_color, dtype=np.float32)
        self.touch_threshold = touch_threshold
        self._maze_map = maze_map
        self._maze_height = maze_height
        self._maze_size_scaling = maze_size_scaling
        self._manual_collision = manual_collision
        self.current_step = 0
                
        self.reward_dim = self._find_number_of_objective_pair(maze_map)
        self.goal_state = ['origin'] * self.reward_dim
        self._offscreen_viewer = None

        
        # Find robot start position (torso offset)
        self._init_torso_x, self._init_torso_y = self._find_robot()
        print(f"Robot start position (x,y): {self._init_torso_x}, {self._init_torso_y}")
        # --- 1. Parse maze_map to find Robot Start and Goals ---
        self.goals = []
        self._goal_map = {} # Temp dict to pair 'O1' and 'D1'

        # First pass: find all goals and their world coords
        for r in range(len(self._maze_map)):
            for c in range(len(self._maze_map[0])):
                cell = self._maze_map[r][c]
                
                # Check for goal cell, e.g., 'O1', 'D2'
                if isinstance(cell, str) and (cell.startswith('O') or cell.startswith('D')):
                    # --- UPDATED PARSING ---
                    goal_type = cell[0]
                    goal_id = cell[1:] # e.g., '1', '2'
                    
                    if not goal_id.isdigit():
                        raise ValueError(f"Invalid goal cell: {cell}. ID must be numeric.")
                    
                    if goal_id not in self._goal_map:
                        self._goal_map[goal_id] = {}
                    # --- END UPDATED PARSING ---
                        
                    # Convert grid (r,c) to world (x,y)
                    world_x, world_y = self._rowcol_to_xy((r, c))
                    print(f"Found goal {cell} at world coords: ({world_x}, {world_y})")
                    
                    if goal_type == 'O':
                        self._goal_map[goal_id]['origin'] = (world_x, world_y)
                    elif goal_type == 'D':
                        self._goal_map[goal_id]['dest'] = (world_x, world_y)


        # Second pass: build self.goals list with paired origins/destinations
        sorted_goal_ids = sorted(self._goal_map.keys())
        for i, goal_id in enumerate(sorted_goal_ids):
            goal_pair = self._goal_map[goal_id]
            if 'origin' not in goal_pair or 'dest' not in goal_pair:
                raise ValueError(f"Goal ID {goal_id} is incomplete (missing O or D).")
            
            goal_pair['color'] = np.array(COLORS[i % len(COLORS)])
            goal_pair['origin'] = np.array(goal_pair['origin'])
            goal_pair['dest'] = np.array(goal_pair['dest'])
            self.goals.append(goal_pair)
            
        self.reward_dim = len(self.goals)

        # --- 2. Build the XML ---
        xml_path = self.LOCOMOTION_ENV.FILE
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        # Add Wall Blocks
        for i in range(len(self._maze_map)):
            for j in range(len(self._maze_map[0])):
                struct = self._maze_map[i][j]
                # --- UPDATED WALL CHECK ---
                if struct == WW:  # Unmovable block.
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * self._maze_size_scaling - self._init_torso_x,
                                        i * self._maze_size_scaling - self._init_torso_y,
                                        self._maze_height / 2 * self._maze_size_scaling),
                        size="%f %f %f" % (0.5 * self._maze_size_scaling,
                                         0.5 * self._maze_size_scaling,
                                         self._maze_height / 2 * self._maze_size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.7 0.5 0.3 1.0",
                    )
        
        # Add "Fairness" Goal Sites
        for i, goal_pair in enumerate(self.goals):
            o_pos = goal_pair['origin']
            d_pos = goal_pair['dest']
            color_val = goal_pair['color']
            color_str = f"{color_val[0]} {color_val[1]} {color_val[2]} {color_val[3]}"
            
            body_z_pos = self._maze_height / 2 * self._maze_size_scaling
            
            # Box Origin
            origin_body = ET.SubElement(
                worldbody, "site",
                name=f"goal_{i}_origin",
                type="box",
                pos=f"{o_pos[0]} {o_pos[1]} {body_z_pos}",
                size=f"{self.touch_threshold / 2} {self.touch_threshold / 2} {body_z_pos}",
                rgba=color_str
            )

            # Cylinder Destination
            ET.SubElement(
                worldbody, "site",
                name=f"goal_{i}_dest", type="cylinder",
                pos=f"{d_pos[0]} {d_pos[1]} {body_z_pos}",
                size=f"{self.touch_threshold / 2} {body_z_pos}",
                rgba=color_str
            )
            
        # --- 3. Initialize the Locomotion Env with new XML ---
        _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)

        # This is the key: the class itself *is* the locomotion env
        self.LOCOMOTION_ENV.__init__(
            self, *args, file_path=file_path, non_zero_reset=non_zero_reset, **kwargs)
        
        # --- 5. Set up Gym Spaces ---
        # Get base obs space from the locomotion env
        base_obs = self.LOCOMOTION_ENV._get_obs(self)
        base_obs_dim = base_obs.shape[0]
        
        # Add our goal states to the observation
        obs_dim = base_obs_dim + self.reward_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Our reward is now a vector
        self.reward_space = spaces.Box(
            low=0, high=1, shape=(self.reward_dim,), dtype=np.float32
        )
                
    def _get_obs(self):
        """Returns the agent's base observation concatenated with our
        stateful goal vector."""
        
        # Get base observation
        base_obs = self.LOCOMOTION_ENV._get_obs(self)
        
        # # Get robot global position
        # robot_xy = self.get_xy()
        
        # Build goal state vector
        goal_state_vec = np.array(
            [0 if s == 'origin' else 1 for s in self.goal_state], 
            dtype=np.float32
        )
        # obs = np.concatenate([robot_xy, base_obs, goal_state_vec])
        obs = np.concatenate([base_obs, goal_state_vec])
        return obs

    def step(self, action):
        """Runs the locomotion step, then applies our stateful reward logic."""
        self.current_step += 1
        
        # 1. Run the base locomotion step, checking for manual collision
        if self._manual_collision:
            old_pos = self.get_xy()
            _, _, done, info = self.LOCOMOTION_ENV.step(self, action)
            new_pos = self.get_xy()
            if self._is_in_collision(new_pos):
                self.set_xy(old_pos)
        else:
            _, _, done, info = self.LOCOMOTION_ENV.step(self, action)
        
        pos = self.get_xy()
        
        # 2. Apply our fairness logic
        rewards = np.zeros(self.reward_dim, dtype=np.float32)
        newly_armed_goal_index = -1
        
        # First, check for origin touches
        for i in range(self.reward_dim):
            goal_pair = self.goals[i]
            dist_origin = np.linalg.norm(pos - goal_pair['origin'])
            
            if self.goal_state[i] == 'origin' and dist_origin < self.touch_threshold:
                newly_armed_goal_index = i
                break # Mutual exclusion
        
        if newly_armed_goal_index != -1:
            self.goal_state = ['origin'] * self.reward_dim
            self.goal_state[newly_armed_goal_index] = 'dest'
        
        # Second, check for destination touches
        for i in range(self.reward_dim):
            if self.goal_state[i] == 'dest':
                goal_pair = self.goals[i]
                dist_dest = np.linalg.norm(pos - goal_pair['dest'])
                if dist_dest < self.touch_threshold:
                    self.goal_state[i] = 'origin'
                    rewards[i] = 1.0

        # 3. Update agent color
        new_color = self.base_agent_color
        armed_goals_indices = [i for i, s in enumerate(self.goal_state) if s == 'dest']
        if len(armed_goals_indices) > 0:
            new_color = self.goals[armed_goals_indices[0]]['color']

        self.sim.model.geom_rgba[self.sim.model.geom_name2id('torso_geom')] = new_color
        
        # 4. Get obs and info
        obs = self._get_obs()
        info['goal_states'] = self.goal_state.copy()

        # 5. Check for timeout
        if hasattr(self, 'max_steps') and self.current_step >= self.max_steps:
            done = True
            
        return obs, rewards, done, info

    def reset(self):
        """Resets the locomotion env, then resets our fairness state."""
        self.current_step = 0
        self.LOCOMOTION_ENV.reset_model(self)
        self.goal_state = ['origin'] * self.reward_dim
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('torso_geom')] = self.base_agent_color
        return self._get_obs()
    
    def close(self):
        self.LOCOMOTION_ENV.close(self)

    # def render(self, *args, **kwargs):
    #     return self.LOCOMOTION_ENV.render(self, *args, **kwargs)
    def _set_up_camera(self, cam):
        # Top-down-ish view
        cam.azimuth = 90        # rotate around z (left/right)
        cam.elevation = -50     # -90 = straight down, -80 is slightly tilted
        cam.distance = 60.0      # zoom: smaller = closer
        xlen = len(self._maze_map[0]) * self._maze_size_scaling
        ylen = len(self._maze_map) * self._maze_size_scaling
        cam.lookat[:] = [0.0 + xlen / 2.0,
                         0.0 + ylen / 2.0,
                         0.0]  # center of the maze (your grid is centered at (0,0))

    def render(self, mode='human', width=500, height=500):
        """Renders the environment."""
        
        if mode == 'rgb_array':
            # Use the offscreen renderer
            if self._offscreen_viewer is None:
                # Initialize offscreen renderer
                self._offscreen_viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
                self._set_up_camera(self._offscreen_viewer.cam)
            
            # Render the scene
            self._offscreen_viewer.render(width=width, height=height)
            
            # Read the pixels
            data = self._offscreen_viewer.read_pixels(width=width, height=height, depth=False)
            
            # Flip pixels to match standard image format
            return data[::-1, :, :]
            
        elif mode == 'human':
            # Use the standard MjViewer
            if self.viewer is None:
                self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer.render()
        
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    
    # --- Helper functions modified for new conventions ---
    def _find_number_of_objective_pair(self,maze_map):
        found_origins = set()
        found_destinations = set()
        
        for row in maze_map:
            for cell in row:
                if not isinstance(cell, str):
                    continue
                    
                # Check for Origin (starts with O, followed by a number)
                if cell.startswith('O') and cell[1:].isdigit():
                    obj_id = int(cell[1:])
                    found_origins.add(obj_id)
                    
                # Check for Destination (starts with D, followed by a number)
                elif cell.startswith('D') and cell[1:].isdigit():
                    obj_id = int(cell[1:])
                    found_destinations.add(obj_id)
        
        if len(found_origins) != len(found_destinations):
            print(f"Found origins: {found_origins}")
            print(f"Found destinations: {found_destinations}")
            raise ValueError("Mismatch in number of origins and destinations found.")
        
        valid_pairs = found_origins.intersection(found_destinations)
        number_of_pairs = len(valid_pairs)
        
        if number_of_pairs == 0:
            raise ValueError("No valid objective pairs (O/D) found in maze map.")
        elif number_of_pairs > MAX_GOALS:
            raise ValueError(f"Number of objective pairs ({number_of_pairs}) exceeds maximum supported ({MAX_GOALS}).")
        
        return number_of_pairs

    def _find_robot(self):
        """Finds the 'SP' cell in the maze map."""
        structure = self._maze_map
        size_scaling = self._maze_size_scaling
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                # --- UPDATED START CHECK ---
                if structure[i][j] == SP:
                    return j * size_scaling, i * size_scaling
        raise ValueError('No robot (SP) in maze specification.')

    def _is_in_collision(self, pos):
        """Checks if a given (x,y) position is inside a wall."""
        x, y = pos
        structure = self._maze_map
        size_scaling = self._maze_size_scaling
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                # --- UPDATED WALL CHECK ---
                if structure[i][j] == WW:
                    minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
                    maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
                    miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
                    maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
                    if minx <= x <= maxx and miny <= y <= maxy:
                        return True
        return False
    
    def _rowcol_to_xy(self, rowcol, add_random_noise=False):
        """Converts a grid (row, col) to a world (x,y) coordinate."""
        row, col = rowcol
        # NOTE: D4RL's convention is (j*scale - x_offset, i*scale - y_offset)
        # This means (col, row)
        x = col * self._maze_size_scaling - self._init_torso_x
        y = row * self._maze_size_scaling - self._init_torso_y
        # y = -y  # Invert y-axis to match world coordinates
        if add_random_noise:
            x = x + np.random.uniform(low=0, high=self._maze_size_scaling * 0.25)
            y = y + np.random.uniform(low=0, high=self._maze_size_scaling * 0.25)
        return (x, y)

    def _xy_to_rowcol(self, xy):
        """
        Convert world (x, y) back to integer (row, col) indices.
        Inverse of _rowcol_to_xy (ignoring noise).
        """
        x, y = xy
        # y = -y  # Invert y-axis to match grid coordinates
        col_f = (x + self._init_torso_x) / self._maze_size_scaling
        row_f = (y + self._init_torso_y) / self._maze_size_scaling

        # Use round to avoid off-by-one due to small floating errors
        col = int(round(col_f))
        row = int(round(row_f))

        return row, col
    
    # navigation policy
    def _get_best_next_rowcol(self, current_rowcol, target_rowcol):
        """
        BFS on the grid to find a shortest path and return the next cell
        on that path from current_rowcol toward target_rowcol.
        """
        current_rowcol = tuple(current_rowcol)
        target_rowcol = tuple(target_rowcol)
        if target_rowcol == current_rowcol:
            return target_rowcol

        visited = {}
        to_visit = [target_rowcol]

        while to_visit:
            next_visit = []
            for rowcol in to_visit:
                visited[rowcol] = True
                row, col = rowcol

                neighbors = [
                    (row, col - 1),  # left
                    (row, col + 1),  # right
                    (row + 1, col),  # down
                    (row - 1, col),  # up
                ]

                for nr, nc in neighbors:
                    next_rowcol = (nr, nc)

                    if next_rowcol == current_rowcol:
                        # Found the immediate predecessor on a shortest path
                        return rowcol

                    # bounds
                    if nr < 0 or nr >= len(self._maze_map):
                        continue
                    if nc < 0 or nc >= len(self._maze_map[0]):
                        continue

                    # traversable? (anything except wall)
                    cell = self._maze_map[nr][nc]
                    if cell == WW:
                        continue

                    if next_rowcol in visited:
                        continue

                    next_visit.append(next_rowcol)

            to_visit = next_visit

        raise ValueError("No path found to target.")


    def create_fair_navigation_policy(
        self,
        low_level_policy_fn,
        script_tokens=[],
        obs_to_robot= lambda obs: obs[:2], # extract robot (x,y) from obs
        ):
        """
        script_tokens: list of strings, e.g. ['O1','D1','O1','D1','O2','D2']
        """
        parsed_script = []
        for token in script_tokens:
            letter = token[0].upper()
            idx_str = token[1:]
            
            obj_id = int(idx_str)
            goal_index = obj_id - 1
            
            if goal_index < 0 or goal_index >= len(self.goals):
                raise ValueError(f"Invalid script token: {token}")
            
            phase = 'origin' if letter == 'O' else 'dest'
            parsed_script.append( (goal_index, phase) )
        
        # Build the navigation policy
        script_pos = 0
        
        def policy_fn(obs):
            nonlocal script_pos
            
            # current robot position in world coords
            robot_x, robot_y = obs_to_robot(obs)            
    
            goal_idx, phase = parsed_script[script_pos]
            target_x, target_y = self.goals[goal_idx][phase]
            
            robot_xy = np.array([robot_x, robot_y])
            target_xy = np.array([target_x, target_y])
            
            # advance script if close enough to target
            if np.linalg.norm(robot_xy - target_xy) < self.touch_threshold:
                # advance to next script position
                # if at end, loop back to start
                script_pos = (script_pos + 1) % len(parsed_script)
                goal_idx, phase = parsed_script[script_pos]
                target_x, target_y = self.goals[goal_idx][phase]
                
            # convert to rowcol for pathfinding
            robot_row, robot_col = self._xy_to_rowcol((robot_x, robot_y))
            target_row, target_col = self._xy_to_rowcol((target_x, target_y))
            
            waypoint_row, waypoint_col = self._get_best_next_rowcol(
                (robot_row, robot_col),
                (target_row, target_col)
            )
                        
            # if waypoint is the target, go directly there
            if (waypoint_row, waypoint_col) == (target_row, target_col):
                waypoint_x, waypoint_y = target_x, target_y
            # otherwise, add some noise to avoid gridlock
            else:
                waypoint_x, waypoint_y = self._rowcol_to_xy(
                    (waypoint_row, waypoint_col), add_random_noise=True
                )

            # relative goal in robot-centric coords
            goal_x = waypoint_x - robot_x
            goal_y = waypoint_y - robot_y          
            
            # print(f'Robot: (row, col)=({robot_row}, {robot_col}) : (x, y) ({robot_x}, {robot_y})')
            # print(f'Waypoint: (row, col)=({waypoint_row}, {waypoint_col}) : (x, y) ({waypoint_x}, {waypoint_y})')
            # print(f'Target: (row, col)=({target_row}, {target_col}) : (x, y) ({target_x}, {target_y})')
            
            return low_level_policy_fn(obs, (goal_x, goal_y))
           
        return policy_fn    



class FairAntMaze(FairMazeEnv, AntEnv):
    LOCOMOTION_ENV = AntEnv

    def __init__(
        self,
        maze_map=MAP_V1,
        maze_size_scaling=4.0,
        maze_height=0.5,
        manual_collision=False,
        non_zero_reset=False,
        touch_threshold=0.5,
        max_steps=1000,
        expose_all_qpos=True,
        **kwargs
    ):
        self.max_steps = max_steps
        super().__init__(
            maze_map=maze_map,
            maze_size_scaling=maze_size_scaling,
            maze_height=maze_height,
            manual_collision=manual_collision,
            non_zero_reset=non_zero_reset,
            touch_threshold=touch_threshold,
            expose_all_qpos=expose_all_qpos,
            **kwargs
        )

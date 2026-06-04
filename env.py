import math

from agent import Agent
from agent import EvasionAgent
from gym import spaces

import matplotlib
import matplotlib.pyplot as plt
import pyglet
from pyglet import shapes
import resources
import random
from matplotlib import rcParams

import numpy as np


matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14


matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'  
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'  


def relative_angle(start_point, end_point):
    "end_point   start_point "
    x_diff = end_point.state.x - start_point.state.x
    y_diff = end_point.state.y - start_point.state.y
    relative_theta = math.atan(y_diff / (x_diff + 0.0001))

    if y_diff > 0 and x_diff < 0:
        relative_theta += math.pi
    elif y_diff < 0 and x_diff < 0:
        relative_theta -= math.pi
    return relative_theta
    

def relative_angle_to_point(agent, point):
    
    x_diff = point[0] - agent.state.x
    y_diff = point[1] - agent.state.y
    relative_theta_to_point = math.atan(y_diff / (x_diff + 0.0001))

    if y_diff > 0 and x_diff < 0:
        relative_theta_to_point += math.pi
    elif y_diff < 0 and x_diff < 0:
        relative_theta_to_point -= math.pi
    return relative_theta_to_point


def relative_dis(start_point, end_point):
    return math.sqrt((start_point.state.x - end_point.state.x) ** 2 + (start_point.state.y - end_point.state.y) ** 2)



def Distance(position_1=np.array([]), position_2=np.array([])):
    dx = np.zeros([len(position_1), len(position_2)], dtype=np.float32)
    dy = np.zeros([len(position_1), len(position_2)], dtype=np.float32)
    for i in range(len(position_1)):
        for j in range(len(position_2)):
            dx[i, j] = position_1[i, 0] - position_2[j, 0]
            dy[i, j] = position_1[i, 1] - position_2[j, 1]
    distance = np.hypot(dx, dy)  
    return distance


class Env:
    viewer = True
    viewer_xy = (400, 400)
    def __init__(self, args):
        self.capture_flags = None
        self.stop_flags = None
        self.fail_flags = False
        self.num_pursuit = args.num_pursuit  
        self.num_evasion = args.num_evasion  
        self.num_obscale = args.num_obstacle  
        self.args = args  
        self.best_solution = None
        self.obs_shape_n = [(self.args.input_state_dim,)] * self.num_pursuit  
        pursuit_action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = [pursuit_action_space] * self.num_pursuit
        self.AgentsContains = {}  
        self.line_batch = pyglet.graphics.Batch()
        self.island_info = np.array([args.island_x, args.island_y], dtype=np.float32)
        self.evasions_info = np.array([0, 0, 0], dtype=np.float32)
        self.pursuits_info = np.array([0, 0, 0], dtype=np.float32)
        self.evasions_paths = [[] for _ in range(self.num_evasion)]
        self.pursuits_paths = [[] for _ in range(self.num_pursuit)]
        self.init_capture_flags()  
        self.init_stop_flags()  

        self.phase_info = {
            'current_phase': 'approach',  
            'phase_switched': False,  
            'approach_steps': 0,  
            'approach_start_info': None,  
            'approach_end_info': None,  
            'surround_start_info': None  
        }

        self.d_cap = args.d_cap

    def set_best_solution(self, best_solution):
        self.best_solution = best_solution

    def Init(self):
        for i in range(self.num_pursuit + self.num_evasion + self.num_obscale):
            if i < self.num_pursuit:
                self.AgentsContains["pursuit_" + str(i + 1)] = Agent("pursuit_" + str(i + 1), self.obs_shape_n,
                                                                     self.action_space, i, self.args)
            elif i >= self.num_pursuit and i < self.num_pursuit + self.num_evasion:
                self.AgentsContains["evasion_" + str(i + 1 - self.num_pursuit)] = EvasionAgent(
                    "evasion_" + str(i + 1 - self.num_pursuit), i, self.args)
            else:
                self.AgentsContains["obscale_" + str(i + 1 - self.num_pursuit - self.num_evasion)] = Agent(
                    "obscale" + str(i + 1 - self.num_pursuit - self.num_evasion), self.obs_shape_n,
                    self.action_space, i, self.args)

    def agents_info(self):
        agents = self.AgentsContains
        pursuits_info = []
        for j in range(self.num_pursuit):
            pursuit_x = agents["pursuit_" + str(j + 1)].state.x
            pursuit_y = agents["pursuit_" + str(j + 1)].state.y
            pursuit_theta = agents["pursuit_" + str(j + 1)].state.theta
            pursuit_info = [pursuit_x, pursuit_y, pursuit_theta]
            pursuits_info.append(pursuit_info)
        pursuits_info = np.array(pursuits_info, dtype=np.float64)

        evasions_info = []
        for j in range(self.num_evasion):
            evasion_x = agents["evasion_" + str(j + 1)].state.x
            evasion_y = agents["evasion_" + str(j + 1)].state.y
            evasion_theta = agents["evasion_" + str(j + 1)].state.theta
            evasion_info = [evasion_x, evasion_y, evasion_theta]
            evasions_info.append(evasion_info)
        evasions_info = np.array(evasions_info, dtype=np.float64)

        island_info = np.array([self.args.island_x, self.args.island_y], dtype=np.float64)

        return pursuits_info, evasions_info, island_info


    def reset(self):
        if len(self.AgentsContains) == 0:
            self.Init()

        self.init_capture_flags()
        self.init_stop_flags()
        self.fail_flags = False
        self.evasions_paths = [[] for _ in range(self.num_evasion)]
        self.pursuits_paths = [[] for _ in range(self.num_pursuit)]

        self.phase_info = {
            'current_phase': 'approach',
            'phase_switched': False,
            'approach_steps': 0,
            'approach_start_info': None,
            'approach_end_info': None,
            'surround_start_info': None
        }

        island_position = np.array([self.args.island_x, self.args.island_y], dtype=np.float32)
        sample_low = self.args.spawn_low
        sample_high = self.args.spawn_high
        min_pursuer_island_distance = self.args.min_pursuer_island_distance
        min_evader_island_distance = self.args.min_evader_island_distance
        min_pursuer_spacing = self.args.min_pursuer_spacing
        max_spawn_attempts = self.args.max_spawn_attempts

        def _sample_pose():
            x = float(np.random.uniform(sample_low, sample_high))
            y = float(np.random.uniform(sample_low, sample_high))
            theta = float(np.degrees(np.random.uniform(-np.pi, np.pi)))
            return x, y, theta

        for index, agent in enumerate(list(self.AgentsContains)):
            if self.num_pursuit <= index < self.num_pursuit + self.num_evasion:
                for _ in range(max_spawn_attempts):
                    evasion_x, evasion_y, evasion_theta = _sample_pose()
                    evader_to_island = np.linalg.norm(
                        np.array([evasion_x, evasion_y], dtype=np.float32) - island_position
                    )
                    if evader_to_island >= min_evader_island_distance:
                        break
                else:
                    raise RuntimeError("failed to sample a valid evader initial position")

                self.AgentsContains[agent].state.x = evasion_x
                self.AgentsContains[agent].state.y = evasion_y
                self.AgentsContains[agent].state.theta = evasion_theta
                self.AgentsContains[agent].state.v = 0.0
                self.AgentsContains[agent].state.w = 0.0

        sampled_pursuer_positions = []
        for index, agent in enumerate(list(self.AgentsContains)):
            if index < self.num_pursuit:
                for _ in range(max_spawn_attempts):
                    random_x, random_y, random_theta = _sample_pose()
                    position = np.array([random_x, random_y], dtype=np.float32)
                    distance_to_island = np.linalg.norm(position - island_position)
                    if distance_to_island < min_pursuer_island_distance:
                        continue

                    if any(
                        np.linalg.norm(position - other_position) < min_pursuer_spacing
                        for other_position in sampled_pursuer_positions
                    ):
                        continue

                    sampled_pursuer_positions.append(position)
                    break
                else:
                    raise RuntimeError("failed to sample valid pursuer initial positions")

                self.AgentsContains[agent].state.x = random_x
                self.AgentsContains[agent].state.y = random_y
                self.AgentsContains[agent].state.theta = random_theta
                self.AgentsContains[agent].state.v = 0.0
                self.AgentsContains[agent].state.w = 0.0

        for index, agent in enumerate(list(self.AgentsContains)):
            if index >= self.num_pursuit + self.num_evasion:
                random_x, random_y, random_theta = _sample_pose()
                self.AgentsContains[agent].state.x = random_x
                self.AgentsContains[agent].state.y = random_y
                self.AgentsContains[agent].state.theta = random_theta
                self.AgentsContains[agent].state.v = 0.0
                self.AgentsContains[agent].state.w = 0.0

        
        self.pursuits_info = self.agents_info()[0]
        self.evasions_info = self.agents_info()[1]
        self.island_info = island_position
        self.record_paths()

        self.phase_info['approach_start_info'] = self.record_phase_info()

        return self.AgentsContains

    def avoid_evader(self):
        rewards = []
        dones = []
        evader_key = list(self.AgentsContains.keys())[self.num_pursuit]

        for i in range(self.num_pursuit):
            done = False
            pursuer_key = list(self.AgentsContains.keys())[i]

            pos_theta = relative_angle(self.AgentsContains[pursuer_key], self.AgentsContains[evader_key])
            min_dis = relative_dis(self.AgentsContains[pursuer_key], self.AgentsContains[evader_key])

            diff_theta = abs(self.AgentsContains[pursuer_key].state.theta * math.pi / 180 - pos_theta)

            if not self.capture_flags[i]:
                if min_dis < 30:
                    angle_away_from_evader = pos_theta + math.pi
                    self.AgentsContains[pursuer_key].state.theta = angle_away_from_evader
                    self.AgentsContains[pursuer_key].state.v = self.AgentsContains[pursuer_key].max_speed
                    self.AgentsContains[pursuer_key].state.w = 0

                    reward = -10
                    self.stop_flags[i] = True
                    done = True
                else:
                    reward = 0.5 * self.AgentsContains[pursuer_key].state.v * math.cos(diff_theta) - 0.5

            else:
                reward = 0
                done = True

            rewards.append(reward)
            dones.append(done)

        island_x, island_y = self.island_info
        distance_evader_to_island = math.sqrt(
            (self.AgentsContains[evader_key].state.y - island_y) ** 2 +
            (self.AgentsContains[evader_key].state.x - island_x) ** 2
        )

        if distance_evader_to_island < self.args.escape_radius:
            self.fail_flags = True
            rewards = [reward - 1000 for reward in rewards]

        return rewards, dones


    def step(self, action):
        next_obs_n = self.process(action)
        rew_n, done_n = self.reward()
        
        self.record_paths()

        return next_obs_n, rew_n, done_n

    def sample_action(self, island_info, evasions_info, pursuits_info):
        actions = []
        for i in range(self.num_pursuit + self.num_evasion + self.num_obscale):
            if i >= self.num_pursuit and i < self.num_pursuit + self.num_evasion:
                key = "evasion_" + str(i + 1 - self.num_pursuit)
                agent = self.AgentsContains[key]
                action = agent.eva_action(island_info, evasions_info, pursuits_info)
                
                actions.append(action[i - self.num_pursuit])

        return actions


    def record_paths(self):
        for i in range(self.num_evasion):
            x = float(self.evasions_info[i, 0])
            y = float(self.evasions_info[i, 1])
            self.evasions_paths[i].append([x, y])  

        for i in range(self.num_pursuit):
            x = float(self.pursuits_info[i, 0])
            y = float(self.pursuits_info[i, 1])
            self.pursuits_paths[i].append([x, y])  

    def plot_paths(self):
        plt.figure(figsize=(6, 6))  
        
        evasion_colors = ['#d62728']
        pursuit_colors = ['#1f77b4',  
                          '#17becf',  
                          '#2ca02c',  
                          '#bcbd22']  

        
        for i, path in enumerate(self.evasions_paths):
            if len(path) > 1:  
                path = np.array(path)
                plt.plot(path[:, 0], path[:, 1], color=evasion_colors[i % len(evasion_colors)], label=f'threat {i + 1}')
                
                plt.scatter(path[0, 0], path[0, 1], color=evasion_colors[i % len(evasion_colors)], marker='o')  
                plt.scatter(path[-1, 0], path[-1, 1], color=evasion_colors[i % len(evasion_colors)], marker='x')  

        for i, path in enumerate(self.pursuits_paths):
            if len(path) > 1:
                path = np.array(path)
                plt.plot(path[:, 0], path[:, 1], color=pursuit_colors[i % len(pursuit_colors)], label=f'USV {i + 1}')
                
                plt.scatter(path[0, 0], path[0, 1], color=pursuit_colors[i % len(pursuit_colors)], marker='o')  
                plt.scatter(path[-1, 0], path[-1, 1], color=pursuit_colors[i % len(pursuit_colors)], marker='x')  

        
        island_x, island_y = 380, 380
        plt.scatter(island_x, island_y, color='black', marker='s', label='Island')  

        
        plt.scatter([], [], color='black', marker='o', label='Start')
        plt.scatter([], [], color='black', marker='x', label='End')

        
        ax = plt.gca()  
        ax.tick_params(direction='in')  

        
        bwith = 1  
        ax.spines['bottom'].set_linewidth(bwith)  
        ax.spines['left'].set_linewidth(bwith)  
        ax.spines['top'].set_linewidth(bwith)  
        ax.spines['right'].set_linewidth(bwith)  

        
        plt.xlabel(r"$\it{x}$ /m", fontsize=14)  
        plt.ylabel(r"$\it{y}$ /m", fontsize=14)  

        plt.xlim(0, self.args.map_width)  
        plt.ylim(0, self.args.map_height)  

        
        ax.set_aspect('equal', adjustable='box')

        
        ax.set_xticks(range(0, self.args.map_width + 1, 50))
        ax.set_yticks(range(0, self.args.map_height + 1, 50))

        
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()

        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        
        x_tick_labels = [f'{int(x)}' if x != 0 else '0' for x in x_ticks]
        y_tick_labels = [f'{int(y)}' if y != 0 else '' for y in y_ticks]

        
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticklabels(y_tick_labels)

        
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()  

    def plot_global_paths(self, save_path):
        plt.figure(figsize=(8, 8))

        
        pursuit_colors = ['#1f77b4', '#17becf', '#2ca02c', '#bcbd22']
        for i, path in enumerate(self.pursuits_paths):
            if len(path) > 1:
                path = np.array(path)
                plt.plot(path[:, 0], path[:, 1],
                         color=pursuit_colors[i % 4],
                         linewidth=1.5,
                         alpha=0.7,
                         label=f'Pursuer {i + 1}')

        if len(self.evasions_paths) > 0 and len(self.evasions_paths[0]) > 0:
            evasion_path = np.array(self.evasions_paths[0]).reshape(-1, 2)  
            plt.plot(evasion_path[:, 0], evasion_path[:, 1],
                     'r--', linewidth=2, label='Threat')

        
        plt.scatter(380, 380, color='black', marker='s', s=100, label='Island')
        plt.xlabel(r"$\it{x}$ /m", fontsize=12)
        plt.ylabel(r"$\it{y}$ /m", fontsize=12)
        plt.xlim(0, self.args.map_width)
        plt.ylim(0, self.args.map_height)
        plt.gca().set_aspect('equal')
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.25, 1))

        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def render(self):
        if self.viewer is True:
            self.viewer = Viewer(*self.viewer_xy, self.island_info, self.evasions_info, self.pursuits_info)
        self.viewer.evasions_info = self.evasions_info
        self.viewer.pursuits_info = self.pursuits_info
        self.viewer.render()


    def process(self, action):
        agents = self.AgentsContains
        for index, key in enumerate(agents.keys()):
            if not self.stop_flags[index]:  
                "Keep theta between -180 and +180 "
                agents[key].state.theta = (agents[key].state.theta * math.pi / 180 - 2 * math.pi * np.floor(
                    (agents[key].state.theta * math.pi / 180 + math.pi) / (2 * math.pi))) * 180 / math.pi

                if index < self.num_pursuit:
                    action[index][0] = float(np.clip(action[index][0], 0.0, 1.0))
                    action[index][1] = float(np.clip(action[index][1], -1.0, 1.0))
                    agents[key].state.v = action[index][0]
                    agents[key].state.w = action[index][1]
                else:
                    agents[key].state.v = action[index][0]
                    agents[key].state.w = action[index][1]

                if index < self.num_pursuit:
                    
                    agents[key].state.x += math.cos(agents[key].state.theta * math.pi / 180) * agents[key].state.v *\
                                           agents[
                                               key].max_v * self.args.timestep
                    agents[key].state.y += math.sin(agents[key].state.theta * math.pi / 180) * agents[key].state.v *\
                                           agents[
                                               key].max_v * self.args.timestep
                else:
                    
                    agents[key].state.x += math.cos(agents[key].state.theta * math.pi / 180) * agents[key].state.v * (
                            agents[
                                key].max_v - 0) * self.args.timestep
                    agents[key].state.y += math.sin(agents[key].state.theta * math.pi / 180) * agents[key].state.v * (
                            agents[
                                key].max_v - 0) * self.args.timestep

                agents[key].state.theta += agents[key].state.w * agents[key].max_w * self.args.timestep * 180 / math.pi

                "Keep theta between -180 and +180 "
                agents[key].state.theta = (agents[key].state.theta * math.pi / 180 - 2 * math.pi * np.floor(
                    (agents[key].state.theta * math.pi / 180 + math.pi) / (2 * math.pi))) * 180 / math.pi
                if agents[key].state.x < 0 or agents[key].state.x > self.args.map_width or agents[key].state.y < 0 or\
                        agents[key].state.y > self.args.map_height:
                    agents[key].state.x = min(max(agents[key].state.x, 0.0), self.args.map_width)
                    agents[key].state.y = min(max(agents[key].state.y, 0.0), self.args.map_height)

                agents[key].state.theta = (agents[key].state.theta * math.pi / 180 - 2 * math.pi * np.floor(
                    (agents[key].state.theta * math.pi / 180 + math.pi) / (2 * math.pi))) * 180 / math.pi

        for i in range(self.num_pursuit):
            agent_name = f"pursuit_{i + 1}"
            self.pursuits_info[i, :] = [agents[agent_name].state.x, agents[agent_name].state.y,
                                        agents[agent_name].state.theta]

        for i in range(self.num_evasion):
            agent_name = f"evasion_{i + 1}"
            self.evasions_info[i, :] = [agents[agent_name].state.x, agents[agent_name].state.y,
                                        agents[agent_name].state.theta]

        
        self.update_phase()

        return self.AgentsContains

    def update_phase(self):
        if self.phase_info['phase_switched']:
            return  

        self.phase_info.setdefault('approach_steps', 0)
        self.phase_info['approach_steps'] += 1  

        
        evader_key = list(self.AgentsContains.keys())[self.num_pursuit]
        evader = self.AgentsContains[evader_key]
        distances = []
        for i in range(self.num_pursuit):
            pursuer_key = list(self.AgentsContains.keys())[i]
            pursuer = self.AgentsContains[pursuer_key]
            distance = math.sqrt((pursuer.state.x - evader.state.x) ** 2 + (pursuer.state.y - evader.state.y) ** 2)
            distances.append(distance)

        
        if all(dist <= self.d_cap for dist in distances):
            self.phase_info['current_phase'] = 'surround'
            self.phase_info['phase_switched'] = True
            self.phase_info['approach_end_info'] = self.record_phase_info()
            self.phase_info['surround_start_info'] = self.phase_info['approach_end_info']

    def record_phase_info(self):
        evader_key = list(self.AgentsContains.keys())[self.num_pursuit]
        evader = self.AgentsContains[evader_key]

        info = []
        for i in range(self.num_pursuit):
            pursuer_key = list(self.AgentsContains.keys())[i]
            pursuer = self.AgentsContains[pursuer_key]

            dx = evader.state.x - pursuer.state.x
            dy = evader.state.y - pursuer.state.y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            angle = math.atan2(dy, dx)
            reward = pursuer.total_reward if hasattr(pursuer, 'total_reward') else 0.0  

            info.append({
                'distance': distance,
                'angle': angle,
                'reward': reward
            })

        return info

    def compute_encirclement_metrics(self):
        if self.num_pursuit <= 0:
            return {
                "d_cap": float(self.args.d_cap),
                "d_suc_low": float(self.args.d_suc_low),
                "d_suc_high": float(self.args.d_suc_high),
                "d_far": float(self.args.d_far),
                "wall_margin": float(self.args.wall_margin),
                "d_bar": 0.0,
                "dists": np.array([], dtype=np.float32),
                "max_gap": 2.0 * math.pi,
                "sigma_gap": math.pi,
                "distance_band_ok": False,
                "angle_closure_ok": False,
                "capture_success": False,
                "escaped": False,
                "phase_switched": bool(self.phase_info.get('phase_switched', False)),
                "pairwise": np.zeros((0, 0), dtype=np.float32),
                "min_pp_distance": float("inf"),
                "evader_pos": np.zeros(2, dtype=np.float32),
                "evader_wall_clearance": float("inf"),
                "evader_near_wall_ratio": 0.0,
                "pursuer_pos": np.zeros((0, 2), dtype=np.float32),
                "pursuer_heading": np.zeros(0, dtype=np.float32),
                "pursuer_v": np.zeros(0, dtype=np.float32),
            }

        d_cap = float(self.args.d_cap)
        d_suc_low = float(self.args.d_suc_low)
        d_suc_high = float(self.args.d_suc_high)
        d_far = float(self.args.d_far)
        wall_margin = float(self.args.wall_margin)

        island_pos = np.array(self.island_info, dtype=np.float32)
        evader_key = list(self.AgentsContains.keys())[self.num_pursuit]
        evader_state = self.AgentsContains[evader_key].state
        evader_pos = np.array([evader_state.x, evader_state.y], dtype=np.float32)
        evader_wall_clearance = float(min(
            evader_pos[0],
            self.args.map_width - evader_pos[0],
            evader_pos[1],
            self.args.map_height - evader_pos[1]
        ))
        evader_near_wall_ratio = float(np.clip(
            (wall_margin - evader_wall_clearance) / max(wall_margin, 1e-6),
            0.0,
            1.0
        ))

        pursuer_states = [
            self.AgentsContains[list(self.AgentsContains.keys())[i]].state
            for i in range(self.num_pursuit)
        ]
        pursuer_pos = np.array([[p.x, p.y] for p in pursuer_states], dtype=np.float32)
        pursuer_heading = np.array([math.radians(p.theta) for p in pursuer_states], dtype=np.float32)
        pursuer_v = np.array([p.v for p in pursuer_states], dtype=np.float32)

        dists = np.linalg.norm(pursuer_pos - evader_pos[None, :], axis=1).astype(np.float32)
        d_bar = float(np.mean(dists))

        rel_angles = np.arctan2(
            pursuer_pos[:, 1] - evader_pos[1],
            pursuer_pos[:, 0] - evader_pos[0]
        ).astype(np.float32)
        sort_idx = np.argsort(rel_angles)
        angles_sorted = rel_angles[sort_idx]
        gaps = np.diff(np.concatenate([angles_sorted, [angles_sorted[0] + 2 * np.pi]]))
        max_gap = float(np.max(gaps)) if gaps.size > 0 else 2.0 * math.pi
        sigma_gap = float(np.std(gaps)) if gaps.size > 0 else math.pi
        target_gap = float((2.0 * math.pi) / max(self.num_pursuit, 1))
        right_gaps = np.zeros_like(rel_angles)
        left_gaps = np.zeros_like(rel_angles)
        if gaps.size > 0:
            right_gaps[sort_idx] = gaps
            left_gaps[sort_idx] = np.roll(gaps, 1)
        min_local_gaps = np.minimum(left_gaps, right_gaps).astype(np.float32)

        pairwise = np.linalg.norm(pursuer_pos[:, None, :] - pursuer_pos[None, :, :], axis=-1).astype(np.float32)
        np.fill_diagonal(pairwise, np.inf)
        min_pp_distance = float(np.min(pairwise)) if self.num_pursuit > 1 else float("inf")

        distance_band_ok = bool(np.all((dists >= d_suc_low) & (dists <= d_suc_high)))
        angle_closure_ok = bool(max_gap < self.args.max_gap_success_ratio_pi * math.pi)
        capture_success = bool(distance_band_ok and angle_closure_ok)
        escaped = bool(np.linalg.norm(evader_pos - island_pos) < self.args.escape_radius)

        return {
            "d_cap": d_cap,
            "d_suc_low": d_suc_low,
            "d_suc_high": d_suc_high,
            "d_far": d_far,
            "wall_margin": wall_margin,
            "d_bar": d_bar,
            "dists": dists,
            "max_gap": max_gap,
            "sigma_gap": sigma_gap,
            "target_gap": target_gap,
            "left_gaps": left_gaps,
            "right_gaps": right_gaps,
            "min_local_gaps": min_local_gaps,
            "distance_band_ok": distance_band_ok,
            "angle_closure_ok": angle_closure_ok,
            "capture_success": capture_success,
            "escaped": escaped,
            "phase_switched": bool(self.phase_info.get('phase_switched', False)),
            "pairwise": pairwise,
            "min_pp_distance": min_pp_distance,
            "evader_pos": evader_pos,
            "evader_wall_clearance": evader_wall_clearance,
            "evader_near_wall_ratio": evader_near_wall_ratio,
            "pursuer_pos": pursuer_pos,
            "pursuer_heading": pursuer_heading,
            "pursuer_v": pursuer_v,
        }

    def _reward_params(self):
        return {
            "r_suc": self.args.r_suc,
            "r_fail": self.args.r_fail,
            "k1": self.args.reward_k1,
            "k2": self.args.reward_k2,
            "k3": self.args.reward_k3,
            "d_safe_pp": self.args.reward_d_safe_pp,
            "lambda_pp": self.args.reward_lambda_pp,
            "lambda_pe": self.args.reward_lambda_pe,
            "w_approach": np.array(self.args.reward_w_approach, dtype=np.float32),
            "w_surround": np.array(self.args.reward_w_surround, dtype=np.float32),
        }

    def reward(self):
        if self.num_pursuit <= 0:
            self.last_reward_info = {
                "dense_rewards": [],
                "terminal_bonus": 0.0,
                "mode": "empty",
            }
            return [], []

        params = self._reward_params()
        r_suc = params["r_suc"]
        r_fail = params["r_fail"]
        k1 = params["k1"]
        k2 = params["k2"]
        k3 = params["k3"]
        d_safe_pp = params["d_safe_pp"]
        lambda_pp = params["lambda_pp"]
        lambda_pe = params["lambda_pe"]
        w_approach = params["w_approach"]
        w_surround = params["w_surround"]

        metrics = self.compute_encirclement_metrics()
        d_cap = metrics["d_cap"]
        d_suc_low = metrics["d_suc_low"]
        d_suc_high = metrics["d_suc_high"]
        d_far = metrics["d_far"]
        dists = metrics["dists"]
        sigma_gap = metrics["sigma_gap"]
        evader_pos = metrics["evader_pos"]
        pursuer_pos = metrics["pursuer_pos"]
        pursuer_heading = metrics["pursuer_heading"]
        pursuer_v = metrics["pursuer_v"]
        pairwise = metrics["pairwise"]

        r3_shared = 1.0 - sigma_gap / math.pi
        r3_shared = float(np.clip(r3_shared, -1.0, 1.0))

        capture_success = metrics["capture_success"]
        escape = metrics["escaped"]

        if capture_success:
            self.capture_flags = [True] * self.num_pursuit
            for i in range(self.num_pursuit):
                self.stop_flags[i] = True
            if len(self.stop_flags) > self.num_pursuit:
                self.stop_flags[self.num_pursuit] = True

            rewards = [r_suc] * self.num_pursuit
            dones = [True] * self.num_pursuit
            self.last_reward_info = {
                "dense_rewards": [0.0] * self.num_pursuit,
                "terminal_bonus": r_suc,
                "mode": "success",
            }
            return rewards, dones

        if escape:
            self.fail_flags = True
            rewards = [r_fail] * self.num_pursuit
            dones = [False] * self.num_pursuit
            self.last_reward_info = {
                "dense_rewards": [0.0] * self.num_pursuit,
                "terminal_bonus": r_fail,
                "mode": "escape",
            }
            return rewards, dones

        if dists.size > 0:
            d_ref = float(np.percentile(dists, self.args.reward_distance_percentile))
        else:
            d_ref = 0.0

        alpha = float(np.clip(
            (d_ref - d_cap) / max(d_far - d_cap, 1e-6),
            0.0,
            1.0
        ))

        w = alpha * w_approach + (1.0 - alpha) * w_surround
        w_sum = float(np.sum(w))
        if w_sum <= 1e-8:
            w = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        else:
            w = w / w_sum

        w1, w2, w3, w4 = [float(v) for v in w]

        rewards = []
        for i in range(self.num_pursuit):
            dx = float(evader_pos[0] - pursuer_pos[i, 0])
            dy = float(evader_pos[1] - pursuer_pos[i, 1])
            phi_i = math.atan2(dy, dx)
            varphi_i = float(pursuer_heading[i])
            d_i = float(dists[i])

            r1 = k1 * float(pursuer_v[i]) * math.cos(
                (varphi_i - phi_i + math.pi) % (2 * math.pi) - math.pi
            ) + k2

            radial_error = max(0.0, d_suc_low - d_i) + max(0.0, d_i - d_suc_high)
            r2 = 2.0 * math.exp(-k3 * radial_error) - 1.0

            if self.num_pursuit > 1:
                d_min_pp_i = float(np.min(pairwise[i]))
            else:
                d_min_pp_i = np.inf

            pp_shortfall = max(0.0, (d_safe_pp - d_min_pp_i) / d_safe_pp)
            pe_shortfall = max(0.0, (d_suc_low - d_i) / d_suc_low)
            r4_safe = -lambda_pp * pp_shortfall ** 2 - lambda_pe * pe_shortfall ** 2

            r_step = w1 * r1 + w2 * r2 + w3 * r3_shared + w4 * r4_safe
            rewards.append(float(r_step))

        dones = [False] * self.num_pursuit
        self.last_reward_info = {
            "dense_rewards": [float(r) for r in rewards],
            "terminal_bonus": 0.0,
            "mode": "dense",
        }
        return rewards, dones


    def init_capture_flags(self):
        
        self.capture_flags = [False] * self.num_pursuit

    def init_stop_flags(self):
        
        self.stop_flags = [False] * (self.num_pursuit + self.num_evasion)



class Viewer(pyglet.window.Window):

    def __init__(self, width, height, island_info, evasions_info, pursuits_info):
        super(Viewer, self).__init__(width, height, caption='cooperate mission')

        pyglet.gl.glClearColor(1, 1, 1, 1)

        self.island_info = island_info
        self.evasions_info = evasions_info
        self.pursuits_info = pursuits_info

        self.num_evasions = self.evasions_info.shape[0]  
        self.num_pursuits = self.pursuits_info.shape[0]

        background = pyglet.graphics.OrderedGroup(0)
        foreground = pyglet.graphics.OrderedGroup(1)

        
        self.line_batch = pyglet.graphics.Batch()
        x0 = 0
        y0 = 0
        Line_H = []
        Line_V = []
        while x0 < 1200:
            x0 += 80
            line_h = shapes.Line(x0, 0, x0, 800,
                                 width=2, color=(0, 0, 0),
                                 batch=self.line_batch, group=foreground)
            line_h.opacity = 16
            Line_H.append(line_h)
            while x0 == 1200 and y0 < 800:
                y0 += 80
                line_v = shapes.Line(0, y0, 1200, y0,
                                     width=2, color=(0, 0, 0),
                                     batch=self.line_batch, group=foreground)
                line_v.opacity = 16
                Line_V.append(line_v)

        
        self.model_batch = pyglet.graphics.Batch()

        island_x, island_y = self.island_info
        self.island = pyglet.sprite.Sprite(img=resources.kernel_image,
                                           x=island_x, y=island_y,
                                           batch=self.model_batch, group=foreground)

        self.evasions = []
        evasions_x = self.evasions_info[:, 0]
        evasions_y = self.evasions_info[:, 1]
        evasions_rotation = self.evasions_info[:, 2]
        for i in range(self.num_evasions):
            new_evasion = pyglet.sprite.Sprite(img=resources.threat_image,
                                               x=evasions_x[i], y=evasions_y[i],
                                               batch=self.model_batch, group=foreground)
            new_evasion.rotation = -evasions_rotation[i]
            self.evasions.append(new_evasion)

        self.pursuits = []
        pursuits_x = self.pursuits_info[:, 0]
        pursuits_y = self.pursuits_info[:, 1]
        pursuits_rotation = self.pursuits_info[:, 2]
        for i in range(self.num_pursuits):
            new_pursuit = pyglet.sprite.Sprite(img=resources.guard_image,
                                               x=pursuits_x[i], y=pursuits_y[i],
                                               batch=self.model_batch, group=foreground)
            new_pursuit.rotation = -pursuits_rotation[i]
            self.pursuits.append(new_pursuit)
    def render(self):
        pyglet.clock.tick()
        self._update()

        self.switch_to()
        self.dispatch_events()  
        self.dispatch_event('on_draw')
        self.flip()  

    def on_draw(self):
        self.clear()

        self.model_batch.draw()
        self.line_batch.draw()

    
    def _update(self):

        for i in range(self.num_evasions):
            self.evasions[i].x = self.evasions_info[i, 0]
            
            self.evasions[i].y = self.evasions_info[i, 1]
            self.evasions[i].rotation = -self.evasions_info[i, 2]

        for i in range(self.num_pursuits):
            self.pursuits[i].x = self.pursuits_info[i, 0]
            self.pursuits[i].y = self.pursuits_info[i, 1]
            self.pursuits[i].rotation = -self.pursuits_info[i, 2]

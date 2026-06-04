
import numpy as np
import math
from env import Env
import matd3
import matd3.common.tf_util as U
import tensorflow as tf  
import argparse
import matplotlib.pyplot as plt
import os
import re
import time
import pandas as pd
import datetime
import random
from weight_calculator import WeightCalculator
from policy_aggregator import aggregate_actor_parameters_tf
from psac_matd3.utils.cbf import apply_cbf_to_actions
from psac_matd3.utils.geometry import build_input_state
from psac_matd3.utils.run_metadata import save_run_metadata
from psac_matd3.utils.config import parse_args_with_config

DEFAULT_TRAIN_LOAD_DIR = ""
DEFAULT_EVAL_LOAD_DIR = os.path.join("results", "run_main", "latest")
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "train_default.json")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def float_list(value):
    if isinstance(value, list):
        values = value
    else:
        values = str(value).split(",")
    try:
        return [float(item) for item in values]
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("expected a comma-separated list of numbers")


def build_parser():
    parser = argparse.ArgumentParser("parse args for multiagent environments")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="grouped JSON configuration file")
    parser.add_argument("--episode_len", type=int, default=2500, help="step per episode")
    parser.add_argument("--episode_num", type=int, default=1200, help="number of episodes")
    parser.add_argument("--load_dir", type=str,
                        default=DEFAULT_TRAIN_LOAD_DIR,
                        help="load model from this directory")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="results root dir")
    parser.add_argument("--load_model", type=str2bool, default=False, help="load model")
    parser.add_argument("--eval_only", type=str2bool, default=False, help="run evaluation only")
    parser.add_argument("--eval_episodes", type=int, default=5, help="number of evaluation episodes")
    parser.add_argument("--eval_checkpoint", type=str, default="latest",
                        help="evaluation checkpoint label; latest uses the default eval model")
    parser.add_argument("--save_every", type=int, default=200, help="save a historical checkpoint every N episodes")
    parser.add_argument("--latest_save_every", type=int, default=50, help="overwrite latest checkpoint every N episodes")

    parser.add_argument("--display", type=str2bool, default=False, help="display pyglet")
    parser.add_argument("--save_plots", type=str2bool, default=False,
                        help="save reward-curve PNG figures")
    parser.add_argument('--random-seed', type=int, help='random seed for repeatability', default=11)

    parser.add_argument("--agent_maxspeed", type=float, default=3.0, help="max speed of agent")
    parser.add_argument("--evader_maxspeed", type=float, default=2.0, help="max speed of evader")
    parser.add_argument("--agent_max_w", type=float, default=1.0, help="max w of agent")

    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount")
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch per update")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--input_state_dim", type=int, default=11, help="the input dim of agent actor net")
    parser.add_argument("--action_dim", type=int, default=2, help="action dim of agent")
    parser.add_argument("--use_critic_noise", type=str2bool, default=False, help="use noise in critic update next action")
    parser.add_argument("--use_critic_noise_self", type=str2bool, default=True,
                        help="use noise in critic update next action")
    parser.add_argument("--critic_action_noise_stddev", type=float, default=0.1)
    parser.add_argument("--action_noise_clip", type=float, default=0.1)
    parser.add_argument("--critic_zero_if_done", type=str2bool, default=True,
                        help="set q value to zero in critic update after done")
    parser.add_argument("--use_policy_sharing", type=str2bool, default=False,
                        help="enable policy aggregation and ARM")
    parser.add_argument("--update_rate", type=int, default=20, help="after this many steps the critic is trained")
    parser.add_argument("--policy_update_rate", type=int, default=2,
                        help="after this many critic updates the target networks and policy are trained")
    parser.add_argument("--use_hper", type=str2bool, default=False,
                        help="enable hybrid priority experience replay")
    parser.add_argument("--hper_beta_start", type=float, default=0.8,
                        help="initial TD-error weight for hybrid priority replay")
    parser.add_argument("--hper_beta_end", type=float, default=0.2,
                        help="final TD-error weight for hybrid priority replay")
    parser.add_argument("--hper_alpha1", type=float, default=0.6,
                        help="TD-error priority exponent for hybrid priority replay")
    parser.add_argument("--hper_alpha2", type=float, default=0.4,
                        help="formation priority exponent for hybrid priority replay")
    parser.add_argument("--hper_rd", type=float, default=40.0,
                        help="desired encirclement radius for hybrid priority replay")

    parser.add_argument("--cbf_enable", type=str2bool, default=False,
                        help="enable CBF safety projection for pursuer linear speeds")
    parser.add_argument("--cbf_apply_in_train", type=str2bool, default=True,
                        help="apply CBF projection during training")
    parser.add_argument("--cbf_apply_in_eval", type=str2bool, default=True,
                        help="apply CBF projection during evaluation")
    parser.add_argument("--cbf_use_prev_safe_neighbor", type=str2bool, default=True,
                        help="use previous executed safe speed for neighboring pursuers")
    parser.add_argument("--dmin_pp", type=float, default=31.0,
                        help="minimum CBF safety distance between pursuers")
    parser.add_argument("--dmin_pe", type=float, default=31.0,
                        help="minimum CBF safety distance between pursuer and evader")
    parser.add_argument("--gamma_pp", type=float, default=2.0,
                        help="CBF gain for pursuer-pursuer constraints")
    parser.add_argument("--gamma_pe", type=float, default=2.0,
                        help="CBF gain for pursuer-evader constraints")
    parser.add_argument("--gamma_border", type=float, default=0.8,
                        help="CBF gain for map-boundary constraints")
    parser.add_argument("--r_act", type=float, default=60.0,
                        help="activation radius for pairwise CBF constraints")
    parser.add_argument("--v_min", type=float, default=0.0,
                        help="minimum normalized linear speed after CBF projection")
    parser.add_argument("--v_max", type=float, default=1.0,
                        help="maximum normalized linear speed after CBF projection")
    parser.add_argument("--eps_trig", type=float, default=1e-6,
                        help="small threshold for near-zero trigonometric coefficients")

    parser.add_argument("--timestep", type=float, default=0.1, help="timestep")
    parser.add_argument("--map_width", type=int, default=400, help="width of env")
    parser.add_argument("--map_height", type=int, default=400, help="height of env")
    parser.add_argument("--num_pursuit", type=int, default=4, help="number of pursuit")
    parser.add_argument("--num_evasion", type=int, default=1, help="number of evasion")
    parser.add_argument("--num_obstacle", type=int, default=0, help="number of obstacle")
    parser.add_argument("--trajectory_save_mode", type=str, default="none",
                        choices=["success", "failure", "all", "none"],
                        help="when to save episode trajectory plots")

    parser.add_argument("--replay_buffer_capacity", type=int, default=1000000)
    parser.add_argument("--min_replay_buffer_len", type=int, default=1000)
    parser.add_argument("--grad_norm_clipping", type=float, default=0.5)

    parser.add_argument("--island_x", type=float, default=380.0)
    parser.add_argument("--island_y", type=float, default=380.0)
    parser.add_argument("--spawn_low", type=float, default=50.0)
    parser.add_argument("--spawn_high", type=float, default=350.0)
    parser.add_argument("--min_pursuer_island_distance", type=float, default=50.0)
    parser.add_argument("--min_evader_island_distance", type=float, default=250.0)
    parser.add_argument("--min_pursuer_spacing", type=float, default=30.0)
    parser.add_argument("--max_spawn_attempts", type=int, default=10000)

    parser.add_argument("--d_cap", type=float, default=120.0)
    parser.add_argument("--d_suc_low", type=float, default=32.0)
    parser.add_argument("--d_suc_high", type=float, default=48.0)
    parser.add_argument("--d_far", type=float, default=180.0)
    parser.add_argument("--wall_margin", type=float, default=60.0)
    parser.add_argument("--escape_radius", type=float, default=10.0)
    parser.add_argument("--max_gap_success_ratio_pi", type=float, default=1.0)
    parser.add_argument("--reward_distance_percentile", type=float, default=90.0)
    parser.add_argument("--r_suc", type=float, default=3000.0)
    parser.add_argument("--r_fail", type=float, default=-1000.0)
    parser.add_argument("--reward_k1", type=float, default=1.2)
    parser.add_argument("--reward_k2", type=float, default=-0.3)
    parser.add_argument("--reward_k3", type=float, default=0.08)
    parser.add_argument("--reward_d_safe_pp", type=float, default=31.0)
    parser.add_argument("--reward_lambda_pp", type=float, default=2.0)
    parser.add_argument("--reward_lambda_pe", type=float, default=1.0)
    parser.add_argument(
        "--reward_w_approach",
        type=float_list,
        default=[0.25, 0.25, 0.25, 0.25],
        help=(
            "four concrete approach reward weights [w1,w2,w3,w4]; suggested tuning ranges "
            "w1=0.20-0.45, w2=0.30-0.55, w3=0.05-0.25, w4=0.05-0.20. "
            "Ranges are guidance only; weights are normalized during reward calculation"
        )
    )
    parser.add_argument(
        "--reward_w_surround",
        type=float_list,
        default=[0.25, 0.25, 0.25, 0.25],
        help=(
            "four concrete surround reward weights [w1,w2,w3,w4]; suggested tuning ranges "
            "w1=0.00-0.15, w2=0.10-0.30, w3=0.35-0.65, w4=0.15-0.35. "
            "Ranges are guidance only; weights are normalized during reward calculation"
        )
    )

    parser.add_argument("--policy_sharing_every", type=int, default=100)
    parser.add_argument("--policy_reward_window", type=int, default=20)
    parser.add_argument("--policy_sigma_d", type=float, default=20.0)
    parser.add_argument("--policy_capture_radius", type=float, default=40.0)
    parser.add_argument("--arm_eta", type=float, default=0.01)
    parser.add_argument("--arm_lambda", type=float, default=0.5)
    parser.add_argument("--arm_train_episodes", type=int, default=20)

    parser.add_argument("--apf_attract_gain", type=float, default=0.002)
    parser.add_argument("--apf_attract_radius", type=float, default=80.0)
    parser.add_argument("--apf_repel_gain", type=float, default=1500000.0)
    parser.add_argument("--apf_repel_radius", type=float, default=50.0)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parse_args_with_config(parser, argv=argv, default_config=DEFAULT_CONFIG)
    positive_names = [
        "episode_len", "episode_num", "batch_size", "num_units", "update_rate",
        "policy_update_rate", "replay_buffer_capacity", "min_replay_buffer_len",
        "map_width", "map_height", "num_pursuit", "num_evasion", "max_spawn_attempts",
        "policy_sharing_every", "policy_reward_window", "arm_train_episodes", "timestep",
        "eval_episodes", "save_every", "latest_save_every"
    ]
    invalid_positive = [name for name in positive_names if getattr(args, name) <= 0]
    if invalid_positive:
        parser.error("these values must be positive: {}".format(", ".join(invalid_positive)))
    if args.spawn_low >= args.spawn_high:
        parser.error("spawn_low must be smaller than spawn_high")
    if args.d_suc_low >= args.d_suc_high:
        parser.error("d_suc_low must be smaller than d_suc_high")
    if args.v_min > args.v_max:
        parser.error("v_min must not exceed v_max")
    if not 0.0 <= args.reward_distance_percentile <= 100.0:
        parser.error("reward_distance_percentile must be between 0 and 100")
    if args.max_gap_success_ratio_pi <= 0:
        parser.error("max_gap_success_ratio_pi must be positive")
    if len(args.reward_w_approach) != 4 or len(args.reward_w_surround) != 4:
        parser.error("reward_w_approach and reward_w_surround must each contain four values")
    if not all(math.isfinite(value) for value in args.reward_w_approach + args.reward_w_surround):
        parser.error("reward weights must be finite numbers")
    if sum(args.reward_w_approach) <= 0 or sum(args.reward_w_surround) <= 0:
        parser.error("each reward weight vector must have a positive sum")
    return args


def _prepare_output_dirs(save_dir):
    output_dirs = {
        "latest": os.path.join(save_dir, "latest"),
        "checkpoints": os.path.join(save_dir, "checkpoints"),
        "logs": os.path.join(save_dir, "logs"),
    }
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)
    return output_dirs


def _latest_ckpt_prefix(latest_dir):
    return os.path.join(latest_dir, "latest")


def _resolve_run_dir(results_root, load_model):
    results_root = os.path.abspath(results_root)
    if load_model:
        os.makedirs(results_root, exist_ok=True)
        return results_root

    os.makedirs(results_root, exist_ok=True)
    run_pattern = re.compile(r"^run_(\d+)$")
    max_run_id = 0
    for name in os.listdir(results_root):
        full_path = os.path.join(results_root, name)
        if not os.path.isdir(full_path):
            continue
        match = run_pattern.match(name)
        if not match:
            continue
        max_run_id = max(max_run_id, int(match.group(1)))
    run_dir = os.path.join(results_root, f"run_{max_run_id + 1}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _reward_curve_df(reward_curve, num_pursuit):
    lengths = [len(reward_curve.get("episode", [])), len(reward_curve.get("average_reward", []))]
    lengths.extend(len(reward_curve.get(f"pursuit{i + 1}", [])) for i in range(num_pursuit))
    if "dense_average_reward" in reward_curve:
        lengths.append(len(reward_curve.get("dense_average_reward", [])))
        lengths.extend(len(reward_curve.get(f"dense_pursuit{i + 1}", [])) for i in range(num_pursuit))
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    if min_len != max_len:
        print(f"[Warning] reward_curve length mismatch {lengths}; truncate to {min_len}")
    data = {
        "episode": reward_curve.get("episode", [])[:min_len],
        "average_reward": reward_curve.get("average_reward", [])[:min_len],
    }
    for i in range(num_pursuit):
        data[f"pursuit{i + 1}"] = reward_curve.get(f"pursuit{i + 1}", [])[:min_len]
    if "dense_average_reward" in reward_curve:
        data["dense_average_reward"] = reward_curve.get("dense_average_reward", [])[:min_len]
        for i in range(num_pursuit):
            data[f"dense_pursuit{i + 1}"] = reward_curve.get(f"dense_pursuit{i + 1}", [])[:min_len]
    return pd.DataFrame(data)


def _has_valid_trajectory(env):
    paths = list(getattr(env, "pursuits_paths", [])) + list(getattr(env, "evasions_paths", []))
    return any(len(path) > 1 for path in paths)


def save_episode_trajectory(env, args, episode_num, is_success, output_dirs):
    mode = args.trajectory_save_mode
    should_save = (
        mode == "all"
        or (mode == "success" and is_success)
        or (mode == "failure" and not is_success)
    )
    if not should_save:
        return
    if not _has_valid_trajectory(env):
        print(f"[Warning] skip trajectory for episode {episode_num}: empty path")
        return

    status = "success" if is_success else "failure"
    traj_dir = output_dirs.get("trajectories", os.path.join(args.save_dir, "trajectories"))
    os.makedirs(traj_dir, exist_ok=True)
    save_path = os.path.join(traj_dir, f"traj_{status}_ep{episode_num:04d}.pdf")
    try:
        env.plot_paths()
        plt.savefig(save_path, dpi=600, format="pdf")
    except Exception as exc:
        print(f"[Warning] failed to save trajectory for episode {episode_num}: {exc}")
    finally:
        plt.close()


def _ckpt_dir(base_dir, ep):
    d = os.path.join(base_dir, "checkpoints", f"ep{ep:04d}")
    os.makedirs(d, exist_ok=True)
    return d


def _ckpt_prefix(base_dir, ep):
    ckpt_dir = _ckpt_dir(base_dir, ep)
    return os.path.join(ckpt_dir, f"ep{ep:04d}")


def _resolve_checkpoint_prefix(path):
    path = os.path.abspath(path)
    if os.path.isfile(path + ".index"):
        return path
    if os.path.isdir(path):
        ckpt_meta = os.path.join(path, "checkpoint")
        if not os.path.isfile(ckpt_meta):
            latest_subdir = os.path.join(path, "latest")
            if os.path.isdir(latest_subdir):
                latest_subdir_resolved = _resolve_checkpoint_prefix(latest_subdir)
                if latest_subdir_resolved != latest_subdir:
                    return latest_subdir_resolved
        if os.path.isfile(ckpt_meta):
            with open(ckpt_meta, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.startswith("model_checkpoint_path:"):
                        continue
                    value = line.split(":", 1)[1].strip().strip('"')
                    if not os.path.isabs(value):
                        value = os.path.join(path, value)
                    return os.path.abspath(value)
        basename = os.path.basename(os.path.normpath(path))
        candidate = os.path.join(path, basename)
        if os.path.isfile(candidate + ".index"):
            return candidate
    return path

def _resolve_eval_load_dir(args):
    if args.load_dir != DEFAULT_TRAIN_LOAD_DIR:
        return _resolve_checkpoint_prefix(args.load_dir)
    if args.eval_checkpoint == "latest":
        return _resolve_checkpoint_prefix(DEFAULT_EVAL_LOAD_DIR)
    return _resolve_checkpoint_prefix(args.load_dir)


def _run_eval_loop(args, env):
    print(f"Loading evaluation model from {args.load_dir}...")
    U.load_state(args.load_dir)
    print("Starting evaluation...")

    successful_episodes = 0
    failed_episodes = 0

    for episode in range(args.eval_episodes):
        obs = env.reset()
        episode_rewards = [0.0] * args.num_pursuit
        episode_step = 0
        v_star_prev = {i: 0.0 for i in range(args.num_pursuit)}

        for _ in range(args.episode_len):
            if args.display:
                env.render()

            input_states = input_state(obs, args.num_pursuit, args.map_width)
            action_pursuit = [
                env.AgentsContains[index].action(obs_i)
                for index, obs_i in zip(list(env.AgentsContains.keys())[:args.num_pursuit], input_states[:args.num_pursuit])
            ]

            pursuits_info, evasions_info, island_info = env.agents_info()
            action_evasion = env.sample_action(island_info, evasions_info, pursuits_info)
            action_n = action_pursuit + action_evasion

            if args.cbf_enable and args.cbf_apply_in_eval:
                apply_cbf_to_actions(env, args, action_n, v_star_prev)

            next_obs_n, rew_n, done_n = env.step(action_n)

            for i in range(args.num_pursuit):
                episode_rewards[i] += rew_n[i]

            obs = next_obs_n
            episode_step += 1

            terminal = ((episode_step % args.episode_len) == 0) or (done_n == [True] * args.num_pursuit) or env.fail_flags
            if terminal:
                is_success = done_n == [True] * args.num_pursuit
                if is_success:
                    successful_episodes += 1
                else:
                    failed_episodes += 1

                print(
                    "eval_episode:{},episode_step:{},success:{},reward:{}".format(
                        episode + 1,
                        episode_step,
                        is_success,
                        episode_rewards
                    )
                )
                break

    print(f"Evaluation finished. Success: {successful_episodes}, Failure: {failed_episodes}")


# Relative distance helpers.


def relative_dis(start_point, end_point):
    # start_point and end_point are agent objects.
    return math.sqrt((start_point.state.x - end_point.state.x) ** 2 + (start_point.state.y - end_point.state.y) ** 2)


def relative_dis_to_point(agent, point):
    distance = math.sqrt((point[0] - agent.state.x) ** 2 + (point[1] - agent.state.y) ** 2)
    return distance


def relative_angle(start_point, end_point):
    # start_point and end_point are agent objects.
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


# Build actor network input.


def input_state(obs, num_pur, map_width):
    """Build 11D actor input for each pursuer.

    Layout:
    [theta, v, w,
     teammate_1_angle, teammate_1_distance,
     teammate_2_angle, teammate_2_distance,
     teammate_3_angle, teammate_3_distance,
     evader_angle, evader_distance]
    Distances are normalized by map width.
    """
    return build_input_state(obs, num_pur, map_width)





def plot_reward_curve(reward_data, num_pursuit, window=50, save_path='reward_1.png'):
    plt.figure(figsize=(12, 6))

    
    for i in range(num_pursuit):
        rewards = reward_data[f'pursuit{i + 1}']
        episodes = reward_data['episode'][:len(rewards)]  
        plt.plot(episodes, rewards, alpha=0.3, linestyle='--')  

    
    avg_rewards = reward_data['average_reward']
    episodes = reward_data['episode'][:len(avg_rewards)]

    
    if len(avg_rewards) >= window:
        smoothed = np.convolve(avg_rewards, np.ones(window) / window, mode='valid')
        plt.plot(episodes[window - 1:], smoothed, label=f'Smoothed (window={window})', color='red', linewidth=2)
    else:
        plt.plot(episodes, avg_rewards, label='Average Reward', color='red', linewidth=2)

    dense_avg_rewards = reward_data.get('dense_average_reward', [])
    if len(dense_avg_rewards) > 0:
        dense_episodes = reward_data['episode'][:len(dense_avg_rewards)]
        if len(dense_avg_rewards) >= window:
            dense_smoothed = np.convolve(dense_avg_rewards, np.ones(window) / window, mode='valid')
            plt.plot(
                dense_episodes[window - 1:],
                dense_smoothed,
                label=f'Dense Reward (window={window})',
                color='blue',
                linewidth=2
            )
        else:
            plt.plot(dense_episodes, dense_avg_rewards, label='Dense Reward', color='blue', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward Curves (Total + Dense)')
    plt.legend()
    plt.grid(True)

    
    if len(avg_rewards) > 0:
        y_min = np.min(avg_rewards) * 0.9 if np.min(avg_rewards) < 0 else 0
        y_max = np.max(avg_rewards) * 1.1
        plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(save_path)  
    plt.close()  


def plot_dense_reward_curve(reward_data, num_pursuit, window=50, save_path='dense_reward.png'):
    plt.figure(figsize=(12, 6))

    for i in range(num_pursuit):
        rewards = reward_data.get(f'dense_pursuit{i + 1}', [])
        episodes = reward_data['episode'][:len(rewards)]
        plt.plot(episodes, rewards, alpha=0.3, linestyle='--')

    dense_avg_rewards = reward_data.get('dense_average_reward', [])
    episodes = reward_data['episode'][:len(dense_avg_rewards)]

    if len(dense_avg_rewards) >= window:
        smoothed = np.convolve(dense_avg_rewards, np.ones(window) / window, mode='valid')
        plt.plot(episodes[window - 1:], smoothed, label=f'Smoothed Dense (window={window})', color='blue', linewidth=2)
    else:
        plt.plot(episodes, dense_avg_rewards, label='Dense Average Reward', color='blue', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Dense Reward')
    plt.title('Dense Training Reward Curve')
    plt.legend()
    plt.grid(True)

    if len(dense_avg_rewards) > 0:
        y_min = np.min(dense_avg_rewards) * 0.9 if np.min(dense_avg_rewards) < 0 else 0
        y_max = np.max(dense_avg_rewards) * 1.1 if np.max(dense_avg_rewards) > 0 else 1
        if y_min == y_max:
            y_max = y_min + 1
        plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    with U.single_threaded_session():
        step = 0  
        timing = time.time()  
        args = parse_args()
        if args.eval_only:
            args.load_model = True
            args.load_dir = _resolve_eval_load_dir(args)
            print(f"Resolved eval load dir: {args.load_dir}")
        else:
            args.save_dir = _resolve_run_dir(args.save_dir, args.load_model)
            if args.load_model:
                args.load_dir = _resolve_checkpoint_prefix(args.load_dir)
        weight_calculator = WeightCalculator(
            sigma_d=args.policy_sigma_d,
            capture_radius=args.policy_capture_radius
        )
        save_base_name = 'reward_curve'
        dense_save_base_name = f'{save_base_name}_dense'
        output_dirs = None
        reward_curve = None
        if not args.eval_only:
            os.makedirs(args.save_dir, exist_ok=True)
            output_dirs = _prepare_output_dirs(args.save_dir)
            save_run_metadata(args, output_dirs)
            print(f"Resolved run dir: {args.save_dir}")
            reward_curve = {
                'episode': [],
                'average_reward': [],
                'dense_average_reward': []
            }
            for i in range(args.num_pursuit):
                reward_curve[f'pursuit{i + 1}'] = []
                reward_curve[f'dense_pursuit{i + 1}'] = []

        
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        tf.set_random_seed(args.random_seed)
        
        env = Env(args)
        obs = env.reset()
        saver = tf.train.Saver()  
        agents = env.AgentsContains
        U.initialize()  

        if args.eval_only:
            _run_eval_loop(args, env)
            raise SystemExit(0)

        if args.load_model:
            print(f'Loading previous state from {args.load_dir}...')
            U.load_state(args.load_dir)


        def _infer_offset_from_load_dir(path):
            
            m = re.search(r'ep(\d{3,6})', path.replace('\\', '/'))
            return int(m.group(1)) if m else 0


        base_offset = _infer_offset_from_load_dir(args.load_dir) if args.load_model else 0


        episode_rew = [[] for i in range(args.num_pursuit)]
        episode_dense_rew = [[] for i in range(args.num_pursuit)]
        print("Starting iterations...")

        successful_episodes = 0
        failed_episodes = 0

        for episode in range(args.episode_num): 

            episode_step = 0
            v_star_prev = {i: 0.0 for i in range(args.num_pursuit)}
            for t in range(args.episode_len):
                if args.display and episode >= args.episode_num - 10:
                    env.render()

                input_states = input_state(obs, args.num_pursuit, args.map_width)
                
                action_pursuit = [env.AgentsContains[index].action(obs) for index, obs in
                                  zip(list(env.AgentsContains.keys())[:args.num_pursuit],
                                      input_states[:args.num_pursuit])]

                pursuits_info, evasions_info, island_info = env.agents_info()
                action_evasion = env.sample_action(island_info, evasions_info, pursuits_info)
                action_n = action_pursuit + action_evasion

                if args.cbf_enable and args.cbf_apply_in_train:
                    apply_cbf_to_actions(env, args, action_n, v_star_prev)

                next_obs_n, rew_n, done_n = env.step(action_n)
                dense_rew_n = list(getattr(env, "last_reward_info", {}).get("dense_rewards", [0.0] * args.num_pursuit))
                if len(dense_rew_n) < args.num_pursuit:
                    dense_rew_n.extend([0.0] * (args.num_pursuit - len(dense_rew_n)))

                step_metrics = env.compute_encirclement_metrics()

                next_input_states = input_state(next_obs_n, args.num_pursuit, args.map_width)

                terminal = ((episode_step + 1) % args.episode_len == 0) or (done_n == [True] * args.num_pursuit) or env.fail_flags
                formation_errors = np.ones(args.num_pursuit, dtype=np.float32)
                if args.use_hper:
                    hper_dists = np.asarray(step_metrics.get("dists", []), dtype=np.float32)
                    hper_gaps = np.asarray(step_metrics.get("min_local_gaps", []), dtype=np.float32)
                    hper_target_gap = float(step_metrics.get("target_gap", (2.0 * math.pi) / max(args.num_pursuit, 1)))
                    if len(hper_dists) >= args.num_pursuit and len(hper_gaps) >= args.num_pursuit:
                        formation_errors = (
                            np.abs(hper_dists[:args.num_pursuit] - float(args.hper_rd))
                            + np.abs(hper_gaps[:args.num_pursuit] - hper_target_gap)
                        ).astype(np.float32)
                for i, key in enumerate(list(agents.keys())[:args.num_pursuit]):
                    done_for_buffer = bool(done_n[i] or env.fail_flags or terminal)
                    agents[key].experience(
                        input_states[i],
                        action_n[i],
                        rew_n[i],
                        next_input_states[i],
                        done_for_buffer,
                        formation_error=float(formation_errors[i])
                    )
                    episode_rew[i].append(rew_n[i])
                    episode_dense_rew[i].append(float(dense_rew_n[i]))

                input_states = next_input_states
                episode_step += 1
                step += 1

                if terminal:
                    
                    is_success = done_n == [True] * args.num_pursuit
                    if is_success:
                        successful_episodes += 1
                    else:
                        failed_episodes += 1
                    save_episode_trajectory(env, args, episode + 1, is_success, output_dirs)

                    avg_reward = float(np.mean([sum(episode_rew[i]) for i in range(args.num_pursuit)]))
                    outcome = "success" if is_success else "failure"
                    print(
                        f"episode:{episode + 1}, step:{episode_step}, total_step:{step}, "
                        f"avg_reward:{avg_reward:.3f}, success_count:{successful_episodes}, outcome:{outcome}"
                    )

                    episode_rewards = [sum(episode_rew[i]) for i in range(args.num_pursuit)]
                    agents_list = [env.AgentsContains[f"pursuit_{i + 1}"] for i in range(args.num_pursuit)]
                    for i, agent in enumerate(agents_list):
                        agent.recent_20_rewards.append(episode_rewards[i])
                        if len(agent.recent_20_rewards) > args.policy_reward_window:
                            agent.recent_20_rewards.pop(0)

                    reward_curve['episode'].append(episode + 1)  
                    reward_curve['average_reward'].append(
                        np.mean([sum(episode_rew[i]) for i in range(args.num_pursuit)])
                    )
                    reward_curve['dense_average_reward'].append(
                        np.mean([sum(episode_dense_rew[i]) for i in range(args.num_pursuit)])
                    )
                    for i in range(args.num_pursuit):
                        reward_curve[f'pursuit{i + 1}'].append(sum(episode_rew[i]))
                        reward_curve[f'dense_pursuit{i + 1}'].append(sum(episode_dense_rew[i]))

                    if (episode + 1) % 20 == 0:
                        _reward_curve_df(reward_curve, args.num_pursuit).to_csv(
                            os.path.join(output_dirs["latest"], "reward_curve_latest.csv"), index=False)

                    env.phase_info['surround_end_info'] = env.record_phase_info()

                    
                    if args.use_policy_sharing and (episode + 1) % args.policy_sharing_every == 0:
                        if env.phase_info['phase_switched']:
                            approach_weights = weight_calculator.compute_approach_weights(
                                env.phase_info['approach_start_info'],
                                env.phase_info['approach_end_info'],
                                time_steps=env.phase_info['approach_steps']
                            )
                            weights = approach_weights
                            if env.phase_info.get('surround_start_info') is not None and env.phase_info.get('surround_end_info') is not None:
                                surround_weights = weight_calculator.compute_surround_weights(
                                    env.phase_info['surround_start_info'],
                                    env.phase_info['surround_end_info'],
                                    num_agents=args.num_pursuit
                                )
                                weights = WeightCalculator.combine_approach_and_surround_weights(
                                    approach_weights, surround_weights
                                )
                            agents_list = [env.AgentsContains[f"pursuit_{i+1}"] for i in range(args.num_pursuit)]

                            theta_fed_expr = aggregate_actor_parameters_tf(agents_list, weights)
                            theta_fed_snapshot = U.get_session().run(theta_fed_expr)
                            replay_lengths_before_arm = [
                                len(agent.replay_buffer) for agent in agents_list
                            ]

                            try:
                                for agent in agents_list:
                                    recent_rewards = np.mean(agent.recent_20_rewards) if len(
                                        agent.recent_20_rewards) > 0 else 0.0
                                    agent.apply_ARM_with_training(
                                        theta_fed=theta_fed_snapshot,
                                        recent_rewards=recent_rewards,
                                        env=env,
                                        agents=agents_list,
                                        eta=args.arm_eta,
                                        lambda_=args.arm_lambda,
                                        train_episodes=args.arm_train_episodes
                                    )
                            finally:
                                replay_lengths_after_arm = [
                                    len(agent.replay_buffer) for agent in agents_list
                                ]
                                if replay_lengths_after_arm != replay_lengths_before_arm:
                                    raise RuntimeError(
                                        "ARM evaluation modified formal replay buffer lengths: "
                                        f"before={replay_lengths_before_arm}, after={replay_lengths_after_arm}"
                                    )
                    for i in range(args.num_pursuit):
                        episode_rew[i].clear()
                        episode_dense_rew[i].clear()

                    obs = env.reset()
                    input_states = input_state(obs, args.num_pursuit, args.map_width)


                    for index in list(agents.keys())[:args.num_pursuit]:
                        agents[index].preupdate()  

                    train_contains = [agents[index] for index in list(agents.keys())[:args.num_pursuit]]

                    for index in list(agents.keys())[:args.num_pursuit]:
                        loss = agents[index].update(train_contains, step)

                    global_ep = base_offset + (episode + 1)
                    if ((episode + 1) % args.latest_save_every) == 0:
                        latest_dir = output_dirs["latest"]
                        latest_prefix = _latest_ckpt_prefix(latest_dir)
                        U.save_state(latest_prefix, saver=saver)
                        df = _reward_curve_df(reward_curve, args.num_pursuit)
                        df.to_csv(os.path.join(latest_dir, "reward_curve_latest.csv"), index=False)
                        if args.save_plots:
                            plot_reward_curve(reward_curve, args.num_pursuit,
                                              save_path=os.path.join(latest_dir, "reward_curve_latest.png"))
                            plot_dense_reward_curve(reward_curve, args.num_pursuit,
                                                    save_path=os.path.join(latest_dir, "dense_reward_curve_latest.png"))
                        print(f"latest checkpoint saved @ {global_ep} -> {latest_prefix}")

                    if ((episode + 1) % args.save_every) == 0:
                        ckpt_dir = _ckpt_dir(args.save_dir, global_ep)
                        ckpt_prefix = _ckpt_prefix(args.save_dir, global_ep)
                        U.save_state(ckpt_prefix, saver=saver)
                        df = _reward_curve_df(reward_curve, args.num_pursuit)
                        df.to_csv(os.path.join(ckpt_dir, f"reward_curve_ep{global_ep:04d}.csv"), index=False)
                        if args.save_plots:
                            plot_reward_curve(reward_curve, args.num_pursuit,
                                              save_path=os.path.join(ckpt_dir, f"reward_curve_ep{global_ep:04d}.png"))
                            plot_dense_reward_curve(reward_curve, args.num_pursuit,
                                                    save_path=os.path.join(ckpt_dir, f"dense_reward_curve_ep{global_ep:04d}.png"))
                        print(f"checkpoint saved @ {global_ep} -> {ckpt_prefix}")

                    break

                for index in list(agents.keys())[:args.num_pursuit]:
                    agents[index].preupdate()  

                train_contains = [agents[index] for index in list(agents.keys())[:args.num_pursuit]]

                for index in list(agents.keys())[:args.num_pursuit]:
                    loss = agents[index].update(train_contains, step)

        print(f'training finishes, time spent: {datetime.timedelta(seconds=int(time.time() - timing))}')
        print(f"Successful Episode Num: {successful_episodes}", f"Failed Episode Num: {failed_episodes}")

        latest_prefix = _latest_ckpt_prefix(output_dirs["latest"])
        U.save_state(latest_prefix, saver=saver)
        _reward_curve_df(reward_curve, args.num_pursuit).to_csv(
            os.path.join(output_dirs["latest"], "reward_curve_latest.csv"), index=False)
        if args.save_plots:
            plot_reward_curve(reward_curve, args.num_pursuit,
                              save_path=os.path.join(output_dirs["latest"], "reward_curve_latest.png"))
            plot_dense_reward_curve(reward_curve, args.num_pursuit,
                                    save_path=os.path.join(output_dirs["latest"], "dense_reward_curve_latest.png"))
        print(f"Training finished. Outputs saved to {args.save_dir}")

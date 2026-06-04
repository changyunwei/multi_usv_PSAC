import numpy as np
import tensorflow as tf
import matd3.common.tf_util as U
from matd3.common.distributions import make_pdtype
from matd3.trainer.hper_replay_buffer import HPERReplayBuffer
from psac_matd3.utils.cbf import apply_cbf_to_actions
import tensorflow.contrib.layers as layers
import math


def _clip_pursuit_action_batch(actions):
    actions = np.asarray(actions, dtype=np.float32).copy()
    actions[..., 0] = np.clip(actions[..., 0], 0.0, 1.0)
    actions[..., 1] = np.clip(actions[..., 1], -1.0, 1.0)
    return actions



def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]



def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)
    
    
    return U.function([], [], updates=[expression])






def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None,
            local_q_func=False, num_units=64, scope="trainer", reuse=None):
    """
    :param make_obs_ph_n:
    :param act_space_n:
    :param agent_idx:
    :param p_func: actor model function
    :param q_func: critic model function
    :param optimizer:
    :param grad_norm_clipping:
    :param local_q_func:
    :param num_units:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        obs_ph_n = [tf.layers.flatten(obs_ph) for obs_ph in make_obs_ph_n]
        
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        
        q = q_func(q_input, 1, scope="q_func" + str(1), reuse=True, num_units=num_units)[:, 0]

        loss = -tf.reduce_mean(q) + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        train = U.function(inputs=make_obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[make_obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([make_obs_ph_n[p_index]], p)

        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[make_obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, q_function_idx, optimizer, grad_norm_clipping=None,
            local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        obs_ph_n = [tf.layers.flatten(obs_ph) for obs_ph in make_obs_ph_n]
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)

        q = q_func(q_input, 1, scope="q_func" + str(q_function_idx), num_units=num_units)[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func" + str(q_function_idx)))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))
        loss = q_loss

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        train = U.function(inputs=make_obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(make_obs_ph_n + act_ph_n, q)

        target_q = q_func(q_input, 1, scope="target_q_func" + str(q_function_idx), num_units=num_units)[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func" + str(q_function_idx)))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(make_obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}



def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out




class state:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.w = 0.0


class Agent:
    def __init__(self, name, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.agent_args = args
        self.state = state()
        self.name = name
        self.max_v = args.agent_maxspeed
        self.max_w = args.agent_max_w

        self.n = len(obs_shape_n)  
        self.agent_index = agent_index  
        self.recent_20_rewards = []  
        obs_ph_n = []
        for i in range(self.n):  
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())

        self.q_train1, self.q_update1, self.q_debug1 = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_function_idx=1,
            q_func=mlp_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.agent_args.lr),
            grad_norm_clipping=self.agent_args.grad_norm_clipping,
            local_q_func=local_q_func,
            num_units=self.agent_args.num_units
        )
        self.q_train2, self.q_update2, self.q_debug2 = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=mlp_model,
            q_function_idx=2,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.agent_args.lr),
            grad_norm_clipping=self.agent_args.grad_norm_clipping,
            local_q_func=local_q_func,
            num_units=self.agent_args.num_units
        )

        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=mlp_model,
            q_func=mlp_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.agent_args.lr),
            grad_norm_clipping=self.agent_args.grad_norm_clipping,
            local_q_func=local_q_func,
            num_units=self.agent_args.num_units
        )
        self.replay_buffer = HPERReplayBuffer(self.agent_args.replay_buffer_capacity)
        self.min_replay_buffer_len = self.agent_args.min_replay_buffer_len
        self.replay_sample_index = None


    def action(self, obs):  
        return self.act(obs[None])[0]


    def experience(self, obs, act, rew, new_obs, done, formation_error=1.0):
        self.replay_buffer.add(obs, act, rew, new_obs, float(done), formation_error=formation_error)


    def preupdate(self):
        self.replay_sample_index = None

    @property
    def q_debug(self):
        return self.q_debug1

    def _current_hper_beta(self, train_step):
        beta_start = float(getattr(self.agent_args, "hper_beta_start", 0.8))
        beta_end = float(getattr(self.agent_args, "hper_beta_end", 0.2))
        total_steps = max(
            1,
            int(getattr(self.agent_args, "episode_len", 1)) * int(getattr(self.agent_args, "episode_num", 1))
        )
        progress = float(np.clip(float(train_step) / float(total_steps), 0.0, 1.0))
        return beta_start + (beta_end - beta_start) * progress

    def update(self, agents, train_step):
        min_len_all = min(len(agents[i].replay_buffer) for i in range(self.n))
        if min_len_all < max(self.min_replay_buffer_len, self.agent_args.batch_size):
            return


        if not train_step % self.agent_args.update_rate == 0:
            return

        lens = [len(agents[i].replay_buffer) for i in range(self.n)]
        base_i = int(np.argmin(lens))
        use_hper = bool(getattr(self.agent_args, "use_hper", True))
        hper_beta = self._current_hper_beta(train_step)
        hper_alpha1 = float(getattr(self.agent_args, "hper_alpha1", 0.6))
        hper_alpha2 = float(getattr(self.agent_args, "hper_alpha2", 0.4))
        self.replay_sample_index = agents[base_i].replay_buffer.generate_sample_indices(
            self.agent_args.batch_size,
            use_hper=use_hper,
            beta=hper_beta,
            alpha1=hper_alpha1,
            alpha2=hper_alpha2
        )

        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
        if self.agent_args.use_critic_noise:
            for agent_idx in range(self.n):
                noise = np.random.normal(0, self.agent_args.critic_action_noise_stddev,
                                         size=target_act_next_n[agent_idx].shape)
                clipped_noise = np.clip(noise, -self.agent_args.action_noise_clip, self.agent_args.action_noise_clip)
                target_act_next_n[agent_idx] = (target_act_next_n[agent_idx] + clipped_noise).tolist()
        elif self.agent_args.use_critic_noise_self:
            noise = np.random.normal(0, self.agent_args.critic_action_noise_stddev,
                                     size=target_act_next_n[self.agent_index].shape)
            clipped_noise = np.clip(noise, -self.agent_args.action_noise_clip, self.agent_args.action_noise_clip)
            target_act_next_n[self.agent_index] = target_act_next_n[self.agent_index] + clipped_noise
        else:
            target_act_next_n = target_act_next_n
        target_act_next_n = [_clip_pursuit_action_batch(action) for action in target_act_next_n]
        target_q_next1 = self.q_debug1['target_q_values'](*(obs_next_n + target_act_next_n))
        target_q_next2 = self.q_debug2['target_q_values'](*(obs_next_n + target_act_next_n))
        target_q_next = np.min([target_q_next1, target_q_next2], 0)
        if self.agent_args.critic_zero_if_done:
            done_cond = done == True
            target_q_next[done_cond] = 0

        target_q = rew + self.agent_args.gamma * target_q_next
        q_loss = self.q_train1(*(obs_n + act_n + [target_q]))
        q_loss = self.q_train2(*(obs_n + act_n + [target_q]))

        
        q_curr1 = self.q_debug1['q_values'](*(obs_n + act_n))
        q_curr2 = self.q_debug2['q_values'](*(obs_n + act_n))
        q_curr_min = np.minimum(q_curr1, q_curr2)
        td_errors = np.abs(target_q - q_curr_min)
        self.replay_buffer.update_priorities(index, td_errors)

        if train_step % (self.agent_args.update_rate * self.agent_args.policy_update_rate) == 0:
            p_loss = self.p_train(*(obs_n + act_n))
            self.p_update()
            self.q_update1()
            self.q_update2()

        return [q_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]

    def apply_ARM_with_training(self, theta_fed, recent_rewards, env, agents,
                                eta=None, lambda_=None, train_episodes=None):
        """
        Evaluate an ARM candidate policy without modifying formal training state.
        """
        from env import Env

        eta = self.agent_args.arm_eta if eta is None else eta
        lambda_ = self.agent_args.arm_lambda if lambda_ is None else lambda_
        train_episodes = self.agent_args.arm_train_episodes if train_episodes is None else train_episodes
        sess = U.get_session()

        local_actor_vars = U.scope_vars(self.name + "/p_func")
        local_actor_snapshot = sess.run(local_actor_vars)

        total_fed_reward = 0.0
        max_arm_steps = int(getattr(env.args, "episode_len", 0))
        if max_arm_steps <= 0:
            max_arm_steps = 500

        arm_env = Env(env.args)

        class _ArmPursuitProxy:
            def __init__(self, name, agent_index, args):
                self.agent_args = args
                self.state = state()
                self.name = name
                self.max_v = args.agent_maxspeed
                self.max_w = args.agent_max_w
                self.agent_index = agent_index

        for i in range(arm_env.num_pursuit):
            arm_env.AgentsContains[f"pursuit_{i + 1}"] = _ArmPursuitProxy(f"pursuit_{i + 1}", i, env.args)
        for i in range(arm_env.num_evasion):
            agent_index = arm_env.num_pursuit + i
            arm_env.AgentsContains[f"evasion_{i + 1}"] = EvasionAgent(f"evasion_{i + 1}", agent_index, env.args)

        agents_by_name = {agent.name: agent for agent in agents}
        arm_map_width = float(getattr(env.args, "map_width", 1.0))

        def _relative_angle(start_obj, end_obj):
            dx = end_obj.state.x - start_obj.state.x
            dy = end_obj.state.y - start_obj.state.y
            return math.atan2(dy, dx)

        def _relative_distance(start_obj, end_obj):
            dx = end_obj.state.x - start_obj.state.x
            dy = end_obj.state.y - start_obj.state.y
            return math.sqrt(dx ** 2 + dy ** 2) / arm_map_width

        def _build_11d_obs(raw_obs, my_name):
            pursuit_names = list(raw_obs.keys())[:arm_env.num_pursuit]
            evasion_name = list(raw_obs.keys())[arm_env.num_pursuit]
            pursuer_obj = raw_obs[my_name]
            evader_obj = raw_obs[evasion_name]

            obs_tmp = [
                pursuer_obj.state.theta * math.pi / 180,
                pursuer_obj.state.v,
                pursuer_obj.state.w
            ]

            for other_name in pursuit_names:
                if other_name == my_name:
                    continue
                other_obj = raw_obs[other_name]
                obs_tmp.append(_relative_angle(pursuer_obj, other_obj))
                obs_tmp.append(_relative_distance(pursuer_obj, other_obj))

            obs_tmp.append(_relative_angle(pursuer_obj, evader_obj))
            obs_tmp.append(_relative_distance(pursuer_obj, evader_obj))
            return np.array(obs_tmp, dtype=np.float32)

        assign_ops = [var.assign(fed_var) for var, fed_var in zip(local_actor_vars, theta_fed)]
        try:
            sess.run(assign_ops)
            for _ in range(train_episodes):
                raw_obs = arm_env.reset()
                pursuit_names = list(raw_obs.keys())[:arm_env.num_pursuit]
                if self.name not in pursuit_names:
                    continue

                agent_idx = pursuit_names.index(self.name)
                obs = _build_11d_obs(raw_obs, self.name)
                episode_reward = 0.0
                arm_step = 0
                v_star_prev = {i: 0.0 for i in range(arm_env.num_pursuit)}

                while arm_step < max_arm_steps:
                    pursuit_actions = []
                    for pursuit_name in pursuit_names:
                        pursuit_obs = obs if pursuit_name == self.name else _build_11d_obs(raw_obs, pursuit_name)
                        local_agent = agents_by_name.get(pursuit_name)
                        if pursuit_name == self.name:
                            action = self.action(pursuit_obs)
                        elif local_agent is not None:
                            action = local_agent.action(pursuit_obs)
                        else:
                            action = np.zeros(env.args.action_dim, dtype=np.float32)
                        pursuit_actions.append(action)

                    pursuits_info, evasions_info, island_info = arm_env.agents_info()
                    evasion_actions = arm_env.sample_action(island_info, evasions_info, pursuits_info)
                    action_list = pursuit_actions + evasion_actions
                    if len(action_list) != arm_env.num_pursuit + arm_env.num_evasion:
                        raise RuntimeError("ARM action_list length does not match environment agent count")

                    if bool(getattr(env.args, "cbf_enable", True)):
                        apply_cbf_to_actions(arm_env, env.args, action_list, v_star_prev)

                    next_raw_obs, rewards, done_flags = arm_env.step(action_list)

                    if isinstance(rewards, (list, tuple, np.ndarray)):
                        my_reward = float(rewards[agent_idx])
                    else:
                        my_reward = float(rewards)

                    if isinstance(done_flags, (list, tuple, np.ndarray)):
                        if len(done_flags) > agent_idx:
                            my_done = bool(done_flags[agent_idx])
                        else:
                            my_done = bool(np.any(done_flags))
                    else:
                        my_done = bool(done_flags)

                    obs = _build_11d_obs(next_raw_obs, self.name)
                    episode_reward += my_reward
                    arm_step += 1

                    terminal = my_done or arm_env.fail_flags or arm_step >= max_arm_steps
                    if terminal:
                        break

                total_fed_reward += episode_reward
        finally:
            restore_ops = [var.assign(snapshot) for var, snapshot in zip(local_actor_vars, local_actor_snapshot)]
            sess.run(restore_ops)

        R_fed = total_fed_reward / max(train_episodes, 1)

        R_local = recent_rewards
        H_i = (R_fed - R_local) / (abs(R_local) + 1e-6)

        if H_i >= eta:
            final_ops = [var.assign(lambda_ * fed + (1 - lambda_) * local)
                         for var, fed, local in zip(local_actor_vars, theta_fed, local_actor_snapshot)]
            sess.run(final_ops)

        return H_i

class ArtificialPotentialField:
    def __init__(self, args):
        self.args = args
        self.evasions_info = np.array([])
        self.num_evasions = int
        self.x_Evasions = np.array([])
        self.y_Evasions = np.array([])

    def action(self, island_info, evasions_info, pursuits_info):
        self.evasions_info = evasions_info
        self.num_evasions = len(evasions_info)
        self.x_Evasions = evasions_info[:, 0]
        self.y_Evasions = evasions_info[:, 1]

        
        U_attract, grad_x_Attract, grad_y_Attract = self.Attract_Field_I_E(island_info)

        
        U_repel, grad_x_Repel, grad_y_Repel = self.Repel_Field_P_E(pursuits_info)

        
        grad_x = -grad_x_Repel - grad_x_Attract
        grad_y = -grad_y_Repel - grad_y_Attract

        action_evasions = self.cal_actions(grad_x, grad_y)  
        
        return action_evasions

    def Attract_Field_I_E(self, island_info):  
        KP = self.args.apf_attract_gain
        d_section = self.args.apf_attract_radius

        Island_position = np.array([island_info])  
        x_island, y_island = Island_position[0, 0], Island_position[0, 1]

        num_evasions = self.num_evasions
        Evasions_position = self.evasions_info
        x_evasions = Evasions_position[:, 0]  
        y_evasions = Evasions_position[:, 1]

        
        d_Island_Evasions = Distance(Evasions_position, Island_position)

        U_attract = np.zeros([num_evasions, 1])
        grad_x_Attract = np.zeros([num_evasions, 1])
        grad_y_Attract = np.zeros([num_evasions, 1])

        for i in range(num_evasions):
            if d_Island_Evasions[i] > d_section:
                U_attract[i] = d_section * KP * d_Island_Evasions[i] - 1 / 2 * KP * d_section ** 2
                grad_x_Attract[i] = KP * d_section * (x_evasions[i] - x_island) / math.sqrt(d_Island_Evasions[i])
                grad_y_Attract[i] = KP * d_section * (y_evasions[i] - y_island) / math.sqrt(d_Island_Evasions[i])
            else:
                U_attract[i] = 0.5 * KP * d_Island_Evasions[i] ** 2
                grad_x_Attract[i] = KP * (x_evasions[i] - x_island)
                grad_y_Attract[i] = KP * (y_evasions[i] - y_island)

        return U_attract, grad_x_Attract, grad_y_Attract  

    def Repel_Field_P_E(self, pursuits_info):
        ETA = self.args.apf_repel_gain
        d_section = self.args.apf_repel_radius
        epsilon = 1e-6

        num_pursuits = len(pursuits_info)
        Pursuits_position = pursuits_info[:, [0, 1]]
        x_Pursuits = Pursuits_position[:, 0]
        y_Pursuits = Pursuits_position[:, 1]

        num_evasions = self.num_evasions
        Evasions_position = self.evasions_info
        x_evasions = Evasions_position[:, 0]  
        y_evasions = Evasions_position[:, 1]

        
        d_Pursuits_Evasions = Distance(Evasions_position, Pursuits_position)

        U_repel = np.zeros([num_evasions, 1])
        grad_x_Repel = np.zeros([num_evasions, 1])
        grad_y_Repel = np.zeros([num_evasions, 1])

        for i in range(num_evasions):
            for j in range(num_pursuits):
                if d_Pursuits_Evasions[i, j] > d_section:
                    U_repel[i] += 0
                    grad_x_Repel[i] += 0
                    grad_y_Repel[i] += 0
                elif d_Pursuits_Evasions[i, j] > epsilon:
                    U_repel[i] += 0.5 * ETA * (1 / d_Pursuits_Evasions[i, j] - 1 / d_section)
                    d_grad_x_R = ETA * (
                            1 / d_section - 1 / d_Pursuits_Evasions[i, j]) * (
                                         x_evasions[i] - x_Pursuits[j]) / (d_Pursuits_Evasions[i, j] ** 3)
                    grad_x_Repel[i] += d_grad_x_R
                    d_grad_y_R = ETA * (
                            1 / d_section - 1 / d_Pursuits_Evasions[i, j]) * (
                                         y_evasions[i] - y_Pursuits[j]) / (d_Pursuits_Evasions[i, j] ** 3)
                    grad_y_Repel[i] += d_grad_y_R
                else:
                    
                    U_repel[i] += float('inf')  
                    grad_x_Repel[i] += 0
                    grad_y_Repel[i] += 0

        return U_repel, grad_x_Repel, grad_y_Repel

    def cal_actions(self, grad_x, grad_y):
        num_evasions = self.num_evasions

        theta = np.zeros([num_evasions, 1])  

        for i in range(num_evasions):
            
            if grad_x[i] == 0 and grad_y[i] == 0:
                theta[i] = self.evasions_info[i, 2]
            elif grad_x[i] == 0:
                if grad_y[i] > 0:
                    theta[i] = 90
                else:
                    theta[i] = -90
            else:
                theta[i] = math.atan(grad_y[i] / grad_x[i]) * 180 / math.pi
                if grad_y[i] == 0 and grad_x[i] > 0:
                    theta[i] = theta[i]
                elif grad_y[i] == 0 and grad_x[i] < 0:
                    theta[i] += 180
                elif grad_x[i] < 0 and grad_y[i] > 0:
                    theta[i] += 180
                elif grad_x[i] < 0 and grad_y[i] < 0:
                    theta[i] -= 180

        theta_current = self.evasions_info[:, 2][:, np.newaxis]
        delta_theta = theta - theta_current

        
        for i in range(num_evasions):
            if delta_theta[i] > 180:
                delta_theta[i] -= 360
            elif delta_theta[i] < -180:
                delta_theta[i] += 360

        
        angular_velocity = delta_theta * math.pi / (self.args.timestep * 180)

        
        actions = np.zeros((num_evasions, 2), dtype=np.float32)  
        actions[:, 0] = np.random.uniform(0, 1, 1)  
        

        actions[:, 1] = angular_velocity[:, 0]  

        return actions.astype(np.float32)  


def Distance(position_1=np.array([]), position_2=np.array([])):
    dx = np.zeros([len(position_1), len(position_2)], dtype=np.float64)
    dy = np.zeros([len(position_1), len(position_2)], dtype=np.float64)
    for i in range(len(position_1)):
        for j in range(len(position_2)):
            dx[i, j] = position_1[i, 0] - position_2[j, 0]
            dy[i, j] = position_1[i, 1] - position_2[j, 1]
    distance = np.hypot(dx, dy)  
    return distance


class EvasionAgent:
    def __init__(self, name, agent_index, args):
        self.agent_args = args
        self.state = state()
        self.name = name
        self.max_v = args.evader_maxspeed
        self.max_w = args.agent_max_w
        self.agent_index = agent_index  
        self.apf = ArtificialPotentialField(args)

    def eva_action(self, island_info, evasions_info, pursuits_info):
        
        actions = self.apf.action(island_info, evasions_info, pursuits_info)
        
        return actions

import tensorflow as tf


def aggregate_actor_parameters_tf(agents, weights):
    """Return weighted actor parameters for policy aggregation."""
    import matd3.common.tf_util as U

    agents_actor_vars = [U.scope_vars(agent.name + "/p_func") for agent in agents]
    fed_actor_vars = []

    num_layers = len(agents_actor_vars[0])
    for layer_idx in range(num_layers):
        aggregated_param = weights[0] * agents_actor_vars[0][layer_idx]
        for agent_idx in range(1, len(agents_actor_vars)):
            aggregated_param += weights[agent_idx] * agents_actor_vars[agent_idx][layer_idx]

        fed_actor_vars.append(aggregated_param)

    return fed_actor_vars

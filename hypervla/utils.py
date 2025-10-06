
import jax
import jax.numpy as jnp
import flax
import numpy as np


def get_base_model_shape(params):
    # initialize the base model
    # dummy_input = jnp.zeros(input_shape)
    # variables = model.init(jax.random.PRNGKey(0), dummy_input)['params']
    # get the shape of each param block
    param_shapes = jax.tree_map(lambda x: x.shape, params)
    flattened_param_shapes = flax.traverse_util.flatten_dict(param_shapes)
    # get the number of parameters in each param block
    param_num = jax.tree_map(lambda x: np.prod(x.shape), params)
    flattened_param_num = flax.traverse_util.flatten_dict(param_num)
    # get the total param number of the base model
    total_param_num = np.sum(list(flattened_param_num.values()))
    # get the param index of each param block
    cumsum_index = np.cumsum(list(flattened_param_num.values()))
    cumsum_index = np.concatenate([np.zeros(1, dtype=cumsum_index.dtype), cumsum_index])
    flattened_param_index = dict()
    for i, key in enumerate(flattened_param_shapes.keys()):
        flattened_param_index[key] = (cumsum_index[i], cumsum_index[i + 1])

    return total_param_num, flattened_param_shapes, flattened_param_index


def convert_flattened_params_to_dict_params(flattened_params, flattened_param_shapes, flattened_param_index):
    dict_params = dict()
    for key in flattened_param_shapes:
        start_idx, end_idx = flattened_param_index[key]
        # dict_params[key] = flattened_params[:, start_idx:end_idx].reshape((flattened_params.shape[0], ) + flattened_param_shapes[key])
        dict_params[key] = flattened_params[start_idx:end_idx].reshape(flattened_param_shapes[key])
    # dict_params = jax.tree_map(lambda idx, s: flattened_params[idx[0]:idx[1]].reshape(s), flattened_param_index, flattened_param_shapes)
    return flax.traverse_util.unflatten_dict(dict_params)



# if __name__ == '__main__':

#     rng = jax.random.PRNGKey(0)
#     model = BaseNetwork()
#     input_shape = (1, 256, 256, 3)
#     params = model.init(rng, np.zeros(input_shape))['params']
#     total_param_num, flattened_param_shapes, flattened_param_index = get_base_model_shape(params)
#     flattened_params = np.zeros((32, total_param_num))
#     dict_params = convert_flattened_params_to_dict_params(flattened_params, flattened_param_shapes, flattened_param_index)

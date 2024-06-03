import jax
import jax.numpy as jnp
from jax.numpy import einsum

import haiku as hk

from einops import rearrange

import jraph
from jraph.models import GraphNetwork


# Graph neural networks

def global_add_pooling(graph):
    return jraph.GraphReduceNodes(jax.vmap(jnp.sum))(graph)


def encoder(hidden_features, num_layers):
    embedding = jraph.GraphMapFeatures(
      embed_edge_fn=jax.vmap(hk.Linear(output_size=hidden_features)),
      embed_node_fn=jax.vmap(hk.Linear(output_size=hidden_features))
    )
    
    graph = embedding(graph)

    @jax.vmap
    @jraph.concatenated_args
    def update_edge_fn(features):
        net = hk.Sequential([
            hk.Linear(hidden_features), 
            jax.nn.relu
        ])
        return net(features)
    
    @jax.vmap
    @jraph.concatenated_args
    def update_nodes_fn(features):
        net = hk.Sequential([
            hk.Linear(hidden_features), 
            jax.nn.relu
        ])
        return net(features)
    

    rep_list = []    
    for _ in range(num_layers):
        graph = jraph.GraphNetwork(update_edge_fn, update_node_fn)(graph)
        rep_list.append(global_add_pooling(graph))
    
    reps = jnp.concatenate(rep_list, axis=-1)


    return reps  

 


def predictor():
    pass

def surrogate():
    pass



# Convolutional Neural Networks
# Maybe import some flax stuff in here

def decoder():
    pass 

# Train some stuff

def train_vae():
    pass

def train_surrogate():
    pass

def train_generator():
    pass

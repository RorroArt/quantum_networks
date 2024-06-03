import jax
import jax.numpy as jnp
import haiku as hk
import jraph
from jax.nn import relu, tanh
from typing import Callable, List, Optional, NamedTuple

# Constants
SMALL_NUMBER = 1e-7  # for avoiding numerical errors
LARGE_NUMBER = 1e10

def global_add_pooling(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    return jraph.GraphsTuple(
        nodes=jax.vmap(jnp.sum)(graph.nodes),
        edges=graph.edges,
        senders=graph.senders,
        receivers=graph.receivers,
        n_node=graph.n_node,
        n_edge=graph.n_edge,
        globals=graph.globals
    )

class EmbedEdgeModel(hk.Module):
    def __init__(self, input_dim_e: int, e_mlp: List[int], act: Callable = relu, dropout: float = 0.5):
        super().__init__()
        self.edge_mlp = self._build_mlp(input_dim_e, e_mlp, act, dropout)
        
    def _build_mlp(self, input_dim: int, mlp: List[int], act: Callable, dropout: float):
        layers = []
        for dim in mlp:
            layers.append(hk.Linear(dim))
            layers.append(act)
            layers.append(hk.Dropout(rate=dropout))
        return hk.Sequential(layers)

    def __call__(self, edge_attr: jnp.ndarray, mask_edge: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if mask_edge is not None:
            in_filtered = edge_attr[mask_edge]
        else:
            in_filtered = edge_attr
        
        out_filtered = self.edge_mlp(in_filtered)
        
        if mask_edge is not None:
            out = jnp.zeros((edge_attr.shape[0], out_filtered.shape[1]))
            out = out.at[mask_edge].set(out_filtered)
            return out
        else:
            return out_filtered

class EmbedNodeModel(hk.Module):
    def __init__(self, input_dim_n: int, n_mlp: List[int], act: Callable = relu, dropout: float = 0.5):
        super().__init__()
        self.node_mlp = self._build_mlp(input_dim_n, n_mlp, act, dropout)
        
    def _build_mlp(self, input_dim: int, mlp: List[int], act: Callable, dropout: float):
        layers = []
        for dim in mlp:
            layers.append(hk.Linear(dim))
            layers.append(act)
            layers.append(hk.Dropout(rate=dropout))
        return hk.Sequential(layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.node_mlp(x)

class EmbedMetaLayer(hk.Module):
    def __init__(self, edge_model: Optional[EmbedEdgeModel] = None, node_model: Optional[EmbedNodeModel] = None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray, edge_attr: Optional[jnp.ndarray] = None, mask_edge: Optional[jnp.ndarray] = None) -> NamedTuple:
        row, col = edge_index
        
        if self.edge_model is not None:
            edge_attr = self.edge_model(edge_attr, mask_edge)
        
        if self.node_model is not None:
            x = self.node_model(x)
        
        return x, edge_attr

class EdgeModel(hk.Module):
    def __init__(self, input_dim_e: int, e_mlp: List[int], input_dim_n: int, act: Callable = relu, dropout: float = 0.5):
        super().__init__()
        self.edge_mlp = self._build_mlp(input_dim_e + 2 * input_dim_n, e_mlp, act, dropout)
        
    def _build_mlp(self, input_dim: int, mlp: List[int], act: Callable, dropout: float):
        layers = []
        for dim in mlp:
            layers.append(hk.Linear(dim))
            layers.append(act)
            layers.append(hk.Dropout(rate=dropout))
        return hk.Sequential(layers)

    def __call__(self, src: jnp.ndarray, dest: jnp.ndarray, edge_attr: jnp.ndarray) -> jnp.ndarray:
        input = jnp.concatenate([src, dest, edge_attr], axis=-1)
        return self.edge_mlp(input)

class NodeModel(hk.Module):
    def __init__(self, input_dim_n: int, n_mlp: List[int], input_dim_e: int, e_mlp: List[int], act: Callable = relu, dropout: float = 0.5):
        super().__init__()
        self.node_mlp = self._build_mlp(input_dim_n + e_mlp[-1], n_mlp, act, dropout)
        
    def _build_mlp(self, input_dim: int, mlp: List[int], act: Callable, dropout: float):
        layers = []
        for dim in mlp:
            layers.append(hk.Linear(dim))
            layers.append(act)
            layers.append(hk.Dropout(rate=dropout))
        return hk.Sequential(layers)

    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray, edge_attr: jnp.ndarray) -> jnp.ndarray:
        row, col = edge_index
        out = jax.ops.segment_sum(edge_attr, col, num_segments=x.shape[0])
        out = jnp.concatenate([x, out], axis=-1)
        return self.node_mlp(out)

class Encoder(hk.Module):
    def __init__(self, input_dim_n: int, input_dim_e: int, em_node_mlp: List[int], em_edge_mlp: List[int], node_mlp: List[int], edge_mlp: List[int], num_fine: int, encoder_out_dim: int, dropout: float = 0.0, encoder_act: int = 1):
        super().__init__()
        self.num_fine = num_fine
        self.encoder_out_dim = encoder_out_dim
        encoder_act_fn = relu if encoder_act == 1 else tanh
        
        self.embed_edge_model = EmbedEdgeModel(input_dim_e, em_edge_mlp, encoder_act_fn, dropout)
        self.embed_node_model = EmbedNodeModel(input_dim_n, em_node_mlp, encoder_act_fn, dropout)
        self.edge_model = EdgeModel(input_dim_e, edge_mlp, input_dim_n, encoder_act_fn, dropout)
        self.node_model = NodeModel(input_dim_n, node_mlp, input_dim_e, edge_mlp, encoder_act_fn, dropout)
        
        self.embed = EmbedMetaLayer(edge_model=self.embed_edge_model, node_model=self.embed_node_model)
        self.gnblock = jraph.GraphNetwork(update_edge_fn=self.edge_model, update_node_fn=self.node_model)

    def __call__(self, graph: jraph.GraphsTuple) -> NamedTuple:
        graph = self.embed(graph.nodes, graph.senders, graph.edges)
        x_list = []
        for _ in range(self.num_fine):
            graph = self.gnblock(graph)
            x_list.append(global_add_pooling(graph))
        inpresentation = jnp.concatenate(x_list, axis=-1)
        encoder_out_mu = inpresentation
        return encoder_out_mu, inpresentation
    
class VAEEncoder(Encoder):
    def __init__(self, input_dim_n: int, input_dim_e: int, em_node_mlp: List[int] = [16], em_edge_mlp: List[int] = [16], node_mlp: List[int] = [16, 16, 16], edge_mlp: List[int] = [16, 16, 16], num_fine: int = 3, encoder_out_dim: int = 32, dropout: float = 0.0, encoder_act: int = 1):
        super().__init__(input_dim_n, input_dim_e, em_node_mlp, em_edge_mlp, node_mlp, edge_mlp, num_fine, encoder_out_dim, dropout, encoder_act)
        self.linear_mu = hk.Linear(encoder_out_dim)
        self.linear_logvar = hk.Linear(encoder_out_dim)

    def __call__(self, graph: jraph.GraphsTuple) -> NamedTuple:
        graph = self.embed(graph.nodes, graph.senders, graph.edges)
        x_list = []
        for _ in range(self.num_fine):
            graph = self.gnblock(graph)
            x_list.append(global_add_pooling(graph))
        
        inpresentation = jnp.concatenate(x_list, axis=-1)
        encoder_out_mu = self.linear_mu(inpresentation)
        encoder_out_logvar = self.linear_logvar(inpresentation)
        return encoder_out_mu, encoder_out_logvar

    def reparameterize(self, mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(hk.next_rng_key(), std.shape)
        return mu + eps * std
    
class Predictor(hk.Module):
    def __init__(self, encoder_out_dim: int = 32, mlp_pre: List[int] = [50, 50, 50], predictor_act: int = 2, dropout: float = 0.0):
        super().__init__()
        self.encoder_out_dim = encoder_out_dim
        self.dropout = dropout
        self.mlp_pre = mlp_pre
        self.predictor_act = relu if predictor_act == 1 else tanh

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        layers = []
        predict_mlp_input_dim = self.encoder_out_dim
        for idx, dim in enumerate(self.mlp_pre):
            layers.append(hk.Linear(dim, name=f"predictor_linear{idx+1}"))
            layers.append(self.predictor_act)
            layers.append(hk.Dropout(self.dropout))
            predict_mlp_input_dim = dim
        layers.append(hk.Linear(1, name="linear_predictor_readout"))
        
        predictor = hk.Sequential(layers)
        return predictor(z)
    
class DeepSurrogate(hk.Module):
    def __init__(self, encoder: hk.Module, predictor: hk.Module):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        encoder_out_mu, _ = self.encoder(graph)
        pre_mu = self.predictor(encoder_out_mu)
        return pre_mu

    def predict(self, graph: jraph.GraphsTuple, weights_nodes=None, weights_edges=None, NSample=10, uncertainty=True, soft=False, tau=None, y_test=None):
        if uncertainty:
            res = []
            for _ in range(NSample):
                if soft:
                    encoder_out_mu, _ = self.encoder.soft_forward(graph.nodes, graph.senders, graph.edges)
                else:
                    encoder_out_mu, _ = self.encoder(graph)
                labels_pre = self.predictor(encoder_out_mu)
                res.append(labels_pre)

            res = jnp.stack(res, axis=-1)
            pre_mu = jnp.mean(res, axis=-1)

            if y_test is not None:
                rmse = jnp.sqrt(jnp.mean((y_test - pre_mu) ** 2))

                ll = (logsumexp(-0.5 * tau * (y_test[:, None] - res) ** 2, axis=1) - np.log(NSample)
                      - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau))
                test_ll = jnp.mean(ll)
            else:
                rmse, test_ll = None, None

            return pre_mu, jnp.std(res, axis=-1), rmse, test_ll
        else:
            if soft:
                encoder_out_mu, _ = self.encoder.soft_forward(graph.nodes, graph.senders, graph.edges)
            else:
                encoder_out_mu, _ = self.encoder(graph)
            labels_pre = self.predictor(encoder_out_mu)

            if y_test is not None:
                rmse = jnp.sqrt(jnp.mean((y_test - labels_pre) ** 2))

            return labels_pre, None, rmse, None

class GeneratorDeConv(hk.Module):
    def __init__(self, max_nodes: int = 30, input_dim: int = 32, num_node_type: int = 5, num_edge_type: int = 3, channels: List[int] = [64, 32, 32, 1], kernels: List[int] = [3, 3, 3, 3], strides: List[tuple] = [(1, 1), (1, 4), (1, 2), (1, 2)], paddings: List[tuple] = [(0, 0), (0, 0), (0, 1), (0, 1)], act: int = 1, dropout: float = 0.0, dataset: str = "qm9"):
        super().__init__()
        
        self.dataset = dataset
        self.input_dim = input_dim
        self.num_node_type = num_node_type
        self.num_edge_type = num_edge_type
        self.max_nodes = max_nodes
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.dropout = dropout
        
        if act == 1:
            self.act = relu
        elif act == 2:
            self.act = tanh
        else:
            self.act = leaky_relu

        layers = []
        for i in range(len(self.channels)):
            if i == 0:
                layers.append(hk.Conv2DTranspose(output_channels=self.channels[i], kernel_shape=self.kernels[i], stride=self.strides[i], padding=self.paddings[i], with_bias=False, name=f"ConvTranspose2d_{i+1}"))
            else:
                layers.append(hk.Conv2DTranspose(output_channels=self.channels[i], kernel_shape=self.kernels[i], stride=self.strides[i], padding=self.paddings[i], with_bias=False, name=f"ConvTranspose2d_{i+1}"))
            
            if i < len(self.channels) - 1:
                layers.append(hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99, name=f"BatchNorm2d_{i+1}"))
                layers.append(self.act)

        self.gen_model = hk.Sequential(layers)

    def __call__(self, Z: jnp.ndarray, temperature: float = 1.0, use_random: bool = False, use_hard: bool = True):
        b, _ = Z.shape
        n = self.max_nodes
        
        out = self.gen_model(Z.reshape(b, -1, 1, 1)).reshape(b, n, -1)
        out_nodes = out[:, :, :self.num_node_type + 1].reshape(b, n, -1)
        
        X = jax.nn.softmax(out_nodes, axis=-1)
        
        all_edges = jnp.array(jnp.triu_indices(n, 1)).astype(jnp.int32)
        
        out_edge = out[:, :, self.num_node_type + 1:].reshape(b, n, n, self.num_edge_type + 1)
        idx_triu = jnp.triu(jnp.ones((b, n, n)), k=1)
        out_edge = out_edge[idx_triu == 1].reshape(b, -1, self.num_edge_type + 1)
        
        visited_edges = jax.nn.softmax(out_edge, axis=-1)
        
        edge_index = all_edges
        batch = jnp.repeat(jnp.arange(b), n)
        edge_index = jnp.concatenate([edge_index, edge_index[::-1]], axis=-1)
        edge_attr = jnp.concatenate([visited_edges, visited_edges], axis=1)
        
        if use_hard:
            X_temp = jnp.zeros_like(X)
            for b_idx in range(b):
                for node_idx in range(n):
                    index_one = jnp.argmax(X[b_idx, node_idx, :])
                    X_temp = X_temp.at[b_idx, node_idx, index_one].set(1.0)

            visited_edges_temp = jnp.zeros_like(visited_edges)
            for b_idx in range(b):
                for edge_idx in range(out_edge.shape[1]):
                    index_one = jnp.argmax(visited_edges[b_idx, edge_idx, :])
                    visited_edges_temp = visited_edges_temp.at[b_idx, edge_idx, index_one].set(1.0)
            
            edge_attr_temp = jnp.concatenate([visited_edges_temp, visited_edges_temp], axis=1)
            log_prob_temp = jnp.sum(jnp.log(X + SMALL_NUMBER) * X_temp) + jnp.sum(jnp.log(visited_edges + SMALL_NUMBER) * visited_edges_temp)

            return [X[:, :, 1:], edge_index, edge_attr[:, :, 1:], 1.0 - X[:, :, 0], 1.0 - edge_attr[:, :, 0], batch], [X_temp[:, :, 1:], edge_index, edge_attr_temp[:, :, 1:], 1.0 - X_temp[:, :, 0], 1.0 - edge_attr_temp[:, :, 0], batch], log_prob_temp
        else:
            return [X[:, :, 1:], edge_index, edge_attr[:, :, 1:], 1.0 - X[:, :, 0], 1.0 - edge_attr[:, :, 0], batch], None, None

    def reparameterize(self, mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(hk.next_rng_key(), std.shape)
        return mu + eps * std
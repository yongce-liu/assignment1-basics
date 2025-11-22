import math
import torch
from copy import deepcopy
from torch.nn import Module
from jaxtyping import Float


class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._factory_params = {"dtype": dtype, "device": device}
        self.weights = torch.nn.Parameter(
            torch.empty(size=(self.in_features, self.out_features), **self._factory_params)
        )
        self.reset_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.to(device=x.device)

    def reset_weights(self) -> None:
        sigma = (2 / (self.in_features + self.out_features)) ** 0.5
        self.weights = torch.nn.init.trunc_normal_(self.weights, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def load_weights(self, weights: Float[torch.Tensor, " d_in d_out"]) -> None:
        # self.weights = torch.nn.Parameter(self.load_state_dict(weights))
        self.weights.data.copy_(weights)


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self._factory_params = {"device": device, "dtype": dtype}
        self.weights = torch.nn.Parameter(torch.empty(size=(num_embeddings, embedding_dim), **self._factory_params))
        self.reset_weights()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids].detach().to(token_ids.device)

    def reset_weights(self) -> None:
        sigma = 1
        self.weights = torch.nn.init.trunc_normal_(self.weights, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def load_weights(self, weights: Float[torch.Tensor, "vocab_size d_model"]) -> None:
        # self.weights = torch.nn.Parameter(self.load_state_dict(weights))
        self.weights.data.copy_(weights)


class RmsNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self._factory_params = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.eps = eps
        self.gain = torch.nn.Parameter(torch.empty(size=(self.d_model,), **self._factory_params))

    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        res = x / torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) / self.d_model + self.eps) * self.gain.to(x.device)

        return res.to(in_dtype)

    def reset_weights(self) -> None:
        self.gain = torch.nn.Parameter(torch.ones(size=(self.d_model,), **self._factory_params))

    def load_weights(self, weights: Float[torch.Tensor, " d_model"]) -> None:
        self.gain.data.copy_(weights)


class SwiGLUFFN(Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self._factory_params = {"device": device, "dtype": dtype}
        self.transforms: list[Linear] = [
            Linear(d_model, d_ff, **self._factory_params),
            Linear(d_ff, d_model, **self._factory_params),
            Linear(d_model, d_ff, **self._factory_params),
        ]

    def reset_weights(self) -> None:
        for transform in self.transforms:
            transform.reset_weights()

    def load_weights(self, weights: list[torch.Tensor]) -> None:
        for weight, transform in zip(weights, self.transforms):
            transform.load_weights(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.transforms[0].forward(x)
        x2 = self.transforms[2].forward(x)
        x1 = x0 * torch.sigmoid(x0) * x2

        return self.transforms[1].forward(x1)


class RoPE(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self._factory_params = {"device": device, "dtype": dtype}
        self.const_theta = theta
        self.d_k = d_k
        pos_mat = []
        for i in range(max_seq_len):
            pos_mat.append(self.get_rot_mat(i).T)  # store the transpose mat, y = Rx -> y^t = x^TR^T
        self.rot_mat: torch.Tensor = torch.stack(pos_mat)  # (max_seq_len, d_k, d_k)
        self.register_buffer(name="rotation matrix", tensor=self.rot_mat, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        input x: [... seq_len d_k]
        output: [... seq_len d_k]
        """
        res = torch.einsum("...k, ...kj -> ...j", x, self.rot_mat.to(x.device)[token_positions])
        return res

    def get_rot_mat(self, pos: int):
        rot_i = []
        for k in range(0, int(self.d_k / 2)):
            theta = pos / (self.const_theta ** (2 * k / self.d_k))
            rot_i.append(self.mat_func(theta, **self._factory_params))

        return torch.block_diag(*rot_i)

    @staticmethod
    def mat_func(theta: float, **kwargs) -> torch.Tensor:
        """
        theta in rad
        """
        return torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], **kwargs)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - max_x
    exp_x = torch.exp(x_stable)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    p = exp_x / sum_exp_x

    return p


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Q: b, ..., s, d_k
    K: b, ..., s, d_k
    V: b, ..., s, d_v
    mask: bool
    -> : b, ..., d_v
    """
    sqrt_d_k = Q.shape[-1] ** 0.5
    val = torch.einsum("... n k, ... m k -> ... n m", Q, K) / sqrt_d_k

    if mask is not None:
        neg_inf = torch.finfo(val.dtype).min
        val = val.masked_fill(~mask, value=neg_inf)
    val = softmax(val, dim=-1)
    atten = torch.einsum("... n m, ... m v -> ... n v", val, V)
    return atten


class MultiHeadAttention(Module):
    def __init__(self, d_in: int, d_model: int, d_v: int, num_heads: int):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_v = d_v
        self.num_heads = num_heads

        self._d_k = int(d_model / num_heads)
        self._d_v = int(d_v / num_heads)
        head_transforms = [
            Linear(in_features=self.d_in, out_features=self._d_k),  # Q
            Linear(in_features=self.d_in, out_features=self._d_k),  # K
            Linear(in_features=self.d_in, out_features=self._d_v),  # V
        ]
        self.transforms = self.num_heads * [deepcopy(head_transforms)]
        self.output_transform = Linear(in_features=self.d_v, out_features=self.d_model)  # O

    def load_weights(self, weights: list[list[torch.Tensor]]) -> None:
        for i, trans in enumerate(self.transforms):
            if isinstance(weights[i], list):
                for j, mod in enumerate(trans):
                    mod.load_weights(weights=weights[i][j])
        self.output_transform.load_weights(weights[-1][-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for models in self.transforms:
            Q = models[0].forward(
                x
            )  # need rope for position encoding, however, in the assignment, we didn't have hyper=params
            K = models[1].forward(x)
            V = models[2].forward(x)
            mask = torch.tril(torch.ones(Q.shape[-2], K.shape[-2], dtype=torch.bool, device=Q.device))
            atten = attention(Q, K, V, mask)
            res.append(atten)
        res = torch.concatenate(res, dim=-1)

        return self.output_transform(res)

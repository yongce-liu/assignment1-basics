import math
import torch
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

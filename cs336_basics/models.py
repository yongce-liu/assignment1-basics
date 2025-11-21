import torch
from jaxtyping import Float


class Linear(torch.nn.Module):
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
        print(x.shape)
        print(self.weights.shape)

        return x @ self.weights.to(device=x.device)

    def reset_weights(self) -> None:
        sigma = (2 / (self.in_features + self.out_features)) ** 0.5
        self.weights = torch.nn.init.trunc_normal_(self.weights, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def load_weights(self, weights: Float[torch.Tensor, " d_out d_in"]) -> None:
        # self.weights = torch.nn.Parameter(self.load_state_dict(weights))
        self.weights.data.copy_(weights.T)


class Embedding(torch.nn.Module):
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
import torch
import torch.nn as nn


class SimpleTransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        context_length: int = 512,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        feedforward_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[1]
        if sequence_length > self.context_length:
            token_ids = token_ids[:, -self.context_length :]
            sequence_length = self.context_length

        positions = torch.arange(sequence_length, device=token_ids.device).unsqueeze(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=token_ids.device),
            diagonal=1,
        ).bool()
        hidden = self.transformer(hidden, mask=mask)
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, generated_tokens: int) -> torch.Tensor:
        self.eval()
        for _ in range(generated_tokens):
            logits = self(token_ids[:, -self.context_length :])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids

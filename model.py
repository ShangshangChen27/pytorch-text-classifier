# model.py
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, num_classes: int = 2, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        """
        input_ids: [batch_size, seq_len]  (padded)
        lengths:   [batch_size]
        """
        # [B, T, D]
        emb = self.embedding(input_ids)

        # mask: True where token != PAD
        mask = (input_ids != self.pad_id).unsqueeze(-1)  # [B, T, 1]
        emb = emb * mask                                 # PAD positions become 0

        # sum over time then divide by lengths -> mean pooling
        summed = emb.sum(dim=1)                           # [B, D]
        lengths = lengths.clamp(min=1).unsqueeze(-1)      # [B, 1]
        pooled = summed / lengths                         # [B, D]

        logits = self.classifier(pooled)                  # [B, 2]
        return logits


if __name__ == "__main__":
    # quick sanity check
    B, T, V = 3, 4, 11
    x = torch.randint(0, V, (B, T))
    lengths = torch.tensor([T, T, T])
    model = TextClassifier(vocab_size=V)
    out = model(x, lengths)
    print("logits shape:", out.shape)  # should be [3, 2]
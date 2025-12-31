# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import ToyTextDataset, collate_batch
from model import TextClassifier

def train():
    # 1) 数据
    dataset = ToyTextDataset()
    pad_id = dataset.vocab[dataset.pad_token]
    vocab_size = len(dataset.vocab)

    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        collate_fn=collate_batch,
    )

    # 2) 模型
    model = TextClassifier(vocab_size=vocab_size, embed_dim=32, num_classes=2, pad_id=pad_id)

    # 3) 损失 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # 4) 训练循环
    model.train()
    for epoch in range(30):
        total_loss = 0.0
        correct = 0
        total = 0

        for input_ids, labels, lengths in loader:
            optimizer.zero_grad()

            logits = model(input_ids, lengths)          # [B, 2]
            loss = criterion(logits, labels)            # labels: [B]

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | acc={acc:.2f}")

    # 5) 简单预测展示
    model.eval()
    examples = [
        "I love this film",
        "This movie is terrible",
        "I hate this film",
        "This film is great",
    ]
    print("\nPredictions:")
    for text in examples:
        ids = torch.tensor(dataset.encode(text), dtype=torch.long).unsqueeze(0)  # [1, T]
        lengths = torch.tensor([ids.size(1)], dtype=torch.long)
        with torch.no_grad():
            logits = model(ids, lengths)
            pred = logits.argmax(dim=-1).item()
        print(f"{text:25s} -> pred={pred} (1=pos, 0=neg)")

if __name__ == "__main__":
    train()
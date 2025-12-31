# data.py
import torch
from torch.utils.data import Dataset

class ToyTextDataset(Dataset):
    """
    A tiny sentiment dataset:
    label 1 = positive
    label 0 = negative
    """

    def __init__(self):
        self.samples = [
            ("I love this movie", 1),
            ("This film is great", 1),
            ("I hate this movie", 0),
            ("This film is terrible", 0),
            ("I really love this film", 1),
            ("This movie is really terrible", 0),
        ]

        # Build a tiny vocab from the samples
        words = set()
        for text, _ in self.samples:
            for w in text.lower().split():
                words.add(w)

        # Reserve 0 for <PAD>
        self.pad_token = "<PAD>"
        self.vocab = {self.pad_token: 0}
        for i, w in enumerate(sorted(words), start=1):
            self.vocab[w] = i

        self.id2word = {i: w for w, i in self.vocab.items()}

    def encode(self, text: str):
        tokens = text.lower().split()
        return [self.vocab[t] for t in tokens]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        x = torch.tensor(self.encode(text), dtype=torch.long)  # [seq_len]
        y = torch.tensor(label, dtype=torch.long)              # scalar
        return x, y


def collate_batch(batch):
    """
    batch: list of (x, y) where x is [seq_len] (variable length)
    We pad them to the same length -> [batch_size, max_len]
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)

    max_len = max(lengths).item()
    padded = torch.zeros((len(xs), max_len), dtype=torch.long)  # PAD=0

    for i, x in enumerate(xs):
        padded[i, : len(x)] = x

    y = torch.stack(ys)  # [batch_size]
    return padded, y, lengths


if __name__ == "__main__":
    ds = ToyTextDataset()
    print("Vocab size:", len(ds.vocab))
    print("Example vocab items:", list(ds.vocab.items())[:10])

    x0, y0 = ds[0]
    print("\nOne sample:")
    print("x:", x0)
    print("y:", y0)

    # quick collate test
    batch = [ds[i] for i in range(3)]
    padded, y, lengths = collate_batch(batch)
    print("\nAfter collate_batch:")
    print("padded shape:", padded.shape)
    print("padded:", padded)
    print("y:", y)
    print("lengths:", lengths)
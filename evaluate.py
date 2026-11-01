"""Evaluates the brain tumor classifier with repeated stratified cross validation.

With only 800 images, one 80/20 split swings by several points depending on the shuffle, so
this reports a mean and spread across 5 folds instead of a single test score.
Usage: python evaluate.py   (reads archive.zip, writes results/)
"""
import io
import json
import pathlib
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.nn import functional as F

SIZE = 100
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(path="archive.zip"):
    xs, ys = [], []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if name.lower().endswith(".jpg"):
                image = Image.open(io.BytesIO(z.read(name))).convert("L").resize((SIZE, SIZE))
                xs.append(np.asarray(image, dtype=np.float32) / 255)
                ys.append(1 if "/tumor/" in name else 0)
    return np.stack(xs)[:, None], np.array(ys)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers, channels = [], 1
        for out in (32, 64, 128, 256):
            layers += [nn.Conv2d(channels, out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.2)]
            channels = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(256 * 6 * 6, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.features(x).flatten(1)).squeeze(1)


def probabilities(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(x[i:i + 64]).to(DEVICE))).cpu().numpy()
                               for i in range(0, len(x), 64)])


def augment(x):
    if torch.rand(1) < 0.5:
        x = x.flip(-1)
    dx, dy = torch.randint(-5, 6, (2,)).tolist()
    return torch.roll(x, (dy, dx), dims=(2, 3))


def train(x, y, seed, epochs=60, patience=12):
    torch.manual_seed(seed)
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.15, stratify=y, random_state=seed)
    model = CNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    y_t = torch.tensor(y_tr, dtype=torch.float32)
    best, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(x_tr))
        for start in range(0, len(order), 16):
            batch = order[start:start + 16]
            loss = F.binary_cross_entropy_with_logits(
                model(augment(torch.from_numpy(x_tr[batch]).to(DEVICE))), y_t[batch].to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        p = np.clip(probabilities(model, x_val), 1e-6, 1 - 1e-6)
        val = -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
        if val < best:
            best, wait = val, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, pred), f1=f1_score(y, pred), sensitivity=recall_score(y, pred),
                specificity=recall_score(y, pred, pos_label=0), auc=roc_auc_score(y, p))


def main():
    x, y = load()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(x, y))
    cnn, base, oof = [], [], np.zeros(len(y))
    for i, (train_idx, test_idx) in enumerate(folds):
        lr = LogisticRegression(max_iter=500).fit(x[train_idx].reshape(len(train_idx), -1), y[train_idx])
        base.append(metrics(y[test_idx], lr.predict_proba(x[test_idx].reshape(len(test_idx), -1))[:, 1]))
        p = probabilities(train(x[train_idx], y[train_idx], seed=i), x[test_idx])
        oof[test_idx] = p
        cnn.append(metrics(y[test_idx], p))
        print("fold", i, cnn[-1], flush=True)

    def summary(rows):
        return {k: dict(mean=float(np.mean([r[k] for r in rows])), std=float(np.std([r[k] for r in rows]))) for k in rows[0]}

    out = pathlib.Path("results")
    out.mkdir(exist_ok=True)
    cm = confusion_matrix(y, (oof >= 0.5).astype(int))
    (out / "metrics.json").write_text(json.dumps(dict(cnn=summary(cnn), logistic_regression=summary(base),
                                                      out_of_fold_confusion=cm.tolist(), folds=cnn), indent=2))
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=160)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Healthy", "Tumor"])
    ax.set_yticks([0, 1], ["Healthy", "Tumor"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Out of fold predictions, all 800 images")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png")
    print(json.dumps(dict(cnn=summary(cnn), base=summary(base)), indent=1))


if __name__ == "__main__":
    main()

"""A small NumPy MLP with manual backprop + Adam — trained on the dataset."""

import numpy as np


def _relu(x):
    return np.maximum(x, np.float32(0.0))


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class MLP:
    """Fully-connected classifier: Linear → ReLU → … → Linear (logits)."""

    def __init__(self, sizes=(784, 256, 64, 10), seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.params = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            W = rng.normal(0.0, np.sqrt(2.0 / fan_in),
                           (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            self.params.append([W, b])

    def forward(self, x, params=None):
        """Returns the list of activations; the last entry is the logits."""
        params = params or self.params
        acts = [x]
        h = x
        for i, (W, b) in enumerate(params):
            h = h @ W + b
            if i < len(params) - 1:
                h = _relu(h)
            acts.append(h)
        return acts

    def predict(self, x, params=None):
        return np.argmax(self.forward(x, params)[-1], axis=1)

    def load(self, arrays):
        self.params = [[arrays[f"W{i}"], arrays[f"b{i}"]]
                       for i in range(len(self.sizes) - 1)]

    def loss_and_grads(self, x, y, params=None):
        """Softmax cross-entropy loss + gradients (manual backprop)."""
        params = params or self.params
        acts = self.forward(x, params)
        logits = acts[-1]
        n = len(x)
        probs = _softmax(logits)
        loss = float(-np.mean(np.log(probs[np.arange(n), y] + 1e-12)))
        delta = probs.copy()
        delta[np.arange(n), y] -= 1.0
        delta = (delta / n).astype(np.float32)

        grads = [None] * len(params)
        for i in range(len(params) - 1, -1, -1):
            W, _ = params[i]
            grads[i] = [acts[i].T @ delta, delta.sum(axis=0)]
            if i > 0:
                delta = (delta @ W.T) * (acts[i] > 0)
        return loss, grads


def evaluate(model, x, y, batch=512):
    """Batched loss + accuracy over a split."""
    total_loss, correct = 0.0, 0
    for s in range(0, len(x), batch):
        xb, yb = x[s:s + batch], y[s:s + batch]
        logits = model.forward(xb)[-1]
        probs = _softmax(logits)
        total_loss += float(-np.sum(
            np.log(probs[np.arange(len(xb)), yb] + 1e-12)))
        correct += int((probs.argmax(axis=1) == yb).sum())
    return total_loss / len(x), correct / len(x)


def train(model, data, epochs=6, batch_size=64, lr=2e-3, seed=0,
          eval_subset=2000):
    """Mini-batch Adam training; returns the learning history."""
    rng = np.random.default_rng(seed)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in model.params]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in model.params]
    x_train, y_train = data["x_train"], data["y_train"]
    xs, ys = x_train[:eval_subset], y_train[:eval_subset]

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    step = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            loss, grads = model.loss_and_grads(x_train[idx], y_train[idx])
            step += 1
            for i, ((W, b), (gW, gb)) in enumerate(zip(model.params, grads)):
                m[i][0] = beta1 * m[i][0] + (1 - beta1) * gW
                v[i][0] = beta2 * v[i][0] + (1 - beta2) * gW * gW
                m[i][1] = beta1 * m[i][1] + (1 - beta1) * gb
                v[i][1] = beta2 * v[i][1] + (1 - beta2) * gb * gb
                W -= lr * (m[i][0] / (1 - beta1 ** step)) / (
                    np.sqrt(v[i][0] / (1 - beta2 ** step)) + eps)
                b -= lr * (m[i][1] / (1 - beta1 ** step)) / (
                    np.sqrt(v[i][1] / (1 - beta2 ** step)) + eps)
        tr_loss, tr_acc = evaluate(model, xs, ys)
        va_loss, va_acc = evaluate(model, data["x_test"], data["y_test"])
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"epoch {epoch}: train_loss {tr_loss:.4f} "
              f"train_acc {tr_acc:.3f} | val_loss {va_loss:.4f} "
              f"val_acc {va_acc:.3f}")
    return history

import numpy as np


class MLP:

    def __init__(self, layer_sizes, learning_rate=0.01, batch_size=32, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.lr = learning_rate
        self.batch_size = batch_size

        self._init_weights()

        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def _init_weights(self):
        self.W = []
        self.b = []
        for i in range(self.L):
            std = np.sqrt(2.0 / self.layer_sizes[i])
            self.W.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * std)
            self.b.append(np.zeros((1, self.layer_sizes[i + 1])))

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def _softmax(self, Z):
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z_stable)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def _cross_entropy(self, y_true, y_pred):
        N = y_true.shape[0]
        eps = 1e-12
        log_probs = -np.log(y_pred[np.arange(N), y_true] + eps)
        return np.mean(log_probs)

    def forward(self, X):
        self.Z = []
        self.A = [X]

        A = X
        for l in range(self.L):
            Z = A @ self.W[l] + self.b[l]
            self.Z.append(Z)
            if l == self.L - 1:
                A = self._softmax(Z)
            else:
                A = self._relu(Z)
            self.A.append(A)

        return self.A[-1]

    def backward(self, y_true):
        N = y_true.shape[0]
        y_onehot = np.zeros((N, self.layer_sizes[-1]))
        y_onehot[np.arange(N), y_true] = 1

        dW = [None] * self.L
        db = [None] * self.L

        dZ = self.A[-1] - y_onehot

        for l in range(self.L - 1, -1, -1):
            dW[l] = (self.A[l].T @ dZ) / N
            db[l] = np.sum(dZ, axis=0, keepdims=True) / N

            if l > 0:
                dA = dZ @ self.W[l].T
                dZ = dA * self._relu_derivative(self.Z[l - 1])

        return dW, db

    def _update(self, dW, db):
        for l in range(self.L):
            self.W[l] -= self.lr * dW[l]
            self.b[l] -= self.lr * db[l]

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, verbose=True):
        N = X_train.shape[0]
        n_batches = max(1, N // self.batch_size)

        for epoch in range(epochs):
            perm = np.random.permutation(N)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            epoch_loss = 0
            epoch_acc = 0

            for i in range(n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)
                loss = self._cross_entropy(y_batch, y_pred)
                epoch_loss += loss * X_batch.shape[0]
                preds = np.argmax(y_pred, axis=1)
                epoch_acc += np.sum(preds == y_batch)

                dW, db = self.backward(y_batch)
                self._update(dW, db)

            epoch_loss /= N
            epoch_acc = epoch_acc / N
            self.train_loss.append(epoch_loss)
            self.train_acc.append(epoch_acc)

            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                self.val_loss.append(val_loss)
                self.val_acc.append(val_acc)

                if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch:4d}/{epochs} - "
                          f"train_loss: {epoch_loss:.4f}, train_acc: {epoch_acc:.4f} | "
                          f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
            else:
                if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch:4d}/{epochs} - "
                          f"train_loss: {epoch_loss:.4f}, train_acc: {epoch_acc:.4f}")

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        loss = self._cross_entropy(y, y_pred)
        preds = np.argmax(y_pred, axis=1)
        acc = np.mean(preds == y)
        return loss, acc

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def predict_proba(self, X):
        return self.forward(X)

    def saliency_map(self, X):
        self.forward(X)
        N = X.shape[0]
        C = self.layer_sizes[-1]
        pred_classes = np.argmax(self.A[-1], axis=1)

        dZ = self.A[-1].copy()
        dZ[np.arange(N), pred_classes] -= 1

        for l in range(self.L - 1, 0, -1):
            dA = dZ @ self.W[l].T
            dZ = dA * self._relu_derivative(self.Z[l - 1])

        dX = dZ @ self.W[0].T
        return np.abs(dX)

    def perturbation_importance(self, X, method='zero'):
        N, D = X.shape
        y_pred_orig = self.forward(X)
        pred_class_orig = np.argmax(y_pred_orig, axis=1)
        prob_orig = y_pred_orig[np.arange(N), pred_class_orig]

        if method == 'mean':
            perturb_value = np.mean(X, axis=0, keepdims=True)
        else:
            perturb_value = np.zeros((1, D))

        importance = np.zeros((N, D))
        for d in range(D):
            X_pert = X.copy()
            X_pert[:, d] = perturb_value[0, d]
            y_pred_pert = self.forward(X_pert)
            prob_pert = y_pred_pert[np.arange(N), pred_class_orig]
            importance[:, d] = prob_orig - prob_pert

        return importance

    def ablation_importance(self, X, y_true=None):
        N, D = X.shape
        y_pred_orig = self.forward(X)
        pred_class_orig = np.argmax(y_pred_orig, axis=1)
        prob_orig = y_pred_orig[np.arange(N), pred_class_orig]

        importance = np.zeros(D)
        for d in range(D):
            X_ablated = np.delete(X, d, axis=1)
            W_ablated = [np.delete(self.W[0], d, axis=0)] + self.W[1:]
            W_ablated[0] = W_ablated[0].copy()

            original_W0 = self.W[0]
            self.W[0] = W_ablated[0]

            y_pred_abl = self.forward(X_ablated)
            prob_abl = y_pred_abl[np.arange(N), pred_class_orig]
            importance[d] = np.mean(prob_orig - prob_abl)

            self.W[0] = original_W0

        return importance

    def explain_prediction(self, x):
        x = x.reshape(1, -1)
        probs = self.forward(x).flatten()
        pred_class = np.argmax(probs)
        saliency = self.saliency_map(x).flatten()
        perturbation = self.perturbation_importance(x).flatten()
        ablation = self.ablation_importance(x)

        return {
            'predicted_class': int(pred_class),
            'probabilities': probs.tolist(),
            'saliency': saliency.tolist(),
            'perturbation_importance': perturbation.tolist(),
            'ablation_importance': ablation.tolist(),
        }

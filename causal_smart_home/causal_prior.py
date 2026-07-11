"""轻量版 causal-relation 风格先验挖掘。

当外部 causal-relation/GCAD 产物没有直接提供时，本模块作为本地 fallback。
它先在事件张量上训练一个小预测器，再分别对每个输出通道的预测损失做反向
传播，用“输出损失对输入通道窗口的梯度大小”估计方向性影响强度。最终得到
的矩阵会被稀疏化、归一化，并保存成可审计的因果先验 JSON。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Sequence
import json
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class CausalPrior:
    """从正常行为数据中挖掘出的、可序列化保存的因果先验。"""

    matrix: list[list[float]]
    channel_to_key: list[str]
    lag: int
    sparse_threshold: float
    method: str = "gradient_causal_miner"
    meta: Optional[dict[str, Any]] = None

    @property
    def np_matrix(self) -> np.ndarray:
        """把 JSON/list 形式保存的因果矩阵转成 NumPy 数组，方便排序和计算。"""
        return np.asarray(self.matrix, dtype=np.float32)

    def top_edges(self, k: int = 20, min_weight: float | None = None, include_self: bool = False) -> list[dict[str, Any]]:
        """按权重从高到低列出非零有向边。

        每条边都保留 source/target 的通道键、矩阵下标、权重和 lag，后续
        GSS 重加权和 prompt 构建都依赖这些字段。
        """
        mat = self.np_matrix
        edges: list[tuple[float, int, int]] = []
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not include_self and i == j:
                    continue
                w = float(mat[i, j])
                if min_weight is not None and w < min_weight:
                    continue
                if w > 0:
                    edges.append((w, i, j))
        edges.sort(reverse=True, key=lambda x: x[0])
        return [
            {
                "source": self.channel_to_key[i],
                "target": self.channel_to_key[j],
                "source_index": i,
                "target_index": j,
                "weight": w,
                "lag": self.lag,
            }
            for w, i, j in edges[:k]
        ]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CausalPrior":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


if nn is not None:

    class _TinyMixer(nn.Module):
        """用于从事件张量挖掘先验的小型 MLP 预测器。"""

        def __init__(self, in_channels: int, lag: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels * lag, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, in_channels),
            )

        def forward(self, x):
            return self.net(x)

else:  # pragma: no cover
    _TinyMixer = None


class GradientCausalMiner:
    """基于梯度的轻量 GCAD/causal-relation 风格因果挖掘器。

    基本思路是：

    1. 在正常行为的多变量时间序列上训练预测器；
    2. 对每个输出通道单独计算预测损失；
    3. 将该损失反传到输入时间窗口；
    4. 对输入通道的绝对梯度在样本维度和 lag 维度上求平均；
    5. 通过矩阵与其转置的差保留方向不对称部分，形成稀疏有向先验。

    外部 causal relation 仓库仍然可以通过 ``causal_relation_adapter`` 接入。
    这里保留一个小实现，是为了让主流程在没有外部依赖时也能跑通并测试。
    """

    def __init__(
        self,
        lag: int = 4,
        epochs: int = 80,
        hidden: int = 64,
        learning_rate: float = 1e-3,
        sparse_threshold: float = 0.0,
        batch_size: int = 64,
        seed: int = 2024,
        device: str | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("torch is required for GradientCausalMiner; install project dependencies or run inside python environment")
        self.lag = lag
        self.epochs = epochs
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.sparse_threshold = sparse_threshold
        self.batch_size = batch_size
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 这个小 MLP 主要服务单元测试和轻量实验。高核 CPU 机器上，PyTorch
        # 默认线程数可能让小模型的调度开销超过实际计算开销；CPU 模式下适度
        # 限制线程数可以减少波动，同时不影响 CUDA 运行和外部 GCAD 项目。
        if self.device == "cpu":
            try:
                torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
            except Exception:
                pass
        self.model: Optional[_TinyMixer] = None
        self.train_loss_: list[float] = []

    def _windows(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """构造监督学习窗口：过去 ``lag`` 个时间步预测当前时间步。"""
        if x.ndim != 2:
            raise ValueError(f"expected [T, C] tensor, got shape {x.shape}")
        if len(x) <= self.lag:
            raise ValueError("tensor is shorter than lag")
        xs, ys = [], []
        for t in range(self.lag, len(x)):
            xs.append(x[t - self.lag : t])
            ys.append(x[t])
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def fit_predictor(self, x: np.ndarray) -> "GradientCausalMiner":
        """在源上下文正常行为张量上训练预测器。"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        xs, ys = self._windows(x)
        n, lag, channels = xs.shape
        self.model = _TinyMixer(channels, lag, hidden=self.hidden).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        x_tensor = torch.tensor(xs, dtype=torch.float32)
        y_tensor = torch.tensor(ys, dtype=torch.float32)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            losses = []
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = x_tensor[idx].to(self.device)
                yb = y_tensor[idx].to(self.device)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            self.train_loss_.append(float(np.mean(losses)))
        return self

    def discover(self, x: np.ndarray, sample_limit: int | None = None) -> np.ndarray:
        """通过输出通道损失的梯度反传来估计有向影响矩阵。"""
        if self.model is None:
            self.fit_predictor(x)
        assert self.model is not None
        self.model.eval()
        xs, ys = self._windows(x)
        if sample_limit is not None and len(xs) > sample_limit:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(xs), size=sample_limit, replace=False)
            xs, ys = xs[idx], ys[idx]
        xb = torch.tensor(xs, dtype=torch.float32, device=self.device, requires_grad=True)
        yb = torch.tensor(ys, dtype=torch.float32, device=self.device)
        pred = self.model(xb)
        channels = pred.shape[-1]
        causal = []
        for out_ch in range(channels):
            self.model.zero_grad(set_to_none=True)
            if xb.grad is not None:
                xb.grad.zero_()
            loss = torch.mean((pred[:, out_ch] - yb[:, out_ch]) ** 2)
            grad = torch.autograd.grad(loss, xb, retain_graph=True, create_graph=False)[0]
            # 梯度形状为 [samples, lag, input_ch]，这里压缩成每个输入通道
            # 对当前输出通道的平均影响强度。
            g = torch.mean(torch.abs(grad), dim=(0, 1))
            causal.append(g.detach().cpu().numpy())
        # 行表示输入/source channel，列表示被预测的 target/output channel。
        mat = np.stack(causal, axis=1).astype(np.float32)
        return self.sparsify(mat, self.sparse_threshold)

    @staticmethod
    def sparsify(matrix: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        # causal-relation 风格的对称差分：A->B 与 B->A 中更强的一侧会被保留，
        # 对称共同部分被削弱，从而突出方向性而不是单纯共现。
        diff = matrix - matrix.T
        out = np.maximum(diff, 0.0)
        diag = np.diag(matrix).copy()
        np.fill_diagonal(out, diag)
        max_positive = float(np.max(out)) if out.size else 0.0
        if max_positive > 0:
            out = out / max_positive
        if threshold > 0:
            out[out < threshold] = 0.0
        return out.astype(np.float32)

    def fit_prior(self, x: np.ndarray, channel_to_key: Sequence[str], sample_limit: int | None = None) -> CausalPrior:
        """完成训练、因果发现，并把矩阵封装成可保存的 ``CausalPrior``。"""
        self.fit_predictor(x)
        mat = self.discover(x, sample_limit=sample_limit)
        return CausalPrior(
            matrix=mat.tolist(),
            channel_to_key=list(channel_to_key),
            lag=self.lag,
            sparse_threshold=self.sparse_threshold,
            meta={"train_loss_last": self.train_loss_[-1] if self.train_loss_ else None, "epochs": self.epochs},
        )

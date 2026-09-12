"""
模型擂台：四层模型，同一个走向前协议下比成绩。

    L0  逻辑回归 + L1     ← 尺子，不是备胎
    L1  LightGBM
    L2  GRU 序列模型       (torch，没装就自动跳过)
    L3  L1+L2 概率加权     (仅当两者误差互补)

L0 为什么是尺子
---------------
如果 LightGBM 只比逻辑回归高 1 个百分点，说明特征已经榨干了，
继续加模型复杂度只会加过拟合风险，那时该回去挖特征而不是换模型。
没有这把尺子，就无法判断"提升"来自模型还是来自过拟合。

样本不平衡怎么处理
------------------
y_t0（真正的起涨点）的正样本率比 y_up 低一个量级——一段行情只有一个
起涨点，但 y_up 会连续亮很多天。直接训练模型会全预测 0。

这里用**按月分层的负样本下采样**：每个月内按固定比例抽负样本，
保持每月的正负比一致。这一手同时干掉两件事：不平衡，以及 2024-09
那个月的样本量碾压其他月份（设计文档 2.1）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("model")

NEG_PER_POS = 12       # 每个正样本配多少负样本
SEED = 7


def stratified_sample(df: pd.DataFrame, y: str,
                      neg_per_pos: int = NEG_PER_POS,
                      seed: int = SEED) -> pd.DataFrame:
    """按月分层下采样负样本。

    不这么做的话，2024-09 一个月的样本量就能压过其他所有月份，
    模型会把那个月的特征分布当成"起涨的样子"。
    """
    rng = np.random.default_rng(seed)
    df = df[np.isfinite(df[y])].copy()
    df["_m"] = df["date"].str[:7]
    out = []
    for m, g in df.groupby("_m"):
        pos = g[g[y] > 0]
        neg = g[g[y] <= 0]
        if len(pos) == 0:
            continue
        take = min(len(neg), len(pos) * neg_per_pos)
        if take > 0:
            idx = rng.choice(len(neg), size=take, replace=False)
            neg = neg.iloc[idx]
        out.append(pd.concat([pos, neg]))
    if not out:
        return df.iloc[:0]
    r = pd.concat(out, ignore_index=True).drop(columns=["_m"])
    return r.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _xy(df: pd.DataFrame, cols: list[str], y: str):
    X = df[cols].to_numpy(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, df[y].to_numpy(np.float32)


# ---------------------------------------------------------------
class L0Logistic:
    """逻辑回归 + L1。第四道特征筛也在这里：系数为 0 的特征被丢掉。"""
    name = "L0_logistic"

    def __init__(self, C: float = 0.05):
        self.C = C
        self.m = None
        self.cols: list[str] = []
        self.mu = None
        self.sd = None

    def fit(self, df, cols, y):
        # sklearn 1.8 起 penalty 被 l1_ratio 取代，但 liblinear 还认 penalty。
        # 警告每月刷一次会把走向前的日志淹掉，这里就地静音。
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning,
                                module="sklearn")
        warnings.filterwarnings("ignore", category=UserWarning,
                                module="sklearn")
        from sklearn.linear_model import LogisticRegression
        self.cols = cols
        X, yy = _xy(df, cols, y)
        # 标准化参数只能来自训练集，predict 时复用（防泄漏）
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.m = LogisticRegression(
            penalty="l1", C=self.C, solver="liblinear", max_iter=2000)
        self.m.fit((X - self.mu) / self.sd, yy)
        nz = int((self.m.coef_[0] != 0).sum())
        log.info("  L0 拟合完成，非零系数 %d/%d", nz, len(cols))
        return self

    def predict_proba(self, df):
        X, _ = _xy(df, self.cols, self.cols[0])
        return self.m.predict_proba((X - self.mu) / self.sd)[:, 1]

    def surviving(self) -> list[str]:
        """L1 之后系数非零的特征，这是第 4 道筛的结果。"""
        return [c for c, w in zip(self.cols, self.m.coef_[0]) if w != 0]


class L1Lgbm:
    name = "L1_lightgbm"

    def __init__(self, **kw):
        self.p = dict(
            objective="binary", n_estimators=400, learning_rate=0.04,
            num_leaves=31, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_lambda=5.0, random_state=SEED, n_jobs=4, verbose=-1)
        self.p.update(kw)
        self.m = None
        self.cols: list[str] = []

    def fit(self, df, cols, y):
        import lightgbm as lgb
        self.cols = cols
        X, yy = _xy(df, cols, y)
        self.m = lgb.LGBMClassifier(**self.p)
        self.m.fit(X, yy)
        return self

    def predict_proba(self, df):
        X, _ = _xy(df, self.cols, self.cols[0])
        return self.m.predict_proba(X)[:, 1]

    def importance(self) -> pd.Series:
        return pd.Series(self.m.feature_importances_,
                         index=self.cols).sort_values(ascending=False)


class L2Gru:
    """序列模型。吃 [N, 5, F] 的窗口，不做手工聚合。

    手工聚合（末值/均值/斜率）会抹掉**次序**：「缩量横盘 4 天后放量突破」
    和「首日放量后缩量回落」的三个统计量可以几乎一样，含义却相反。
    这是 L2 存在的唯一理由，不是因为神经网络时髦。
    """
    name = "L2_gru"

    def __init__(self, hidden: int = 32, epochs: int = 6, bs: int = 1024,
                 lr: float = 1e-3):
        self.hidden, self.epochs, self.bs, self.lr = hidden, epochs, bs, lr
        self.m = None
        self.base: list[str] = []
        self.mu = None
        self.sd = None

    @staticmethod
    def available() -> bool:
        """torch 装没装。用 find_spec 而不是 try-import：pyflakes 不认
        # noqa 注释，一个只为探测而存在的 import 会被它一直报成未使用。"""
        import importlib.util
        return importlib.util.find_spec("torch") is not None

    def _seq(self, df, base):
        """从 __last/__mean/__slope 三列还原出 5 步序列的近似。

        真正的逐日序列在 build 阶段没有单独落盘（会让训练表膨胀 5 倍），
        用三个统计量线性重建：slope 给出首末差，mean 给出中心。
        这是有损的，但保留了「先高后低」还是「先低后高」这个**次序方向**，
        而那正是聚合口径丢掉的东西。
        """
        n = len(df)
        out = np.zeros((n, 5, len(base)), dtype=np.float32)
        for j, b in enumerate(base):
            last = df.get(f"{b}__last", pd.Series(np.zeros(n))).to_numpy(np.float32)
            mean = df.get(f"{b}__mean", pd.Series(last)).to_numpy(np.float32)
            slope = df.get(f"{b}__slope", pd.Series(np.zeros(n))).to_numpy(np.float32)
            first = last - slope * 4.0
            for k in range(5):
                lin = first + (last - first) * (k / 4.0)
                out[:, k, j] = lin + (mean - lin) * 0.5
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, df, cols, y):
        import torch
        import torch.nn as nn
        self.base = sorted({c.split("__")[0] for c in cols})
        X = self._seq(df, self.base)
        self.mu = X.reshape(-1, X.shape[2]).mean(0)
        self.sd = X.reshape(-1, X.shape[2]).std(0) + 1e-9
        X = (X - self.mu) / self.sd
        yy = df[y].to_numpy(np.float32)

        torch.manual_seed(SEED)

        class Net(nn.Module):
            def __init__(self, f, h):
                super().__init__()
                self.gru = nn.GRU(f, h, num_layers=2, batch_first=True,
                                  dropout=0.1)
                self.head = nn.Sequential(nn.Linear(h, 16), nn.ReLU(),
                                          nn.Linear(16, 1))

            def forward(self, x):
                o, _ = self.gru(x)
                return self.head(o[:, -1]).squeeze(-1)

        self.m = Net(len(self.base), self.hidden)
        opt = torch.optim.Adam(self.m.parameters(), lr=self.lr)
        pos_w = torch.tensor(max((yy <= 0).sum() / max((yy > 0).sum(), 1), 1.0),
                             dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        xt = torch.from_numpy(X)
        yt = torch.from_numpy(yy)
        n = len(xt)
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            tot = 0.0
            self.m.train()
            for i in range(0, n, self.bs):
                idx = perm[i:i + self.bs]
                opt.zero_grad()
                out = self.m(xt[idx])
                loss = lossf(out, yt[idx])
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            log.info("  L2 epoch %d/%d loss %.4f", ep + 1, self.epochs, tot / n)
        return self

    def predict_proba(self, df):
        import torch
        X = (self._seq(df, self.base) - self.mu) / self.sd
        self.m.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 4096):
                out.append(torch.sigmoid(
                    self.m(torch.from_numpy(X[i:i + 4096]))).numpy())
        return np.concatenate(out) if out else np.zeros(len(X))


class L3Blend:
    name = "L3_blend"

    def __init__(self, a, b, w: float = 0.5):
        self.a, self.b, self.w = a, b, w

    def fit(self, df, cols, y):
        return self

    def predict_proba(self, df):
        return self.w * self.a.predict_proba(df) + \
            (1 - self.w) * self.b.predict_proba(df)

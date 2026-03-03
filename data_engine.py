"""
data_engine.py — PBOC货币政策数据引擎
======================================
职责:
  1. 通过 xbbg/blpapi 接口获取Bloomberg终端数据
  2. 在无终端连接时, 自动回退至基于Ornstein-Uhlenbeck过程的拟真数据生成器
  3. 平稳性检验与预处理 (ADF单位根检验, 差分, Z-score标准化)

Bloomberg Tickers (日频):
  - DR007        : CNFR007 Index   (银行间7天回购加权利率)
  - 1Y NCD       : CNAA1Y Index    (1年期同业存单收益率)
  - 1Y MLF       : CHLR12M Index   (1年期中期借贷便利利率)
  - RRR          : CHRRRP Index    (法定存款准备金率)
  - OMO Net Inj. : CNNIOMO Index   (公开市场净投放, 需滚动平滑)
"""

import warnings
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ============================================================================
# 常量定义
# ============================================================================

# Bloomberg Ticker 映射表
BBG_TICKERS: Dict[str, str] = {
    "DR007": "CNFR007 Index",
    "NCD_1Y": "CNAA1Y Index",
    "MLF_1Y": "CHLR12M Index",
    "RRR": "CHRRRP Index",
    "OMO_NET": "CNNIOMO Index",
}

# Ornstein-Uhlenbeck 拟真数据校准参数 (基于历史统计矩)
# 格式: (长期均值 mu, 均值回复速度 kappa, 波动率 sigma)
OU_PARAMS: Dict[str, Tuple[float, float, float]] = {
    "DR007": (2.10, 0.15, 0.35),       # 均值~2.10%, 较快均值回复
    "NCD_1Y": (2.50, 0.05, 0.20),      # 均值~2.50%, 中等回复速度
    "MLF_1Y": (2.75, 0.02, 0.05),      # 均值~2.75%, 政策利率变动缓慢
    "RRR": (8.50, 0.01, 0.15),         # 均值~8.50%, 极低频调整
    "OMO_NET": (0.0, 0.20, 500.0),     # 均值~0, 高频波动 (亿元)
}

# 各指标的平稳性首选处理方式
# "auto":   先ADF检验, 平稳→保持level, 否则自动差分 (利率类推荐)
# "diff":   强制一阶差分 (RRR等低频阶梯式调整)
# "zscore": 滚动Z标准化 (OMO等高波动数量型工具)
STATIONARITY_TREATMENT: Dict[str, str] = {
    "DR007": "auto",
    "NCD_1Y": "auto",
    "MLF_1Y": "auto",
    "RRR": "diff",
    "OMO_NET": "zscore",
}

# ADF检验显著性水平
ADF_SIGNIFICANCE: float = 0.05


# ============================================================================
# 数据获取类
# ============================================================================

class PBOCDataEngine:
    """
    PBOC货币政策工具数据引擎

    支持两种数据源:
      - Bloomberg终端 (通过xbbg)
      - 拟真数据 (Ornstein-Uhlenbeck过程, 自动回退)
    """

    def __init__(
        self,
        start_date: str = "2018-01-01",
        end_date: str = "2025-12-31",
        omo_rolling_window: int = 20,
        force_mock: bool = False,
    ):
        """
        Parameters
        ----------
        start_date : str
            数据起始日期 (YYYY-MM-DD)
        end_date : str
            数据截止日期 (YYYY-MM-DD)
        omo_rolling_window : int
            OMO净投放滚动求和窗口 (交易日), 默认20日
        force_mock : bool
            强制使用拟真数据 (跳过Bloomberg连接)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.omo_rolling_window = omo_rolling_window
        self.force_mock = force_mock
        self._raw_data: Optional[pd.DataFrame] = None
        self._processed_data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def fetch(self) -> pd.DataFrame:
        """获取原始数据 (Bloomberg优先, 自动回退拟真数据)"""
        if self._raw_data is not None:
            return self._raw_data.copy()

        if not self.force_mock:
            try:
                self._raw_data = self._fetch_bloomberg()
                logger.info("✓ Bloomberg数据获取成功")
                return self._raw_data.copy()
            except Exception as e:
                logger.warning(f"Bloomberg连接失败 ({e}), 回退至拟真数据")

        self._raw_data = self._generate_quasi_data()
        logger.info("✓ 拟真数据生成完毕 (Ornstein-Uhlenbeck)")
        return self._raw_data.copy()

    def get_processed(self) -> pd.DataFrame:
        """获取经平稳性处理后的建模用数据"""
        if self._processed_data is not None:
            return self._processed_data.copy()

        raw = self.fetch()
        self._processed_data = self._apply_stationarity(raw)
        return self._processed_data.copy()

    # ------------------------------------------------------------------
    # Bloomberg 数据接口
    # ------------------------------------------------------------------

    def _fetch_bloomberg(self) -> pd.DataFrame:
        """通过xbbg获取Bloomberg日频数据"""
        from xbbg import blp

        frames = {}
        for name, ticker in BBG_TICKERS.items():
            df = blp.bdh(
                tickers=ticker,
                flds="PX_LAST",
                start_date=self.start_date,
                end_date=self.end_date,
            )
            if df.empty:
                raise ValueError(f"Ticker {ticker} 返回空数据")
            # xbbg返回多级列索引, 取第一层
            series = df.iloc[:, 0]
            series.name = name
            frames[name] = series

        raw = pd.DataFrame(frames)
        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "date"

        # OMO净投放: 滚动求和平滑
        if "OMO_NET" in raw.columns:
            raw["OMO_NET"] = (
                raw["OMO_NET"]
                .rolling(window=self.omo_rolling_window, min_periods=1)
                .sum()
            )

        return raw.dropna(how="all").ffill()

    # ------------------------------------------------------------------
    # 拟真数据生成器 (Ornstein-Uhlenbeck)
    # ------------------------------------------------------------------

    def _generate_quasi_data(self, seed: int = 42) -> pd.DataFrame:
        """
        基于Ornstein-Uhlenbeck过程生成拟真数据

        dX_t = kappa * (mu - X_t) * dt + sigma * dW_t

        每个指标独立模拟, 参数从OU_PARAMS校准表读取.
        OMO_NET额外进行滚动求和平滑以模拟实际公告累计效应.
        MLF_1Y模拟阶梯式调整 (离散政策利率特征).
        """
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start=self.start_date, end=self.end_date)
        n = len(dates)
        dt = 1.0 / 252.0  # 日频时间步长

        data = {}
        for name, (mu, kappa, sigma) in OU_PARAMS.items():
            x = np.zeros(n)
            x[0] = mu + rng.normal(0, sigma * 0.1)

            for t in range(1, n):
                dx = kappa * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * rng.normal()
                x[t] = x[t - 1] + dx

            # MLF: 量化为5bp阶梯 (模拟政策利率离散调整)
            if name == "MLF_1Y":
                x = np.round(x * 200) / 200  # 5bp grid

            # RRR: 量化为25bp阶梯
            if name == "RRR":
                x = np.round(x * 40) / 40  # 25bp grid

            data[name] = x

        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"

        # OMO: 滚动求和平滑
        df["OMO_NET"] = (
            df["OMO_NET"]
            .rolling(window=self.omo_rolling_window, min_periods=1)
            .sum()
        )

        return df

    # ------------------------------------------------------------------
    # 平稳性检验与预处理
    # ------------------------------------------------------------------

    def _apply_stationarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对各指标执行自适应平稳性处理:
          - "auto":   ADF检验 → 平稳保level, 否则差分, 仍不平稳则二阶差分
          - "diff":   强制一阶差分
          - "zscore": 滚动Z标准化
          - "level":  强制保持原值 (不推荐, 仅供覆盖)

        每列的实际处理方式记录在 self.stationarity_log 中.
        """
        from statsmodels.tsa.stattools import adfuller

        processed = pd.DataFrame(index=df.index)
        self.stationarity_log: Dict[str, str] = {}

        for col in df.columns:
            treatment = STATIONARITY_TREATMENT.get(col, "auto")
            series = df[col].copy()

            if treatment == "diff":
                processed[col] = series.diff()
                self.stationarity_log[col] = "diff (强制一阶差分)"

            elif treatment == "zscore":
                roll_mean = series.rolling(60, min_periods=20).mean()
                roll_std = series.rolling(60, min_periods=20).std()
                roll_std = roll_std.replace(0, np.nan)
                processed[col] = (series - roll_mean) / roll_std
                self.stationarity_log[col] = "zscore (滚动Z标准化)"

            elif treatment == "level":
                processed[col] = series
                self.stationarity_log[col] = "level (强制保持原值)"

            else:  # auto — 自适应策略
                clean = series.dropna()
                # 第一步: 检验原序列
                try:
                    adf_p = adfuller(clean, autolag="AIC")[1]
                except Exception:
                    adf_p = 1.0

                if adf_p < ADF_SIGNIFICANCE:
                    processed[col] = series
                    self.stationarity_log[col] = f"level (ADF p={adf_p:.4f}, 原序列平稳)"
                    continue

                # 第二步: 一阶差分后检验
                diff1 = series.diff().dropna()
                try:
                    adf_p1 = adfuller(diff1, autolag="AIC")[1]
                except Exception:
                    adf_p1 = 1.0

                if adf_p1 < ADF_SIGNIFICANCE:
                    processed[col] = series.diff()
                    self.stationarity_log[col] = (
                        f"diff (原序列ADF p={adf_p:.4f} 非平稳 → "
                        f"一阶差分后 p={adf_p1:.4f}, 平稳)"
                    )
                    continue

                # 第三步: 二阶差分 (极端情况兜底)
                diff2 = series.diff().diff().dropna()
                try:
                    adf_p2 = adfuller(diff2, autolag="AIC")[1]
                except Exception:
                    adf_p2 = 1.0
                processed[col] = series.diff().diff()
                self.stationarity_log[col] = (
                    f"diff2 (一阶差分ADF p={adf_p1:.4f} 仍非平稳 → "
                    f"二阶差分 p={adf_p2:.4f})"
                )

            logger.info(f"  {col}: {self.stationarity_log[col]}")

        return processed.dropna()

    # ------------------------------------------------------------------
    # 单位根检验报告
    # ------------------------------------------------------------------

    @staticmethod
    def adf_test(series: pd.Series, name: str = "") -> Dict:
        """
        Augmented Dickey-Fuller 单位根检验

        Returns
        -------
        dict : 包含检验统计量, p值, 临界值, 结论
        """
        from statsmodels.tsa.stattools import adfuller

        clean = series.dropna()
        if len(clean) < 20:
            return {"name": name, "error": "样本量不足"}

        result = adfuller(clean, autolag="AIC")
        conclusion = "平稳 (拒绝H0)" if result[1] < 0.05 else "非平稳 (无法拒绝H0)"

        return {
            "指标": name,
            "ADF统计量": round(result[0], 4),
            "p值": round(result[1], 4),
            "滞后阶数": result[2],
            "1%临界值": round(result[4]["1%"], 4),
            "5%临界值": round(result[4]["5%"], 4),
            "10%临界值": round(result[4]["10%"], 4),
            "结论": conclusion,
        }

    def run_adf_battery(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """对所有指标执行ADF检验, 返回汇总表"""
        if df is None:
            df = self.fetch()
        results = [self.adf_test(df[col], name=col) for col in df.columns]
        return pd.DataFrame(results)


# ============================================================================
# IRS / CGB 基准利率数据 (用于Alpha验证回归)
# ============================================================================

class BenchmarkDataEngine:
    """
    获取基准利率数据: 1Y IRS (CNRB1Y Curncy), 10Y CGB
    同样支持拟真数据回退.
    """

    BBG_BENCHMARKS = {
        "IRS_1Y": "CNRB1Y Curncy",
        "CGB_10Y": "GCNY10YR Index",
    }

    OU_BENCHMARK_PARAMS = {
        "IRS_1Y": (2.30, 0.08, 0.25),
        "CGB_10Y": (2.85, 0.04, 0.15),
    }

    def __init__(
        self,
        start_date: str = "2018-01-01",
        end_date: str = "2025-12-31",
        force_mock: bool = False,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.force_mock = force_mock

    def fetch(self, seed: int = 99) -> pd.DataFrame:
        """获取基准利率 (Bloomberg优先, 自动回退)"""
        if not self.force_mock:
            try:
                return self._fetch_bloomberg()
            except Exception:
                pass

        return self._generate_quasi(seed)

    def _fetch_bloomberg(self) -> pd.DataFrame:
        from xbbg import blp

        frames = {}
        for name, ticker in self.BBG_BENCHMARKS.items():
            df = blp.bdh(ticker, "PX_LAST", self.start_date, self.end_date)
            if df.empty:
                raise ValueError(f"{ticker} 空数据")
            frames[name] = df.iloc[:, 0]

        result = pd.DataFrame(frames)
        result.index = pd.to_datetime(result.index)
        result.index.name = "date"
        return result.dropna(how="all").ffill()

    def _generate_quasi(self, seed: int) -> pd.DataFrame:
        """Ornstein-Uhlenbeck拟真基准利率"""
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(self.start_date, self.end_date)
        n = len(dates)
        dt = 1.0 / 252.0

        data = {}
        for name, (mu, kappa, sigma) in self.OU_BENCHMARK_PARAMS.items():
            x = np.zeros(n)
            x[0] = mu
            for t in range(1, n):
                dx = kappa * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * rng.normal()
                x[t] = x[t - 1] + dx
            data[name] = x

        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"
        return df

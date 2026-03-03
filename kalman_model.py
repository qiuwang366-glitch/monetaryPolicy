"""
kalman_model.py — 动态因子模型 (DFM) 与 Kalman滤波器
=====================================================
状态空间表示:
  观测方程:  Y_t = Λ F_t + ε_t,   ε_t ~ N(0, R)
  状态转移:  F_t = Φ F_{t-1} + η_t,  η_t ~ N(0, Q)

  - Y_t: k×1 可观测政策工具向量 (DR007, NCD, MLF, ΔRRR, Z(OMO))
  - F_t: 1×1 潜在货币政策因子 (PBOC流动性真实立场)
  - Λ:   k×1 因子载荷矩阵
  - Φ:   1×1 状态持续性系数 (AR(1))
  - R:   k×k 观测噪声协方差 (对角)
  - Q:   1×1 状态扰动方差

方法论参考: Hélène Rey et al., NBER w34626 — "The Ins & Outs of Chinese
Monetary Policy Transmission", 动态因子模型识别PBOC货币政策立场.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

warnings_module = __import__("warnings")
warnings_module.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ============================================================================
# 模型参数容器
# ============================================================================

@dataclass
class DFMResults:
    """动态因子模型估计结果"""
    latent_factor: pd.Series            # 滤波后的潜在因子 F_t
    smoothed_factor: pd.Series          # RTS平滑后的因子 (双向)
    factor_loadings: pd.Series          # Λ (因子载荷)
    transition_coeff: float             # Φ (AR(1)持续性)
    obs_noise_var: pd.Series            # diag(R)
    state_noise_var: float              # Q
    log_likelihood: float               # 对数似然值
    aic: float                          # AIC信息准则
    bic: float                          # BIC信息准则
    fitted_values: pd.DataFrame         # 模型拟合值
    residuals: pd.DataFrame             # 观测残差
    model_summary: str                  # 模型摘要文本


# ============================================================================
# DFM Kalman滤波器核心类
# ============================================================================

class PBOCDynamicFactorModel:
    """
    基于statsmodels状态空间框架的动态因子模型

    核心思路:
      从PBOC多维政策工具(利率、准备金率、公开市场操作)中
      提取单一潜在因子, 代表央行真实的流动性调控意图.

    使用方法:
      model = PBOCDynamicFactorModel(data)
      results = model.fit()
      factor = results.smoothed_factor
    """

    def __init__(
        self,
        data: pd.DataFrame,
        k_factors: int = 1,
        factor_order: int = 1,
        standardize: bool = True,
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            经平稳性处理后的观测数据, 列=指标, 行=日期
        k_factors : int
            潜在因子数量, 默认1 (单因子)
        factor_order : int
            因子VAR阶数 (AR(p)中的p), 默认1
        standardize : bool
            是否对输入数据标准化 (推荐, 便于载荷系数可比)
        """
        self.raw_data = data.copy()
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.standardize = standardize

        # 标准化参数 (反标准化用)
        self._means: Optional[pd.Series] = None
        self._stds: Optional[pd.Series] = None

        # 预处理
        self.data = self._prepare_data(data)
        self._model = None
        self._fit_result = None

    # ------------------------------------------------------------------
    # 数据预处理
    # ------------------------------------------------------------------

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗与标准化"""
        clean = df.dropna()

        if self.standardize:
            self._means = clean.mean()
            self._stds = clean.std()
            self._stds = self._stds.replace(0, 1.0)  # 防止除零
            clean = (clean - self._means) / self._stds

        return clean

    # ------------------------------------------------------------------
    # 模型估计
    # ------------------------------------------------------------------

    def fit(self, maxiter: int = 500, disp: bool = False) -> DFMResults:
        """
        通过最大似然估计 (MLE) 拟合DFM

        内部调用statsmodels.tsa.statespace.DynamicFactor,
        采用EM算法初始化 + L-BFGS-B优化器.

        Parameters
        ----------
        maxiter : int
            最大迭代次数
        disp : bool
            是否输出优化过程

        Returns
        -------
        DFMResults : 完整估计结果
        """
        self._model = DynamicFactor(
            endog=self.data,
            k_factors=self.k_factors,
            factor_order=self.factor_order,
        )

        # 尝试多种优化策略, 确保收敛
        try:
            self._fit_result = self._model.fit(
                maxiter=maxiter,
                disp=disp,
                method="lbfgs",
            )
        except Exception:
            logger.warning("L-BFGS-B未收敛, 切换至Powell优化器")
            self._fit_result = self._model.fit(
                maxiter=maxiter * 2,
                disp=disp,
                method="powell",
            )

        return self._extract_results()

    # ------------------------------------------------------------------
    # 结果提取
    # ------------------------------------------------------------------

    def _extract_results(self) -> DFMResults:
        """从statsmodels拟合对象中提取结构化结果"""
        res = self._fit_result
        variables = self.data.columns.tolist()

        # 因子载荷 Λ
        loading_params = [
            res.params.get(f"loading.f1.{var}", np.nan) for var in variables
        ]
        factor_loadings = pd.Series(loading_params, index=variables, name="Λ")

        # 状态持续性 Φ
        transition_coeff = res.params.get("L1.f1.f1", np.nan)

        # 观测噪声方差 diag(R)
        obs_vars = []
        for var in variables:
            key = f"sigma2.{var}"
            obs_vars.append(res.params.get(key, np.nan))
        obs_noise_var = pd.Series(obs_vars, index=variables, name="diag(R)")

        # 状态噪声方差 Q — 在DynamicFactor中归一化为1
        state_noise_var = 1.0

        # 滤波与平滑因子
        filtered = res.filtered_state[0]
        smoothed = res.smoothed_state[0]

        factor_filtered = pd.Series(
            filtered, index=self.data.index, name="潜在因子_滤波"
        )
        factor_smoothed = pd.Series(
            smoothed, index=self.data.index, name="潜在因子_平滑"
        )

        # 符号识别: 确保因子与DR007正相关 (紧缩=因子上升)
        corr_dr007 = np.corrcoef(
            factor_smoothed.values, self.data.iloc[:, 0].values
        )[0, 1]
        if corr_dr007 < 0:
            factor_filtered = -factor_filtered
            factor_smoothed = -factor_smoothed
            factor_loadings = -factor_loadings
            logger.info("符号翻转: 确保因子与DR007正相关 (紧缩→因子↑)")

        # 拟合值与残差
        fitted_values = pd.DataFrame(
            res.fittedvalues, index=self.data.index, columns=variables
        )
        residuals = self.data - fitted_values

        # 模型摘要
        try:
            summary_str = str(res.summary())
        except Exception:
            summary_str = "模型摘要生成失败"

        return DFMResults(
            latent_factor=factor_filtered,
            smoothed_factor=factor_smoothed,
            factor_loadings=factor_loadings,
            transition_coeff=transition_coeff,
            obs_noise_var=obs_noise_var,
            state_noise_var=state_noise_var,
            log_likelihood=res.llf,
            aic=res.aic,
            bic=res.bic,
            fitted_values=fitted_values,
            residuals=residuals,
            model_summary=summary_str,
        )


# ============================================================================
# Alpha验证: 因子对基准利率的预测能力
# ============================================================================

class AlphaValidator:
    """
    交易Alpha验证模块

    将潜在因子 F_t 回归至基准利率 (1Y IRS / 10Y CGB),
    分析残差分布以识别定价偏离 (Convexity交易机会).
    """

    def __init__(
        self,
        factor: pd.Series,
        benchmark: pd.DataFrame,
    ):
        """
        Parameters
        ----------
        factor : pd.Series
            潜在因子时间序列
        benchmark : pd.DataFrame
            基准利率数据 (IRS_1Y, CGB_10Y)
        """
        self.factor = factor
        self.benchmark = benchmark
        self._results: Dict[str, Dict] = {}

    def run_regression(self, target_col: str = "IRS_1Y") -> Dict:
        """
        OLS回归: target = α + β * F_t + ε_t

        Parameters
        ----------
        target_col : str
            回归目标列名

        Returns
        -------
        dict : 回归结果 (系数, R², 残差序列, 模型对象)
        """
        # 对齐时间索引
        combined = pd.DataFrame({
            "factor": self.factor,
            "target": self.benchmark[target_col],
        }).dropna()

        if len(combined) < 30:
            raise ValueError(f"有效样本量不足: {len(combined)}")

        X = sm.add_constant(combined["factor"])
        y = combined["target"]

        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})

        residuals = model.resid
        residual_zscore = (residuals - residuals.mean()) / residuals.std()

        result = {
            "target": target_col,
            "alpha": model.params.iloc[0],
            "beta": model.params.iloc[1],
            "t_stat_beta": model.tvalues.iloc[1],
            "p_value_beta": model.pvalues.iloc[1],
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "durbin_watson": sm.stats.stattools.durbin_watson(residuals),
            "residuals": residuals,
            "residual_zscore": residual_zscore,
            "fitted": model.fittedvalues,
            "model": model,
            "combined_data": combined,
        }

        self._results[target_col] = result
        return result

    def identify_divergence_regimes(
        self,
        target_col: str = "IRS_1Y",
        zscore_threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        识别因子与基准利率的显著偏离区间 (Convexity交易信号)

        Parameters
        ----------
        target_col : str
            目标基准利率
        zscore_threshold : float
            Z-score阈值, 超过视为显著偏离

        Returns
        -------
        pd.DataFrame : 偏离区间标注 (日期, 方向, Z-score)
        """
        if target_col not in self._results:
            self.run_regression(target_col)

        res = self._results[target_col]
        zscores = res["residual_zscore"]

        regimes = pd.DataFrame(index=zscores.index)
        regimes["残差Z值"] = zscores
        regimes["偏离方向"] = np.where(
            zscores > zscore_threshold,
            "基准偏高 (做空机会)",
            np.where(
                zscores < -zscore_threshold,
                "基准偏低 (做多机会)",
                "均衡区间",
            ),
        )
        regimes["信号强度"] = np.abs(zscores)

        return regimes

    def get_regression_summary(self, target_col: str = "IRS_1Y") -> str:
        """生成回归结果的格式化摘要"""
        if target_col not in self._results:
            self.run_regression(target_col)

        r = self._results[target_col]
        return (
            f"回归目标: {r['target']}\n"
            f"  α (截距):    {r['alpha']:.4f}\n"
            f"  β (因子系数): {r['beta']:.4f}  "
            f"[t={r['t_stat_beta']:.2f}, p={r['p_value_beta']:.4f}]\n"
            f"  R²:          {r['r_squared']:.4f}\n"
            f"  Adj R²:      {r['adj_r_squared']:.4f}\n"
            f"  DW统计量:    {r['durbin_watson']:.4f}\n"
        )


# ============================================================================
# 因子分析工具函数
# ============================================================================

def compute_factor_stats(factor: pd.Series) -> Dict:
    """计算潜在因子的描述性统计与体制特征"""
    return {
        "均值": factor.mean(),
        "标准差": factor.std(),
        "偏度": factor.skew(),
        "峰度": factor.kurtosis(),
        "最小值": factor.min(),
        "最大值": factor.max(),
        "当前值": factor.iloc[-1],
        "最近20日均值": factor.iloc[-20:].mean(),
        "百分位数_25%": factor.quantile(0.25),
        "百分位数_75%": factor.quantile(0.75),
    }


def classify_policy_regime(
    factor: pd.Series,
    tight_threshold: float = 0.5,
    loose_threshold: float = -0.5,
) -> pd.Series:
    """
    基于潜在因子划分货币政策体制

    Returns
    -------
    pd.Series : "紧缩" / "中性" / "宽松"
    """
    return pd.Series(
        np.where(
            factor > tight_threshold,
            "紧缩",
            np.where(factor < loose_threshold, "宽松", "中性"),
        ),
        index=factor.index,
        name="政策体制",
    )

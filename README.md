# 🏛️ PBOC 潜在货币政策因子提取框架

> **Dynamic Factor Model (DFM) + Kalman Filter — 从央行多维工具中萃取流动性真实立场**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14-4C72B0)](https://www.statsmodels.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly)](https://plotly.com)
[![Bloomberg](https://img.shields.io/badge/Bloomberg-xbbg-FF6600)](https://github.com/alpha-xone/xbbg)
[![NBER](https://img.shields.io/badge/NBER-w34626-8B0000)](https://www.nber.org/papers/w34626)

---

## 📐 方法论

基于 **Hélène Rey et al.** *"The Ins & Outs of Chinese Monetary Policy Transmission"* ([NBER w34626](https://www.nber.org/papers/w34626)) 的研究框架, 构建状态空间模型:

```
观测方程:  Y_t = Λ F_t + ε_t     ε_t ~ N(0, R)
状态转移:  F_t = Φ F_{t-1} + η_t   η_t ~ N(0, Q)
```

| 符号 | 维度 | 含义 |
|:---:|:---:|:---|
| **Y_t** | k × 1 | 可观测 PBOC 政策工具向量 |
| **F_t** | 1 × 1 | **潜在货币政策因子** (流动性真实立场) |
| **Λ** | k × 1 | 因子载荷 — 各工具对潜在因子的敏感度 |
| **Φ** | 1 × 1 | AR(1) 持续性系数 |
| **R** | k × k | 观测噪声协方差 (对角矩阵) |
| **Q** | 1 × 1 | 状态扰动方差 |

**核心思路:** 从 PBOC 5 个异质性政策工具中提取**单一潜在因子**, 刻画央行真实的流动性调控意图, 用于识别市场定价偏离 (Convexity 交易机会).

---

## 📊 数据源

### 可观测政策工具 (Y_t)

| 指标 | Bloomberg Ticker | 类型 | 平稳性处理 | 说明 |
|:---|:---|:---:|:---:|:---|
| **DR007** | `CNFR007 Index` | 价格型 | Level | 银行间 7 天回购加权利率 |
| **1Y NCD** | `CNAA1Y Index` | 价格型 | Level | 1 年期同业存单收益率 |
| **1Y MLF** | `CHLR12M Index` | 价格型 | Level | 1 年期中期借贷便利利率 |
| **RRR** | `CHRRRP Index` | 数量型 | Δ (一阶差分) | 法定存款准备金率 |
| **OMO 净投放** | `CNNIOMO Index` | 数量型 | Z-score (滚动 60 日) | 公开市场净投放 (20 日滚动累计) |

### Alpha 验证基准

| 指标 | Bloomberg Ticker | 用途 |
|:---|:---|:---|
| **1Y IRS** | `CNRB1Y Curncy` | 利率互换 — 回归目标 |
| **10Y CGB** | `GCNY10YR Index` | 国债收益率 — 回归目标 |

> **数据回退机制:** 无 Bloomberg 终端时, 自动回退至基于 **Ornstein-Uhlenbeck 过程** 的拟真数据生成器, 保证全流程可执行.

---

## 🏗️ 项目架构

```
monetaryPolicy/
│
├── data_engine.py               ← 数据引擎 (Bloomberg + OU 拟真回退)
│   ├── class PBOCDataEngine         获取 & 预处理 5 个政策工具
│   ├── class BenchmarkDataEngine    获取 IRS / CGB 基准利率
│   └── ADF 单位根检验电池
│
├── kalman_model.py              ← DFM 核心 (状态空间 + Kalman 滤波)
│   ├── class PBOCDynamicFactorModel   MLE 估计, 因子提取
│   ├── class AlphaValidator           OLS 回归, 偏离体制识别
│   ├── dataclass DFMResults           结构化估计结果
│   └── 工具函数                       因子统计, 政策体制分类
│
├── pboc_factor_analysis.ipynb   ← 主 Notebook (CIO 级交互报告)
│   └── 导入模块 → 执行流水线 → Plotly 可视化
│
└── README.md
```

### 设计原则

- **关注点分离:** 数据获取、模型估计、可视化三层解耦
- **Notebook 即报告:** `.ipynb` 仅含高层调用与图表, 零业务逻辑
- **双数据源:** Bloomberg 实时 → OU 拟真, 无缝切换

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install numpy pandas scipy statsmodels plotly jupyter
# Bloomberg 终端用户额外安装:
pip install xbbg blpapi
```

### 2. 运行 Notebook

```bash
jupyter notebook pboc_factor_analysis.ipynb
```

> 无 Bloomberg 终端? 无需任何配置 — 框架自动回退至 OU 拟真数据, 全流程正常执行.

### 3. 强制使用拟真数据

```python
engine = PBOCDataEngine(force_mock=True)
```

---

## 📈 Notebook 执行流水线

```
┌─────────────────────────────────────────────────────────┐
│  1. 数据获取                                              │
│     Bloomberg (xbbg) ──失败──→ OU 拟真数据                 │
│     ↓                                                    │
│  2. 平稳性检验 & 预处理                                    │
│     ADF 单位根 → Level / Δ / Z-score                      │
│     ↓                                                    │
│  3. DFM 估计 (Kalman 滤波)                                │
│     MLE (L-BFGS-B / Powell) → F_t, Λ, Φ                  │
│     ↓                                                    │
│  4. 因子分析                                              │
│     载荷解读 → 政策体制分类 → 偏离区间识别                    │
│     ↓                                                    │
│  5. Alpha 验证                                            │
│     OLS: IRS/CGB ~ F_t → 残差 Z-score → Convexity 信号    │
│     ↓                                                    │
│  6. 综合仪表盘 (Plotly 交互)                               │
│     CIO 决策视图: 因子 + 基准利率 + 体制 + 信号              │
└─────────────────────────────────────────────────────────┘
```

---

## 📉 可视化清单

Notebook 包含 **7 个交互式 Plotly 图表** (深色主题, 支持缩放/悬停):

| # | 图表 | 说明 |
|:---:|:---|:---|
| 1 | **原始序列总览** | 5 面板子图 — DR007, NCD, MLF, RRR, OMO |
| 2 | **因子载荷柱状图** | Λ 值排序 — 各工具对潜在因子的贡献 |
| 3 | **潜在因子时序** | F(t) + 紧缩/宽松区域着色 |
| 4 | **因子 vs 可观测工具** | F(t) vs DR007 & ΔRRR, 偏离区间竖线标注 |
| 5 | **政策体制饼图** | 紧缩 / 中性 / 宽松 分布占比 |
| 6 | **回归散点 + OLS 拟合线** | F(t) → IRS_1Y / CGB_10Y, R² 标注 |
| 7 | **综合仪表盘** | 三面板 CIO 视图: 因子+IRS / 残差Z值 / 体制时间线 |

---

## 🔬 核心 API 参考

### `PBOCDataEngine`

```python
engine = PBOCDataEngine(
    start_date="2018-01-01",
    end_date="2025-12-31",
    omo_rolling_window=20,   # OMO 滚动求和窗口 (交易日)
    force_mock=False,        # True = 跳过 Bloomberg
)

raw_data = engine.fetch()              # 原始数据 (自动回退)
processed = engine.get_processed()     # 平稳性处理后
adf_table = engine.run_adf_battery()   # ADF 检验汇总
```

### `PBOCDynamicFactorModel`

```python
dfm = PBOCDynamicFactorModel(
    data=processed,
    k_factors=1,       # 潜在因子数
    factor_order=1,    # AR(p) 阶数
    standardize=True,  # 输入标准化
)

results = dfm.fit(maxiter=500)

results.smoothed_factor      # pd.Series — 平滑因子 F(t)
results.factor_loadings      # pd.Series — 载荷 Λ
results.transition_coeff     # float     — 持续性 Φ
results.aic / results.bic    # 信息准则
```

### `AlphaValidator`

```python
validator = AlphaValidator(factor=results.smoothed_factor, benchmark=benchmark_data)

reg = validator.run_regression("IRS_1Y")       # OLS + HAC
regimes = validator.identify_divergence_regimes("IRS_1Y", zscore_threshold=1.5)
print(validator.get_regression_summary("IRS_1Y"))
```

---

## ⚙️ Ornstein-Uhlenbeck 拟真数据校准

拟真数据基于独立 OU 过程 `dX = κ(μ − X)dt + σdW`, 参数从历史统计矩校准:

| 指标 | μ (长期均值) | κ (回复速度) | σ (波动率) | 特殊处理 |
|:---|:---:|:---:|:---:|:---|
| DR007 | 2.10% | 0.15 | 0.35 | — |
| 1Y NCD | 2.50% | 0.05 | 0.20 | — |
| 1Y MLF | 2.75% | 0.02 | 0.05 | 5bp 阶梯量化 |
| RRR | 8.50% | 0.01 | 0.15 | 25bp 阶梯量化 |
| OMO 净投放 | 0 | 0.20 | 500 亿 | 20 日滚动累计 |

---

## 📚 参考文献

- Rey, H., Jiang, Z., & Richmond, R. (2024). *The Ins & Outs of Chinese Monetary Policy Transmission*. NBER Working Paper No. 34626. [[Link]](https://www.nber.org/papers/w34626)
- Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.
- `statsmodels.tsa.statespace.DynamicFactor` [[Documentation]](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.dynamic_factor.DynamicFactor.html)

---

## ⚠️ 风险提示

- OU 拟真数据**仅用于框架验证**, 生产部署须接入 Bloomberg 实时数据
- 单因子模型可能无法捕捉结构性转变 (LPR 改革、利率走廊完善等)
- 残差信号需结合宏观基本面、市场微观结构综合研判
- 本框架不构成任何投资建议

---

<p align="center"><sub>Built for institutional CIO review — all narratives in professional financial Chinese</sub></p>

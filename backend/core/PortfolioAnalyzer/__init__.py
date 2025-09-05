# 初始化PortfolioAnalyzer模块

# 导入主要类
from .diagnostics import PortfolioAnalyzer
from .optimizer import PortfolioOptimizer
from .optimizer_factors import FactorOptimizer
from .report import PortfolioReport

# 导入工具函数
from .utils import (
    preprocess_returns,
    normalize_weights,
    get_time_window_data,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_sortino_ratio,
    calculate_omega_ratio,
    prepare_chart_data,
    prepare_pie_chart_data,
    prepare_heatmap_data,
    calculate_risk_parity_weights,
    calculate_conditional_value_at_risk,
    calculate_liquidity_score,
    validate_returns_data,
    validate_weights
)

# 导入兼容性函数
from .diagnostics import get_portfolio_metrics
from .optimizer import optimize_portfolio
from .optimizer_factors import optimize_with_factors
from .report import generate_portfolio_report

# 定义__all__变量，指定公共API
__all__ = [
    # 类
    'PortfolioAnalyzer',
    'PortfolioOptimizer',
    'FactorOptimizer',
    'PortfolioReport',
    
    # 工具函数
    'preprocess_returns',
    'normalize_weights',
    'get_time_window_data',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    'calculate_sortino_ratio',
    'calculate_omega_ratio',
    'prepare_chart_data',
    'prepare_pie_chart_data',
    'prepare_heatmap_data',
    'calculate_risk_parity_weights',
    'calculate_conditional_value_at_risk',
    'calculate_liquidity_score',
    'validate_returns_data',
    'validate_weights',
    
    # 兼容性函数
    'get_portfolio_metrics',
    'optimize_portfolio',
    'optimize_with_factors',
    'generate_portfolio_report'
]

# 模块版本信息
__version__ = '1.0.0'

# 模块描述
__description__ = '投资组合分析与优化模块，提供完整的组合诊断、优化和报告功能'
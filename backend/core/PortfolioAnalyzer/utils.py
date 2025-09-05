import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

# 数据预处理工具
def preprocess_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理收益率数据，处理缺失值和异常值
    
    :param returns_df: 原始收益率数据
    :return: 预处理后的收益率数据
    """
    # 复制数据以避免修改原始数据
    df = returns_df.copy()
    
    # 处理缺失值 - 使用前向填充和后向填充
    df = df.ffill().bfill()
    
    # 处理异常值 - 限制在±3倍标准差范围内
    for col in df.columns:
        if df[col].std() > 0:
            # 计算3倍标准差
            std_3 = 3 * df[col].std()
            mean = df[col].mean()
            
            # 限制异常值
            df[col] = np.clip(df[col], mean - std_3, mean + std_3)
    
    return df

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    归一化权重，确保所有权重之和为1
    
    :param weights: 原始权重字典
    :return: 归一化后的权重字典
    """
    total_weight = sum(weights.values())
    
    if total_weight == 0:
        # 如果所有权重都为0，返回等权重
        n = len(weights)
        return {code: 1/n for code in weights}
    
    return {code: weight/total_weight for code, weight in weights.items()}

def get_time_window_data(returns_df: pd.DataFrame, window_type: str = "1y") -> pd.DataFrame:
    """
    获取指定时间窗口的数据
    
    :param returns_df: 完整的收益率数据
    :param window_type: 时间窗口类型，可选值："1m"(1个月), "3m"(3个月), "6m"(6个月), "1y"(1年), "3y"(3年), "all"(全部)
    :return: 指定时间窗口的数据
    """
    if window_type == "all":
        return returns_df
    
    # 确保索引是日期类型
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        returns_df.index = pd.to_datetime(returns_df.index)
    
    # 获取当前日期的最后一天
    end_date = returns_df.index.max()
    
    # 根据窗口类型计算开始日期
    if window_type == "1m":
        start_date = end_date - timedelta(days=30)
    elif window_type == "3m":
        start_date = end_date - timedelta(days=90)
    elif window_type == "6m":
        start_date = end_date - timedelta(days=180)
    elif window_type == "1y":
        start_date = end_date - timedelta(days=365)
    elif window_type == "3y":
        start_date = end_date - timedelta(days=365*3)
    else:
        # 默认使用1年的数据
        start_date = end_date - timedelta(days=365)
    
    # 筛选时间窗口内的数据
    mask = (returns_df.index >= start_date) & (returns_df.index <= end_date)
    return returns_df.loc[mask]

# 指标计算工具
def calculate_max_drawdown(nav: pd.Series) -> float:
    """
    计算最大回撤
    
    :param nav: 累计净值序列
    :return: 最大回撤值
    """
    if len(nav) == 0:
        return 0
    
    roll_max = nav.cummax()
    drawdown = (nav - roll_max) / roll_max
    return drawdown.min()

def calculate_calmar_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    计算卡尔玛比率（年化收益率除以最大回撤）
    
    :param returns: 收益率序列
    :param risk_free_rate: 无风险收益率
    :return: 卡尔玛比率
    """
    # 计算年化收益率
    annual_return = returns.mean() * 252
    
    # 计算累计净值
    nav = (1 + returns).cumprod()
    
    # 计算最大回撤
    max_drawdown = calculate_max_drawdown(nav)
    
    # 计算卡尔玛比率
    if max_drawdown >= 0:  # 如果没有回撤
        return float('inf')
    
    return (annual_return - risk_free_rate) / abs(max_drawdown)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    计算索提诺比率（使用下行波动率）
    
    :param returns: 收益率序列
    :param risk_free_rate: 无风险收益率
    :return: 索提诺比率
    """
    # 计算年化收益率
    annual_return = returns.mean() * 252
    
    # 计算下行波动率
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        downside_volatility = 0
    else:
        downside_volatility = np.sqrt((downside_returns ** 2).mean() * 252)
    
    # 计算索提诺比率
    if downside_volatility == 0:
        return float('inf') if annual_return > risk_free_rate else -float('inf')
    
    return (annual_return - risk_free_rate) / downside_volatility

def calculate_omega_ratio(returns: pd.Series, risk_free_rate: float = 0.02, threshold: float = None) -> float:
    """
    计算欧米茄比率
    
    :param returns: 收益率序列
    :param risk_free_rate: 无风险收益率
    :param threshold: 阈值，默认为无风险收益率
    :return: 欧米茄比率
    """
    # 如果没有提供阈值，使用无风险收益率的日度值
    if threshold is None:
        threshold = risk_free_rate / 252
    
    # 计算超额收益
    excess_returns = returns - threshold
    
    # 计算正超额收益和负超额收益
    positive_excess = excess_returns[excess_returns > 0].sum()
    negative_excess = abs(excess_returns[excess_returns < 0].sum())
    
    # 计算欧米茄比率
    if negative_excess == 0:
        return float('inf') if positive_excess > 0 else 0
    
    return positive_excess / negative_excess

# 数据可视化工具
def prepare_chart_data(returns_df: pd.DataFrame, weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    准备用于图表显示的数据
    
    :param returns_df: 收益率数据
    :param weights: 权重字典（可选）
    :return: 图表数据字典
    """
    # 确保索引是日期类型
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        returns_df.index = pd.to_datetime(returns_df.index)
    
    chart_data = {
        'dates': returns_df.index.strftime('%Y-%m-%d').tolist(),
        'series': []
    }
    
    # 如果提供了权重，计算组合收益率
    if weights:
        # 确保所有基金都在权重中
        for code in returns_df.columns:
            if code not in weights:
                weights[code] = 0
        
        # 计算组合收益率
        weights_array = np.array([weights[code] for code in returns_df.columns])
        portfolio_returns = returns_df.dot(weights_array)
        
        # 计算累计净值
        portfolio_nav = (1 + portfolio_returns).cumprod()
        
        # 添加组合数据到图表
        chart_data['series'].append({
            'name': '优化组合',
            'data': portfolio_nav.tolist(),
            'type': 'line',
            'color': '#1890ff'
        })
    
    # 添加各个基金的数据
    # 限制最多显示5个基金，避免图表过于复杂
    display_codes = returns_df.columns[:5]
    
    # 定义颜色列表
    colors = ['#ff7875', '#52c41a', '#faad14', '#722ed1', '#13c2c2']
    
    for i, code in enumerate(display_codes):
        # 计算基金的累计净值
        fund_nav = (1 + returns_df[code]).cumprod()
        
        chart_data['series'].append({
            'name': code,
            'data': fund_nav.tolist(),
            'type': 'line',
            'color': colors[i % len(colors)],
            'lineStyle': {
                'type': 'dashed'  # 非组合数据使用虚线
            },
            'showSymbol': False  # 不显示数据点
        })
    
    return chart_data

def prepare_pie_chart_data(weights: Dict[str, float], max_items: int = 5) -> Dict[str, Any]:
    """
    准备饼图数据
    
    :param weights: 权重字典
    :param max_items: 最多显示的项目数
    :return: 饼图数据字典
    """
    # 排序权重
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    # 处理超过max_items的情况
    if len(sorted_weights) > max_items:
        # 取前max_items-1个项目
        top_items = sorted_weights[:max_items-1]
        # 计算其他项目的总权重
        other_weight = sum([w for _, w in sorted_weights[max_items-1:]])
        # 添加"其他"项目
        top_items.append(('其他', other_weight))
        sorted_weights = top_items
    
    # 定义颜色列表
    colors = ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1', '#13c2c2', '#eb2f96']
    
    # 准备饼图数据
    pie_data = {
        'data': [],
        'colors': colors[:len(sorted_weights)]
    }
    
    for code, weight in sorted_weights:
        # 只显示权重大于0.1%的项目
        if weight > 0.001:
            pie_data['data'].append({
                'name': code,
                'value': round(weight * 100, 2)  # 转换为百分比
            })
    
    return pie_data

def prepare_heatmap_data(cov_matrix: pd.DataFrame) -> Dict[str, Any]:
    """
    准备热力图数据
    
    :param cov_matrix: 协方差矩阵或相关性矩阵
    :return: 热力图数据字典
    """
    # 限制显示的基金数量
    max_funds = 10
    funds_to_display = cov_matrix.columns[:max_funds]
    
    # 准备热力图数据
    heatmap_data = {
        'funds': funds_to_display.tolist(),
        'values': [],
        'min_value': float('inf'),
        'max_value': -float('inf')
    }
    
    # 提取数据并计算极值
    for i in range(len(funds_to_display)):
        row = []
        for j in range(len(funds_to_display)):
            value = cov_matrix.iloc[i, j]
            row.append(round(value, 4))
            
            # 更新极值
            if value < heatmap_data['min_value']:
                heatmap_data['min_value'] = value
            if value > heatmap_data['max_value']:
                heatmap_data['max_value'] = value
        heatmap_data['values'].append(row)
    
    return heatmap_data

# 风险评估工具
def calculate_risk_parity_weights(cov_matrix: pd.DataFrame) -> np.ndarray:
    """
    计算风险平价权重
    
    :param cov_matrix: 协方差矩阵
    :return: 风险平价权重数组
    """
    n = len(cov_matrix)
    
    # 初始权重
    weights = np.array([1 / n] * n)
    
    # 迭代求解风险平价权重
    for _ in range(100):  # 最多迭代100次
        # 计算边际风险贡献
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_risk_contribution = np.dot(cov_matrix, weights) / portfolio_vol
        total_risk_contribution = weights * marginal_risk_contribution
        
        # 计算目标风险贡献（相等）
        target_risk_contribution = np.array([np.mean(total_risk_contribution)] * n)
        
        # 更新权重
        weights = weights * (target_risk_contribution / total_risk_contribution)
        
        # 归一化权重
        weights = weights / sum(weights)
        
        # 检查收敛条件
        if np.max(np.abs(total_risk_contribution - target_risk_contribution)) < 1e-6:
            break
    
    return weights

def calculate_conditional_value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    计算条件在险价值(CVaR)
    
    :param returns: 收益率数组
    :param alpha: 显著性水平
    :return: CVaR值
    """
    # 计算VaR
    var = np.percentile(returns, alpha * 100)
    
    # 计算CVaR
    cvar = returns[returns <= var].mean()
    
    return cvar

def calculate_liquidity_score(fund_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    计算基金的流动性得分
    
    :param fund_data: 基金数据字典，包含规模、成交量等信息
    :return: 流动性得分字典
    """
    liquidity_scores = {}
    
    # 计算每个基金的流动性得分
    for fund_code, data in fund_data.items():
        # 提取相关数据
        size = data.get('size', 1)  # 基金规模，默认1亿
        turnover = data.get('turnover', 0.5)  # 换手率，默认0.5
        
        # 计算流动性得分（规模越大、换手率越高，流动性越好）
        # 这里使用简化的计算方法
        liquidity_score = (np.log(size + 1) * 0.5) + (turnover * 0.5)
        
        liquidity_scores[fund_code] = liquidity_score
    
    # 归一化流动性得分
    max_score = max(liquidity_scores.values())
    if max_score > 0:
        for fund_code in liquidity_scores:
            liquidity_scores[fund_code] = liquidity_scores[fund_code] / max_score
    
    return liquidity_scores

# 数据验证工具
def validate_returns_data(returns_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    验证收益率数据的有效性
    
    :param returns_df: 收益率数据
    :return: (是否有效, 错误信息)
    """
    # 检查是否为空
    if returns_df.empty:
        return False, "收益率数据为空"
    
    # 检查索引是否为日期类型
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        try:
            returns_df.index = pd.to_datetime(returns_df.index)
        except:
            return False, "索引不是有效的日期格式"
    
    # 检查是否有足够的数据点
    min_data_points = 60  # 至少需要60个数据点（约3个月）
    if len(returns_df) < min_data_points:
        return False, f"数据点不足，至少需要{min_data_points}个数据点"
    
    # 检查是否有NaN值
    if returns_df.isnull().values.any():
        # 检查NaN值的比例
        nan_ratio = returns_df.isnull().sum().sum() / (returns_df.shape[0] * returns_df.shape[1])
        if nan_ratio > 0.1:  # 如果NaN值超过10%
            return False, f"数据中NaN值比例过高({nan_ratio:.1%})"
    
    # 检查收益率是否在合理范围内（-20%到20%之间）
    if (returns_df < -0.2).any().any() or (returns_df > 0.2).any().any():
        return False, "数据中存在异常的收益率值（超出±20%）"
    
    return True, "数据验证通过"

def validate_weights(weights: Dict[str, float]) -> Tuple[bool, str]:
    """
    验证权重数据的有效性
    
    :param weights: 权重字典
    :return: (是否有效, 错误信息)
    """
    # 检查是否为空
    if not weights:
        return False, "权重数据为空"
    
    # 检查权重是否为数字
    for code, weight in weights.items():
        try:
            float(weight)
        except:
            return False, f"基金{code}的权重不是有效的数字"
    
    # 检查权重是否为非负数
    for code, weight in weights.items():
        if weight < 0:
            return False, f"基金{code}的权重为负数"
    
    # 检查权重总和是否在合理范围内（0.95到1.05之间）
    total_weight = sum(weights.values())
    if total_weight < 0.95 or total_weight > 1.05:
        return False, f"权重总和({total_weight:.4f})不在合理范围内（0.95-1.05）"
    
    return True, "权重验证通过"
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any

class PortfolioOptimizer:
    """
    投资组合优化模块
    基础优化：均值-方差、夏普最大化、最小波动率
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        :param returns: DataFrame, index=日期, columns=基金代码, values=日收益率
        :param risk_free_rate: 无风险收益率，默认为2%
        """
        self.returns = returns
        self.fund_codes = list(returns.columns)
        self.num_assets = len(self.fund_codes)
        self.risk_free_rate = risk_free_rate
        
        # 计算必要的统计数据
        self.mean_returns = returns.mean() * 252  # 年化收益率
        self.cov_matrix = returns.cov() * 252     # 年化协方差矩阵
    
    def _portfolio_return(self, weights: np.ndarray) -> float:
        """计算投资组合的预期收益率"""
        return np.dot(weights, self.mean_returns)
    
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """计算投资组合的波动率"""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def _sharpe_ratio(self, weights: np.ndarray) -> float:
        """计算夏普比率"""
        port_return = self._portfolio_return(weights)
        port_vol = self._portfolio_volatility(weights)
        return (port_return - self.risk_free_rate) / port_vol
    
    def _max_sharpe_objective(self, weights: np.ndarray) -> float:
        """最大化夏普比率的目标函数（用于minimize优化）"""
        # 因为scipy的minimize只能求最小值，所以返回负的夏普比率
        return -self._sharpe_ratio(weights)
    
    def _min_volatility_objective(self, weights: np.ndarray) -> float:
        """最小化波动率的目标函数"""
        return self._portfolio_volatility(weights)
    
    def _constraints(self) -> List[Dict[str, Any]]:
        """定义优化问题的约束条件"""
        # 权重总和为1
        return [{
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1
        }]
    
    def _bounds(self) -> List[Tuple[float, float]]:
        """定义权重的边界条件"""
        # 默认每个资产的权重在0到1之间
        return [(0, 1) for _ in range(self.num_assets)]
    
    def optimize(self, target: str = "sharpe", **kwargs) -> Dict[str, Any]:
        """
        执行投资组合优化
        
        :param target: 优化目标，可选值：'sharpe'(最大化夏普比率), 'volatility'(最小化波动率), 'return'(最大化收益率)
        :param kwargs: 额外参数，如target_return(当优化目标为'return'时)
        :return: 优化结果字典
        """
        # 初始权重 - 平均分配
        initial_weights = np.array([1 / self.num_assets] * self.num_assets)
        
        # 约束条件
        constraints = self._constraints()
        
        # 边界条件
        bounds = self._bounds()
        
        # 根据目标选择不同的优化问题
        if target == "sharpe":
            # 最大化夏普比率
            result = minimize(
                self._max_sharpe_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': False}
            )
        elif target == "volatility":
            # 最小化波动率
            result = minimize(
                self._min_volatility_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': False}
            )
        elif target == "return":
            # 在给定波动率限制下最大化收益率
            target_volatility = kwargs.get('target_volatility', None)
            if target_volatility is None:
                raise ValueError("当优化目标为'return'时，必须提供'target_volatility'参数")
            
            # 添加波动率约束
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self._portfolio_volatility(x) - target_volatility
            })
            
            # 目标函数：最大化收益率（注意minimize求的是最小值，所以返回负的收益率）
            result = minimize(
                lambda x: -self._portfolio_return(x),
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': False}
            )
        else:
            raise ValueError("不支持的优化目标。可选值：'sharpe', 'volatility', 'return'")
        
        # 检查优化是否成功
        if not result.success:
            raise RuntimeError(f"优化失败: {result.message}")
        
        # 获取优化后的权重
        optimized_weights = result.x
        
        # 构建优化结果
        return {
            'weights': {
                code: weight for code, weight in zip(self.fund_codes, optimized_weights)
            },
            'return': self._portfolio_return(optimized_weights),
            'volatility': self._portfolio_volatility(optimized_weights),
            'sharpe': self._sharpe_ratio(optimized_weights),
            'optimization_method': target,
            'success': result.success
        }
    
    def efficient_frontier(self, n_points: int = 20) -> Dict[str, List[float]]:
        """
        生成有效前沿
        
        :param n_points: 有效前沿上的点数量
        :return: 包含收益率、波动率和权重的字典
        """
        # 找到最小波动率组合
        min_vol_result = self.optimize(target="volatility")
        min_vol = min_vol_result['volatility']
        min_vol_return = min_vol_result['return']
        
        # 找到最大夏普比率组合
        max_sharpe_result = self.optimize(target="sharpe")
        max_sharpe_vol = max_sharpe_result['volatility']
        max_sharpe_return = max_sharpe_result['return']
        
        # 找到最大收益率组合（不受限制）
        # 临时移除边界条件
        original_bounds = self._bounds()
        self._bounds = lambda: [(0, 1) for _ in range(self.num_assets)]  # 保持原值，这里只是为了演示
        
        # 找到最大可能的收益率（通过设置一个高目标波动率）
        max_return_vol = min_vol * 2  # 设置一个足够大的目标波动率
        try:
            max_return_result = self.optimize(target="return", target_volatility=max_return_vol)
            max_return = max_return_result['return']
        except Exception:
            # 如果失败，使用最大夏普点的收益率加上一定的缓冲
            max_return = max_sharpe_return * 1.2
        
        # 生成有效前沿上的点
        returns = []
        volatilities = []
        weights_list = []
        
        # 确保我们生成的点覆盖从最小波动率到最大可能收益率的范围
        # 但避免出现不现实的高收益率
        target_returns = np.linspace(min_vol_return, max_return, n_points)
        
        for target_return in target_returns:
            # 创建约束条件：收益率等于目标收益率
            constraints = self._constraints()
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self._portfolio_return(x) - target_return
            })
            
            # 目标函数：最小化波动率
            result = minimize(
                self._min_volatility_objective,
                np.array([1 / self.num_assets] * self.num_assets),
                method='SLSQP',
                bounds=self._bounds(),
                constraints=constraints,
                options={'disp': False}
            )
            
            if result.success:
                returns.append(target_return)
                volatilities.append(self._portfolio_volatility(result.x))
                weights_list.append(result.x)
        
        # 返回有效前沿数据
        return {
            'returns': returns,
            'volatilities': volatilities,
            'weights': weights_list,
            'min_volatility_portfolio': {
                'return': min_vol_return,
                'volatility': min_vol,
                'weights': min_vol_result['weights']
            },
            'max_sharpe_portfolio': {
                'return': max_sharpe_return,
                'volatility': max_sharpe_vol,
                'sharpe': max_sharpe_result['sharpe'],
                'weights': max_sharpe_result['weights']
            }
        }
    
    def get_allocation_recommendations(self, current_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        根据当前权重提供配置建议
        
        :param current_weights: 当前持仓权重
        :return: 包含不同优化策略建议的字典
        """
        # 计算各种优化策略的结果
        recommendations = {
            'max_sharpe': self.optimize(target="sharpe"),
            'min_volatility': self.optimize(target="volatility")
        }
        
        # 添加相对于当前权重的变化
        for strategy, result in recommendations.items():
            changes = {}
            for code in self.fund_codes:
                current_weight = current_weights.get(code, 0)
                recommended_weight = result['weights'].get(code, 0)
                changes[code] = recommended_weight - current_weight
            
            recommendations[strategy]['changes'] = changes
        
        # 添加当前组合的表现
        current_weights_array = np.array([current_weights.get(code, 0) for code in self.fund_codes])
        recommendations['current'] = {
            'weights': current_weights,
            'return': self._portfolio_return(current_weights_array),
            'volatility': self._portfolio_volatility(current_weights_array),
            'sharpe': self._sharpe_ratio(current_weights_array)
        }
        
        return recommendations
    
    def optimize_for_risk_tolerance(self, risk_tolerance: str = "balanced") -> Dict[str, Any]:
        """
        根据风险承受能力推荐投资组合
        
        :param risk_tolerance: 风险承受能力，可选值：'conservative'(保守), 'balanced'(平衡), 'aggressive'(进取)
        :return: 推荐的投资组合配置
        """
        # 生成有效前沿
        ef = self.efficient_frontier(n_points=50)
        
        # 根据风险承受能力选择不同的点
        if risk_tolerance == "conservative":
            # 选择波动率最小的组合
            portfolio = ef['min_volatility_portfolio']
        elif risk_tolerance == "aggressive":
            # 选择收益率最高的组合（在有效前沿上）
            max_return_idx = np.argmax(ef['returns'])
            portfolio = {
                'return': ef['returns'][max_return_idx],
                'volatility': ef['volatilities'][max_return_idx],
                'weights': {code: weight for code, weight in zip(self.fund_codes, ef['weights'][max_return_idx])}
            }
        else:  # balanced
            # 选择最大夏普比率的组合
            portfolio = ef['max_sharpe_portfolio']
        
        # 计算夏普比率
        portfolio['sharpe'] = (portfolio['return'] - self.risk_free_rate) / portfolio['volatility']
        portfolio['risk_tolerance'] = risk_tolerance
        
        return portfolio
# 兼容性函数，用于服务调用
def optimize_portfolio(returns_df: pd.DataFrame, current_weights: dict = None, optimization_target: str = "sharpe", **kwargs):
    """
    用于服务调用的投资组合优化函数
    
    :param returns_df: 收益率数据
    :param current_weights: 当前权重（可选）
    :param optimization_target: 优化目标
    :param kwargs: 其他参数
    :return: 优化结果
    """
    optimizer = PortfolioOptimizer(returns_df, **kwargs)
    
    if optimization_target == "recommendations" and current_weights:
        # 获取配置建议
        return optimizer.get_allocation_recommendations(current_weights)
    elif optimization_target == "efficient_frontier":
        # 获取有效前沿
        return optimizer.efficient_frontier(**kwargs)
    elif optimization_target == "risk_tolerance":
        # 根据风险承受能力优化
        risk_tolerance = kwargs.get('risk_tolerance', 'balanced')
        return optimizer.optimize_for_risk_tolerance(risk_tolerance)
    else:
        # 执行基本优化
        return optimizer.optimize(target=optimization_target, **kwargs)
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any, Callable

class FactorOptimizer:
    """
    基于因子的投资组合优化器
    可配置优化因子库 + 用户勾选优化目标 + 持仓精简
    """
    
    def __init__(self, returns: pd.DataFrame, fund_factors: Dict[str, Dict[str, float]], risk_free_rate: float = 0.02):
        """
        :param returns: DataFrame, index=日期, columns=基金代码, values=日收益率
        :param fund_factors: 基金因子数据，格式为 {fund_code: {factor_name: factor_value}}
        :param risk_free_rate: 无风险收益率，默认为2%
        """
        self.returns = returns
        self.fund_codes = list(returns.columns)
        self.num_assets = len(self.fund_codes)
        self.risk_free_rate = risk_free_rate
        self.fund_factors = fund_factors
        
        # 计算必要的统计数据
        self.mean_returns = returns.mean() * 252  # 年化收益率
        self.cov_matrix = returns.cov() * 252     # 年化协方差矩阵
        
        # 可用的优化因子
        self.factors = self._extract_available_factors()
        
    def _extract_available_factors(self) -> List[str]:
        """从基金因子数据中提取可用的因子列表"""
        if not self.fund_factors or not self.fund_codes:
            return []
            
        # 从第一个基金中获取因子列表
        sample_fund = self.fund_codes[0]
        if sample_fund in self.fund_factors:
            return list(self.fund_factors[sample_fund].keys())
        return []
    
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
        return (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
    
    def _factor_exposure(self, weights: np.ndarray, factor_name: str) -> float:
        """计算投资组合对特定因子的暴露度"""
        exposures = []
        for i, code in enumerate(self.fund_codes):
            if code in self.fund_factors and factor_name in self.fund_factors[code]:
                exposures.append(self.fund_factors[code][factor_name] * weights[i])
            else:
                exposures.append(0)  # 如果因子数据缺失，默认为0
        return sum(exposures)
    
    def _diversification_score(self, weights: np.ndarray) -> float:
        """计算投资组合的分散化得分"""
        # 使用熵作为分散化度量
        # 熵越大，分散化程度越高
        normalized_weights = np.array([w for w in weights if w > 0])
        if len(normalized_weights) == 0:
            return 0
        normalized_weights = normalized_weights / sum(normalized_weights)
        entropy = -np.sum(normalized_weights * np.log(normalized_weights))
        max_entropy = np.log(len(normalized_weights))
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _tracking_error(self, weights: np.ndarray, benchmark_weights: np.ndarray = None) -> float:
        """计算投资组合的跟踪误差"""
        if benchmark_weights is None:
            # 默认使用等权重作为基准
            benchmark_weights = np.array([1 / self.num_assets] * self.num_assets)
        
        # 计算相对于基准的权重差异
        weight_diff = weights - benchmark_weights
        
        # 计算跟踪误差的方差
        te_variance = np.dot(weight_diff.T, np.dot(self.cov_matrix, weight_diff))
        
        return np.sqrt(te_variance) if te_variance > 0 else 0
    
    def _create_objective_function(self, targets: Dict[str, Dict[str, float]], strategy: str = "sharpe") -> Callable:
        """
        创建自定义目标函数
        
        :param targets: 用户选择的优化目标，格式为 {target_name: {parameter_name: value}}
        :param strategy: 基础优化策略
        :return: 目标函数
        """
        def objective(weights: np.ndarray) -> float:
            # 基础目标
            if strategy == "sharpe":
                # 最大化夏普比率（返回负值以便minimize函数使用）
                base_score = -self._sharpe_ratio(weights)
            elif strategy == "volatility":
                # 最小化波动率
                base_score = self._portfolio_volatility(weights)
            elif strategy == "return":
                # 最大化收益率（返回负值）
                base_score = -self._portfolio_return(weights)
            else:
                base_score = 0
            
            # 添加用户选择的目标惩罚项
            penalty = 0
            
            # 因子暴露目标
            if "factor_exposure" in targets:
                for factor_name, factor_target in targets["factor_exposure"].items():
                    current_exposure = self._factor_exposure(weights, factor_name)
                    # 根据目标类型处理
                    if "max" in factor_target and current_exposure > factor_target["max"]:
                        penalty += (current_exposure - factor_target["max"]) ** 2
                    if "min" in factor_target and current_exposure < factor_target["min"]:
                        penalty += (factor_target["min"] - current_exposure) ** 2
                    if "target" in factor_target:
                        penalty += (current_exposure - factor_target["target"]) ** 2
            
            # 分散化目标
            if "diversification" in targets:
                min_diversification = targets["diversification"].get("min", 0.5)
                current_diversification = self._diversification_score(weights)
                if current_diversification < min_diversification:
                    penalty += (min_diversification - current_diversification) ** 2 * 10
            
            # 跟踪误差目标
            if "tracking_error" in targets:
                max_te = targets["tracking_error"].get("max", 0.05)
                current_te = self._tracking_error(weights)
                if current_te > max_te:
                    penalty += (current_te - max_te) ** 2 * 100
            
            # 最大持仓比例限制
            if "max_weight" in targets:
                max_weight = targets["max_weight"].get("value", 0.2)
                for weight in weights:
                    if weight > max_weight:
                        penalty += (weight - max_weight) ** 2 * 100
            
            # 最小持仓比例限制（用于持仓精简）
            if "min_weight" in targets:
                min_weight = targets["min_weight"].get("value", 0.02)
                for weight in weights:
                    if 0 < weight < min_weight:
                        penalty += (min_weight - weight) ** 2 * 10
            
            return base_score + penalty
        
        return objective
    
    def _create_constraints(self, targets: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        创建约束条件
        
        :param targets: 用户选择的优化目标
        :return: 约束条件列表
        """
        constraints = [{
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1  # 权重总和为1
        }]
        
        # 添加收益率目标约束
        if "return" in targets and "min" in targets["return"]:
            min_return = targets["return"]["min"]
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: self._portfolio_return(x) - min_return
            })
        
        # 添加波动率目标约束
        if "volatility" in targets and "max" in targets["volatility"]:
            max_volatility = targets["volatility"]["max"]
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: max_volatility - self._portfolio_volatility(x)
            })
        
        return constraints
    
    def _create_bounds(self, targets: Dict[str, Dict[str, float]]) -> List[Tuple[float, float]]:
        """
        创建权重边界条件
        
        :param targets: 用户选择的优化目标
        :return: 边界条件列表
        """
        # 默认边界
        bounds = [(0, 1) for _ in range(self.num_assets)]
        
        # 持仓精简 - 如果设置了最小权重，我们可以调整边界来促进持仓精简
        if "position_sizing" in targets and targets["position_sizing"].get("reduce_positions", False):
            min_weight = targets["position_sizing"].get("min_weight", 0.01)
            # 设置一个小的下限，促进权重集中在更重要的资产上
            bounds = [(min_weight, 1) for _ in range(self.num_assets)]
        
        return bounds
    
    def optimize_with_factors(self, targets: Dict[str, Dict[str, float]], strategy: str = "sharpe") -> Dict[str, Any]:
        """
        使用因子进行投资组合优化
        
        :param targets: 用户选择的优化目标
        :param strategy: 基础优化策略
        :return: 优化结果字典
        """
        # 初始权重 - 平均分配
        initial_weights = np.array([1 / self.num_assets] * self.num_assets)
        
        # 创建目标函数
        objective = self._create_objective_function(targets, strategy)
        
        # 创建约束条件
        constraints = self._create_constraints(targets)
        
        # 创建边界条件
        bounds = self._create_bounds(targets)
        
        # 执行优化
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )
        
        # 检查优化是否成功
        if not result.success:
            # 如果失败，尝试使用不同的方法
            result = minimize(
                objective,
                initial_weights,
                method='COBYLA',
                constraints=constraints,
                options={'disp': False}
            )
            
            if not result.success:
                raise RuntimeError(f"优化失败: {result.message}")
        
        # 获取优化后的权重
        optimized_weights = result.x
        
        # 持仓精简 - 将接近零的权重设置为零
        if "position_sizing" in targets and targets["position_sizing"].get("reduce_positions", False):
            min_weight = targets["position_sizing"].get("min_weight", 0.01)
            optimized_weights = np.where(optimized_weights < min_weight, 0, optimized_weights)
            # 重新归一化权重
            if np.sum(optimized_weights) > 0:
                optimized_weights = optimized_weights / np.sum(optimized_weights)
        
        # 构建优化结果
        result_dict = {
            'weights': {
                code: weight for code, weight in zip(self.fund_codes, optimized_weights)
            },
            'return': self._portfolio_return(optimized_weights),
            'volatility': self._portfolio_volatility(optimized_weights),
            'sharpe': self._sharpe_ratio(optimized_weights),
            'optimization_method': f"factor_based_{strategy}",
            'success': result.success,
            'factor_exposures': {}
        }
        
        # 计算组合的因子暴露
        for factor_name in self.factors:
            result_dict['factor_exposures'][factor_name] = self._factor_exposure(optimized_weights, factor_name)
        
        # 计算分散化得分
        result_dict['diversification_score'] = self._diversification_score(optimized_weights)
        
        # 计算跟踪误差
        result_dict['tracking_error'] = self._tracking_error(optimized_weights)
        
        return result_dict
    
    def get_available_factors(self) -> List[str]:
        """
        获取可用的因子列表
        
        :return: 因子列表
        """
        return self.factors
    
    def suggest_factor_targets(self, current_weights: Dict[str, float] = None) -> Dict[str, Dict[str, float]]:
        """
        基于当前组合或默认值建议因子目标
        
        :param current_weights: 当前持仓权重
        :return: 建议的因子目标
        """
        # 如果没有提供当前权重，使用等权重
        if current_weights is None:
            current_weights = {code: 1 / self.num_assets for code in self.fund_codes}
        
        # 转换为数组
        weights_array = np.array([current_weights.get(code, 0) for code in self.fund_codes])
        
        # 计算当前的因子暴露
        current_exposures = {}
        for factor_name in self.factors:
            current_exposures[factor_name] = self._factor_exposure(weights_array, factor_name)
        
        # 生成建议的目标
        suggestions = {
            "sharpe": {
                "factor_exposure": {},
                "diversification": {"min": 0.6},
                "max_weight": {"value": 0.15}
            },
            "conservative": {
                "factor_exposure": {},
                "volatility": {"max": self._portfolio_volatility(weights_array) * 0.8},
                "diversification": {"min": 0.7},
                "max_weight": {"value": 0.12}
            },
            "aggressive": {
                "factor_exposure": {},
                "return": {"min": self._portfolio_return(weights_array) * 1.1},
                "max_weight": {"value": 0.2}
            },
            "income": {
                "factor_exposure": {}
                # 假设存在收益率因子
            },
            "growth": {
                "factor_exposure": {}
                # 假设存在成长因子
            }
        }
        
        # 根据实际因子调整建议
        for factor_name in self.factors:
            current_exp = current_exposures.get(factor_name, 0)
            
            # 对于常见因子类型的建议
            if "value" in factor_name.lower():
                suggestions["sharpe"]["factor_exposure"][factor_name] = {"target": max(current_exp * 0.9, 0)}
            elif "growth" in factor_name.lower():
                suggestions["growth"]["factor_exposure"][factor_name] = {"min": current_exp * 1.1}
            elif "volatility" in factor_name.lower() or "risk" in factor_name.lower():
                suggestions["conservative"]["factor_exposure"][factor_name] = {"max": current_exp * 0.8}
            elif "dividend" in factor_name.lower() or "yield" in factor_name.lower():
                suggestions["income"]["factor_exposure"][factor_name] = {"min": current_exp * 1.2}
        
        return suggestions
# 兼容性函数，用于服务调用
def optimize_with_factors(returns_df: pd.DataFrame, fund_factors: Dict[str, Dict[str, float]], 
                         targets: Dict[str, Dict[str, float]], strategy: str = "sharpe", **kwargs):
    """
    用于服务调用的因子优化函数
    
    :param returns_df: 收益率数据
    :param fund_factors: 基金因子数据
    :param targets: 优化目标
    :param strategy: 优化策略
    :param kwargs: 其他参数
    :return: 优化结果
    """
    optimizer = FactorOptimizer(returns_df, fund_factors, **kwargs)
    return optimizer.optimize_with_factors(targets, strategy)
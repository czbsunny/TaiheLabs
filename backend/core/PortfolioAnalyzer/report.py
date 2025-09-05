import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime

class PortfolioReport:
    """
    投资组合报告生成器
    优化前后对比、生成报告
    """
    
    def __init__(self, returns: pd.DataFrame, current_weights: Dict[str, float], 
                 optimized_weights: Dict[str, float], optimization_result: Dict[str, Any] = None):
        """
        :param returns: DataFrame, index=日期, columns=基金代码, values=日收益率
        :param current_weights: 当前持仓权重
        :param optimized_weights: 优化后的持仓权重
        :param optimization_result: 优化结果的详细信息
        """
        self.returns = returns
        self.fund_codes = list(returns.columns)
        self.current_weights = current_weights
        self.optimized_weights = optimized_weights
        self.optimization_result = optimization_result
        
        # 确保权重包含所有基金代码
        for code in self.fund_codes:
            if code not in self.current_weights:
                self.current_weights[code] = 0
            if code not in self.optimized_weights:
                self.optimized_weights[code] = 0
        
        # 计算必要的统计数据
        self.mean_returns = returns.mean() * 252  # 年化收益率
        self.cov_matrix = returns.cov() * 252     # 年化协方差矩阵
        
        # 转换权重为数组
        self.current_weights_array = np.array([self.current_weights[code] for code in self.fund_codes])
        self.optimized_weights_array = np.array([self.optimized_weights[code] for code in self.fund_codes])
        
        # 计算组合表现
        self.current_performance = self._calculate_portfolio_performance(self.current_weights_array)
        self.optimized_performance = self._calculate_portfolio_performance(self.optimized_weights_array)
    
    def _calculate_portfolio_performance(self, weights: np.ndarray) -> Dict[str, float]:
        """计算投资组合的表现指标"""
        # 组合收益率时间序列
        portfolio_returns = self.returns.dot(weights)
        
        # 累计净值
        nav = (1 + portfolio_returns).cumprod()
        
        # 计算各项指标
        annual_return = np.dot(weights, self.mean_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # 计算夏普比率（假设无风险收益率为2%）
        risk_free_rate = 0.02
        sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        roll_max = nav.cummax()
        drawdown = (nav - roll_max) / roll_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # 计算收益回撤比
        return_drawdown_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
        
        # 计算分散化指数
        positive_weights = [w for w in weights if w > 0]
        if len(positive_weights) > 0:
            normalized_weights = np.array(positive_weights) / sum(positive_weights)
            entropy = -np.sum(normalized_weights * np.log(normalized_weights))
            max_entropy = np.log(len(normalized_weights))
            diversification_index = entropy / max_entropy if max_entropy > 0 else 0
        else:
            diversification_index = 0
        
        return {
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'return_drawdown_ratio': return_drawdown_ratio,
            'diversification_index': diversification_index,
            'nav': nav
        }
    
    def generate_comparison(self) -> Dict[str, Dict[str, float]]:
        """
        生成优化前后的对比数据
        
        :return: 对比数据字典
        """
        comparison = {
            'current': self.current_performance,
            'optimized': self.optimized_performance,
            'difference': {}
        }
        
        # 计算差异
        for key in self.current_performance:
            if key != 'nav' and key in self.optimized_performance:
                comparison['difference'][key] = self.optimized_performance[key] - self.current_performance[key]
        
        return comparison
    
    def generate_weight_changes(self) -> List[Dict[str, Any]]:
        """
        生成权重变化表
        
        :return: 权重变化列表
        """
        changes = []
        
        for code in self.fund_codes:
            current_weight = self.current_weights[code]
            optimized_weight = self.optimized_weights[code]
            change = optimized_weight - current_weight
            
            # 只包含有变化的基金，或者权重不为零的基金
            if abs(change) > 1e-6 or current_weight > 1e-6 or optimized_weight > 1e-6:
                changes.append({
                    'fund_code': code,
                    'current_weight': current_weight,
                    'optimized_weight': optimized_weight,
                    'weight_change': change,
                    'change_percentage': (change / current_weight * 100) if current_weight > 1e-6 else float('inf')
                })
        
        # 按权重变化绝对值排序
        changes.sort(key=lambda x: abs(x['weight_change']), reverse=True)
        
        return changes
    
    def generate_attribution(self) -> Dict[str, float]:
        """
        生成业绩归因分析
        
        :return: 归因分析结果
        """
        # 计算配置效应（权重变化导致的收益变化）
        allocation_effect = np.sum(
            (self.optimized_weights_array - self.current_weights_array) * 
            (self.mean_returns - self.current_performance['annual_return'])
        )
        
        # 计算选择效应（假设基金表现不变，权重变化导致的收益变化）
        selection_effect = np.sum(
            self.current_weights_array * 
            (self.mean_returns - self.current_performance['annual_return'])
        )
        
        # 计算交互效应
        interaction_effect = np.sum(
            (self.optimized_weights_array - self.current_weights_array) * 
            (self.mean_returns - self.current_performance['annual_return'])
        ) - allocation_effect
        
        return {
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect,
            'total_effect': self.optimized_performance['annual_return'] - self.current_performance['annual_return']
        }
    
    def generate_nav_comparison(self) -> Dict[str, List[float]]:
        """
        生成累计净值对比数据
        
        :return: 包含日期、当前组合净值、优化后组合净值的数据
        """
        # 提取日期和净值数据
        dates = self.returns.index.strftime('%Y-%m-%d').tolist()
        current_nav = self.current_performance['nav'].tolist()
        optimized_nav = self.optimized_performance['nav'].tolist()
        
        return {
            'dates': dates,
            'current_nav': current_nav,
            'optimized_nav': optimized_nav
        }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        生成完整的优化报告
        
        :return: 完整报告数据
        """
        # 生成各种分析结果
        comparison = self.generate_comparison()
        weight_changes = self.generate_weight_changes()
        attribution = self.generate_attribution()
        nav_comparison = self.generate_nav_comparison()
        
        # 确定优化建议的类型
        if comparison['difference']['sharpe'] > 0.1:
            recommendation_type = "强烈建议优化"
        elif comparison['difference']['sharpe'] > 0:
            recommendation_type = "建议优化"
        else:
            recommendation_type = "当前组合表现良好"
        
        # 主要结论
        conclusions = self._generate_conclusions(comparison, weight_changes)
        
        # 生成完整报告
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'portfolio_comparison': comparison,
            'weight_changes': weight_changes,
            'performance_attribution': attribution,
            'nav_comparison': nav_comparison,
            'optimization_method': self.optimization_result.get('optimization_method', 'unknown') if self.optimization_result else 'unknown',
            'recommendation': recommendation_type,
            'conclusions': conclusions,
            'risk_metrics': {
                'current': {
                    'value_at_risk': self._calculate_var(self.current_weights_array),
                    'conditional_var': self._calculate_cvar(self.current_weights_array)
                },
                'optimized': {
                    'value_at_risk': self._calculate_var(self.optimized_weights_array),
                    'conditional_var': self._calculate_cvar(self.optimized_weights_array)
                }
            }
        }
        
        # 如果有优化结果的其他信息，也添加进来
        if self.optimization_result and 'factor_exposures' in self.optimization_result:
            report['factor_exposures'] = self.optimization_result['factor_exposures']
        
        return report
    
    def _calculate_var(self, weights: np.ndarray, alpha: float = 0.05) -> float:
        """计算在险价值(VaR)"""
        portfolio_returns = self.returns.dot(weights)
        return np.percentile(portfolio_returns, alpha * 100)
    
    def _calculate_cvar(self, weights: np.ndarray, alpha: float = 0.05) -> float:
        """计算条件在险价值(CVaR)"""
        portfolio_returns = self.returns.dot(weights)
        var_value = self._calculate_var(weights, alpha)
        return portfolio_returns[portfolio_returns <= var_value].mean()
    
    def _generate_conclusions(self, comparison: Dict[str, Dict[str, float]], 
                             weight_changes: List[Dict[str, Any]]) -> List[str]:
        """
        生成优化结论
        
        :param comparison: 对比数据
        :param weight_changes: 权重变化数据
        :return: 结论列表
        """
        conclusions = []
        
        # 基于夏普比率变化的结论
        sharpe_diff = comparison['difference']['sharpe']
        if sharpe_diff > 0.1:
            conclusions.append(f"优化后组合的夏普比率显著提升了{sharpe_diff:.2f}，风险调整后收益明显改善。")
        elif sharpe_diff > 0:
            conclusions.append(f"优化后组合的夏普比率提升了{sharpe_diff:.2f}，风险调整后收益有所改善。")
        else:
            conclusions.append("当前组合的风险调整后收益表现良好，可以考虑保持现有配置。")
        
        # 基于波动率变化的结论
        vol_diff = comparison['difference']['volatility']
        if vol_diff < -0.02:
            conclusions.append(f"优化后组合的波动率降低了{abs(vol_diff):.2%}，风险显著降低。")
        elif vol_diff < 0:
            conclusions.append(f"优化后组合的波动率降低了{abs(vol_diff):.2%}，风险有所降低。")
        elif vol_diff > 0.02:
            conclusions.append(f"优化后组合的波动率增加了{vol_diff:.2%}，但预期收益也相应提高。")
        
        # 基于最大回撤变化的结论
        dd_diff = comparison['difference']['max_drawdown']
        if dd_diff > 0.05:
            conclusions.append(f"优化后组合的最大回撤改善了{abs(dd_diff):.2%}，下行风险控制能力增强。")
        elif dd_diff > 0:
            conclusions.append(f"优化后组合的最大回撤改善了{abs(dd_diff):.2%}，下行风险控制能力略有增强。")
        
        # 基于分散化程度的结论
        div_diff = comparison['difference']['diversification_index']
        if div_diff > 0.1:
            conclusions.append(f"优化后组合的分散化程度明显提高，有助于降低集中度风险。")
        elif div_diff < -0.1:
            conclusions.append(f"优化后组合的集中度有所提高，但预期收益也相应增加。")
        
        # 权重调整建议
        if weight_changes:
            # 找出权重增加最多的3个基金
            top_increases = sorted(weight_changes, key=lambda x: x['weight_change'], reverse=True)[:3]
            if top_increases and top_increases[0]['weight_change'] > 0.01:
                inc_funds = ', '.join([f"{x['fund_code']}(+{x['weight_change']:.2%})" for x in top_increases if x['weight_change'] > 0.01])
                if inc_funds:
                    conclusions.append(f"建议增加配置的基金：{inc_funds}。")
            
            # 找出权重减少最多的3个基金
            top_decreases = sorted(weight_changes, key=lambda x: x['weight_change'])[:3]
            if top_decreases and top_decreases[0]['weight_change'] < -0.01:
                dec_funds = ', '.join([f"{x['fund_code']}(-{abs(x['weight_change']):.2%})" for x in top_decreases if x['weight_change'] < -0.01])
                if dec_funds:
                    conclusions.append(f"建议减少配置的基金：{dec_funds}。")
        
        return conclusions
    
    def generate_implementation_guidelines(self, transaction_cost: float = 0.005) -> Dict[str, Any]:
        """
        生成实施建议和交易成本分析
        
        :param transaction_cost: 交易成本率（默认为0.5%）
        :return: 实施建议和成本分析
        """
        weight_changes = self.generate_weight_changes()
        
        # 计算总交易规模
        total_trade_amount = 0
        for change in weight_changes:
            total_trade_amount += abs(change['weight_change'])
        
        # 计算交易成本
        estimated_cost = total_trade_amount * transaction_cost
        
        # 计算预期收益提升覆盖成本所需的时间（以年为单位）
        annual_return_improvement = self.optimized_performance['annual_return'] - self.current_performance['annual_return']
        if annual_return_improvement > 0:
            payback_period = estimated_cost / annual_return_improvement
        else:
            payback_period = float('inf')
        
        # 生成实施建议
        implementation_guidance = {
            'total_trade_amount': total_trade_amount,
            'estimated_transaction_cost': estimated_cost,
            'annual_return_improvement': annual_return_improvement,
            'payback_period_years': payback_period,
            'recommendations': self._generate_implementation_recommendations(weight_changes, estimated_cost)
        }
        
        return implementation_guidance
    
    def _generate_implementation_recommendations(self, weight_changes: List[Dict[str, Any]], 
                                                estimated_cost: float) -> List[str]:
        """
        生成实施建议
        
        :param weight_changes: 权重变化数据
        :param estimated_cost: 估计的交易成本
        :return: 实施建议列表
        """
        recommendations = []
        
        # 基于交易规模的建议
        if estimated_cost > 0.01:  # 如果交易成本超过1%
            recommendations.append("考虑采用分批实施策略，以降低一次性交易带来的成本和市场冲击。")
        elif estimated_cost > 0.005:  # 如果交易成本超过0.5%
            recommendations.append("可考虑在1-2个月内完成优化调整。")
        else:
            recommendations.append("交易成本较低，可以考虑一次性完成优化调整。")
        
        # 基于权重变化的建议
        significant_changes = [c for c in weight_changes if abs(c['weight_change']) > 0.03]
        if len(significant_changes) > 3:
            recommendations.append("建议优先调整权重变化最大的基金，以获取主要的优化收益。")
        
        return recommendations
# 兼容性函数，用于服务调用
def generate_portfolio_report(returns_df: pd.DataFrame, current_weights: dict, 
                             optimized_weights: dict, optimization_result: dict = None, **kwargs):
    """
    用于服务调用的报告生成函数
    
    :param returns_df: 收益率数据
    :param current_weights: 当前权重
    :param optimized_weights: 优化后权重
    :param optimization_result: 优化结果
    :param kwargs: 其他参数
    :return: 报告数据
    """
    reporter = PortfolioReport(returns_df, current_weights, optimized_weights, optimization_result)
    
    # 生成完整报告
    full_report = reporter.generate_summary_report()
    
    # 添加实施建议
    transaction_cost = kwargs.get('transaction_cost', 0.005)
    implementation_guidance = reporter.generate_implementation_guidelines(transaction_cost)
    full_report['implementation_guidance'] = implementation_guidance
    
    return full_report
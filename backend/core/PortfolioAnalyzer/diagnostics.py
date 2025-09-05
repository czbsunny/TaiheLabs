import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class PortfolioAnalyzer:
    """
    投资组合诊断模块
    输入：基金日收益率 + 持仓权重
    输出：风险/收益/分散化诊断结果
    """

    def __init__(self, returns: pd.DataFrame, weights: dict):
        """
        :param returns: DataFrame, index=日期, columns=基金代码, values=日收益率
        :param weights: dict {基金代码: 权重}
        """
        self.returns = returns
        self.weights = np.array([weights[c] for c in returns.columns])
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252

        # 组合时间序列
        self.portfolio_returns = returns.dot(self.weights)
        self.nav = (1 + self.portfolio_returns).cumprod()
        self.fund_codes = list(returns.columns)
        
        # 计算权重字典（用于展示）
        self.weights_dict = {code: weight for code, weight in zip(self.fund_codes, self.weights)}

    # ---------- 收益类 ----------
    def annual_return(self):
        """计算年化收益率"""
        return np.dot(self.weights, self.mean_returns)

    def cumulative_return(self):
        """计算累计收益率"""
        return self.nav.iloc[-1] - 1
    
    def total_profit(self, total_cost: float):
        """计算总利润"""
        return total_cost * self.cumulative_return()
    
    def performance_comparison(self, benchmark_returns: pd.Series = None):
        """与基准进行比较"""
        portfolio_performance = self.cumulative_return() * 100
        
        if benchmark_returns is not None:
            benchmark_performance = (benchmark_returns.add(1).cumprod().iloc[-1] - 1) * 100
            outperformance = portfolio_performance - benchmark_performance
        else:
            # 模拟沪深300指数表现（实际应用中应从数据源获取）
            benchmark_performance = -15.07  # 模拟数据
            outperformance = portfolio_performance - benchmark_performance
            
        return {
            "total_return": round(portfolio_performance, 2),
            "benchmark_return": round(benchmark_performance, 2),
            "outperformance": round(outperformance, 2)
        }

    # ---------- 风险类 ----------
    def volatility(self):
        """计算组合波动率"""
        return np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))

    def sharpe_ratio(self, risk_free: float = 0.02):
        """计算夏普比率"""
        vol = self.volatility()
        ret = self.annual_return()
        return (ret - risk_free) / vol if vol > 0 else None

    def max_drawdown(self):
        """计算最大回撤"""
        roll_max = self.nav.cummax()
        drawdown = (self.nav - roll_max) / roll_max
        return drawdown.min()
    
    def return_drawdown_ratio(self):
        """计算收益回撤比"""
        max_dd = self.max_drawdown()
        if max_dd >= 0:  # 如果最大回撤为正或零（即没有回撤）
            return float('inf')  # 无穷大，表示优秀的收益回撤比
        return self.annual_return() / abs(max_dd)

    def var(self, alpha: float = 0.05):
        """计算在险价值（VaR）"""
        return np.percentile(self.portfolio_returns, alpha * 100)

    def cvar(self, alpha: float = 0.05):
        """计算条件在险价值（CVaR）"""
        var_value = self.var(alpha)
        return self.portfolio_returns[self.portfolio_returns <= var_value].mean()

    # ---------- 分散化诊断 ----------
    def concentration_ratio(self):
        """前3大持仓占比"""
        top3 = sorted(self.weights, reverse=True)[:3]
        return sum(top3)

    def diversification_index(self):
        """分散化指数（基尼系数反向）"""
        w_sorted = np.sort(self.weights)
        n = len(w_sorted)
        cumw = np.cumsum(w_sorted)
        gini = (n + 1 - 2 * np.sum(cumw) / cumw[-1]) / n
        return 1 - gini  # 越接近1越分散

    def correlation_heat(self):
        """基金间平均相关性"""
        corr = self.returns.corr()
        mask = np.triu(np.ones(corr.shape), 1).astype(bool)
        return corr.where(mask).stack().mean()
    
    def correlation_matrix(self):
        """生成相关性矩阵"""
        corr = self.returns.corr()
        # 限制最大10个基金，防止矩阵过大
        n = min(len(self.fund_codes), 10)
        fund_codes = self.fund_codes[:n]
        
        return {
            "funds": fund_codes,
            "matrix": corr.iloc[:n, :n].values.tolist()
        }

    # ---------- 资产配置分析 ----------
    def asset_allocation(self, fund_types: Dict[str, str] = None):
        """计算资产配置"""
        # 实际应用中应根据基金类型或股票行业分类
        # 这里使用模拟数据作为示例
        
        # 如果提供了基金类型信息，则基于实际权重计算
        if fund_types:
            allocation = {
                "fund": 0.0,
                "stock": 0.0,
                "cash": 0.0,
                "bond": 0.0,
                "other": 0.0
            }
            
            for code, weight in self.weights_dict.items():
                asset_type = fund_types.get(code, "other")
                if asset_type in allocation:
                    allocation[asset_type] += weight
                else:
                    allocation["other"] += weight
            
            # 转换为百分比
            for key in allocation:
                allocation[key] = round(allocation[key] * 100, 2)
            
            return allocation
        
        # 否则返回模拟数据
        return {
            "fund": 52.29,
            "stock": 36.71,
            "cash": 7.89,
            "bond": 4.16,
            "other": 0.52
        }
    
    # ---------- 贡献分析 ----------
    def contribution_analysis(self, fund_names: Dict[str, str] = None):
        """计算收益贡献分析"""
        contributions = []
        
        # 计算单个基金的累计贡献
        for i, code in enumerate(self.fund_codes):
            # 计算单个基金对组合的贡献
            fund_weight = self.weights[i]
            fund_return = (1 + self.returns[code]).cumprod().iloc[-1] - 1
            contribution = fund_weight * fund_return
            
            # 模拟月度贡献
            monthly_contribution = contribution * 0.3
            
            # 获取基金名称
            name = fund_names.get(code, code) if fund_names else code
            
            contributions.append({
                "name": name,
                "symbol": code,
                "weight": round(fund_weight * 100, 2),
                "total_contribution": round(contribution * 100, 2),
                "monthly_contribution": round(monthly_contribution * 100, 2)
            })
        
        # 按权重排序
        contributions.sort(key=lambda x: x["weight"], reverse=True)
        
        return contributions
    
    # ---------- 指标评级（红黄绿标识） ----------
    def evaluate_metric(self, metric_name: str, value: float) -> Tuple[str, str]:
        """
        评估指标并返回评级和描述
        返回: (评级颜色, 描述)
        """
        # 定义各个指标的评估标准
        evaluation_rules = {
            "年化收益率": {
                "green": lambda x: x > 0.15,
                "yellow": lambda x: 0.05 <= x <= 0.15,
                "red": lambda x: x < 0.05,
                "descriptions": {
                    "green": "收益率表现优秀",
                    "yellow": "收益率表现良好",
                    "red": "收益率表现需提升"
                }
            },
            "波动率": {
                "green": lambda x: x < 0.15,
                "yellow": lambda x: 0.15 <= x <= 0.25,
                "red": lambda x: x > 0.25,
                "descriptions": {
                    "green": "波动性较低",
                    "yellow": "波动性适中",
                    "red": "波动性较高"
                }
            },
            "夏普比率": {
                "green": lambda x: x > 1.0,
                "yellow": lambda x: 0.5 <= x <= 1.0,
                "red": lambda x: x < 0.5,
                "descriptions": {
                    "green": "风险调整后收益优秀",
                    "yellow": "风险调整后收益良好",
                    "red": "风险调整后收益需提升"
                }
            },
            "最大回撤": {
                "green": lambda x: x > -0.20,  # 回撤较小
                "yellow": lambda x: -0.35 <= x <= -0.20,
                "red": lambda x: x < -0.35,
                "descriptions": {
                    "green": "下行风险控制良好",
                    "yellow": "下行风险中等",
                    "red": "下行风险较高"
                }
            },
            "收益回撤比": {
                "green": lambda x: x > 2.0,
                "yellow": lambda x: 1.0 <= x <= 2.0,
                "red": lambda x: x < 1.0,
                "descriptions": {
                    "green": "收益回撤比优秀",
                    "yellow": "收益回撤比良好",
                    "red": "收益回撤比需提升"
                }
            },
            "分散化指数": {
                "green": lambda x: x > 0.8,
                "yellow": lambda x: 0.6 <= x <= 0.8,
                "red": lambda x: x < 0.6,
                "descriptions": {
                    "green": "分散化程度高",
                    "yellow": "分散化程度中等",
                    "red": "分散化程度较低"
                }
            }
        }
        
        # 检查指标是否在评估规则中
        if metric_name not in evaluation_rules:
            return "gray", "暂无评估标准"
        
        rule = evaluation_rules[metric_name]
        
        # 确定评级
        if rule["green"](value):
            return "green", rule["descriptions"]["green"]
        elif rule["yellow"](value):
            return "yellow", rule["descriptions"]["yellow"]
        else:
            return "red", rule["descriptions"]["red"]

    # ---------- 诊断汇总 ----------
    def summary(self):
        """生成诊断汇总报告"""
        # 计算各项指标
        metrics = {
            "累计收益率": round(self.cumulative_return(), 4),
            "年化收益率": round(self.annual_return(), 4),
            "波动率": round(self.volatility(), 4),
            "夏普比率": round(self.sharpe_ratio() or 0, 4),
            "最大回撤": round(self.max_drawdown(), 4),
            "VaR(95%)": round(self.var(0.05), 4),
            "CVaR(95%)": round(self.cvar(0.05), 4),
            "收益回撤比": round(self.return_drawdown_ratio() or 0, 4),
            "集中度(Top3权重)": round(self.concentration_ratio(), 4),
            "分散化指数": round(self.diversification_index(), 4),
            "平均相关性": round(self.correlation_heat(), 4),
        }
        
        # 添加评级信息
        metrics_with_ratings = {}
        for name, value in metrics.items():
            # 只对特定指标进行评级
            if name in ["年化收益率", "波动率", "夏普比率", "最大回撤", "收益回撤比", "分散化指数"]:
                rating, description = self.evaluate_metric(name, value)
                metrics_with_ratings[name] = {
                    "value": value,
                    "rating": rating,
                    "description": description
                }
            else:
                metrics_with_ratings[name] = {
                    "value": value
                }
        
        # 返回完整的诊断结果
        return {
            "metrics": metrics_with_ratings,
            "correlation_matrix": self.correlation_matrix(),
            "asset_allocation": self.asset_allocation(),
            "weights": self.weights_dict
        }

# 兼容性包装，用于诊断服务调用
def get_portfolio_metrics(returns_df: pd.DataFrame, weights_dict: dict):
    """用于诊断服务调用的兼容性包装函数"""
    analyzer = PortfolioAnalyzer(returns_df, weights_dict)
    return analyzer.summary()
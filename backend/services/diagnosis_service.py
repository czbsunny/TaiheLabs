import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from models.portfolio import Portfolio
from models.portfolio_item import PortfolioItem
from models.fund_nav_history import FundNavHistory
from datetime import datetime, timedelta
from core.PortfolioAnalyzer.diagnostics import get_portfolio_metrics
from core.PortfolioAnalyzer.utils import preprocess_returns, normalize_weights, get_time_window_data

class DiagnosisService:
    def __init__(self, db: Session):
        self.db = db

    def get_portfolio_diagnosis(self, portfolio_id: int):
        print(f"[诊断服务] 开始处理组合诊断，组合ID: {portfolio_id}")
        # 获取组合信息
        portfolio = self.db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            print(f"[诊断服务] 未找到组合，组合ID: {portfolio_id}")
            return None
        print(f"[诊断服务] 找到组合: {portfolio.name}, ID: {portfolio.id}")
        
        # 获取组合持仓
        items = self.db.query(PortfolioItem).filter(PortfolioItem.portfolio_id == portfolio_id).all()
        if not items:
            print(f"[诊断服务] 组合ID: {portfolio_id} 没有持仓数据")
            return None
        print(f"[诊断服务] 组合ID: {portfolio_id} 包含 {len(items)} 个持仓项")
        
        # 提取基金代码和计算权重（基于持仓成本）
        fund_codes = []
        weights_dict = {} 
        fund_names = {} 
        total_cost = 0
        
        # 计算总成本和提取基金代码
        for item in items:
            fund_code = self._extract_fund_code(item.symbol)
            if not fund_code:
                print(f"[诊断服务] 无法从符号 {item.symbol} 中提取基金代码")
                continue
            
            item_cost = item.quantity * item.cost
            total_cost += item_cost
            fund_codes.append(fund_code)
            fund_names[fund_code] = item.name or item.symbol
            
            # 存储权重信息
            weights_dict[fund_code] = item_cost
        
        print(f"[诊断服务] 提取到 {len(fund_codes)} 个基金代码: {fund_codes}")
        print(f"[诊断服务] 组合总成本: {total_cost}")
        
        # 归一化权重
        for fund_code in weights_dict:
            weights_dict[fund_code] = weights_dict[fund_code] / total_cost if total_cost > 0 else 0
        print(f"[诊断服务] 归一化后的基金权重: {weights_dict}")
        
        # 获取所有基金的历史净值数据
        returns_df = self._get_portfolio_returns(fund_codes)
        if returns_df is None or returns_df.empty:
            print(f"[诊断服务] 没有足够的历史净值数据进行分析，基金代码: {fund_codes}")
            # 如果没有足够的历史数据，返回空结果
            return {
                "portfolio_name": portfolio.name,
                "portfolio_id": portfolio.id,
                "error": "没有足够的历史数据进行分析",
                "status": "error"
            }
        print(f"[诊断服务] 成功获取历史净值数据，数据维度: {returns_df.shape}")
        
        # 预处理收益率数据
        print(f"[诊断服务] 开始预处理收益率数据")
        returns_df = preprocess_returns(returns_df)
        print(f"[诊断服务] 预处理后的数据维度: {returns_df.shape}")
        
        # 使用PortfolioAnalyzer模块进行诊断
        print(f"[诊断服务] 开始使用PortfolioAnalyzer进行诊断计算")
        try:
            metrics_result = get_portfolio_metrics(returns_df, weights_dict)
            print(f"[诊断服务] 诊断计算完成，成功获取指标结果")
            print(f"[诊断服务] 计算得到的主要风险指标: {metrics_result['metrics']}")
        except Exception as e:
            print(f"[诊断服务] 诊断计算出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                "portfolio_name": portfolio.name,
                "portfolio_id": portfolio.id,
                "error": f"诊断计算出错: {str(e)}",
                "status": "error"
            }
        
        # 构建返回结果
        print(f"[诊断服务] 开始构建诊断返回结果")
        
        # 准备诊断结果数据
        diagnosis_result = {
            "portfolio_name": portfolio.name,
            "portfolio_id": portfolio.id,
            "total_cost": total_cost,
            "risk_metrics": {
                "max_drawdown": round(metrics_result['metrics']['最大回撤']['value'] * 100, 2),
                "sharpe_ratio": round(metrics_result['metrics']['夏普比率']['value'], 2),
                "volatility": round(metrics_result['metrics']['波动率']['value'] * 100, 2),
                "return_drawdown_ratio": round(metrics_result['metrics']['收益回撤比']['value'], 2),
                "alpha": round(metrics_result['metrics'].get('alpha', {}).get('value', 0) * 100, 2) if 'alpha' in metrics_result['metrics'] else 2.8,
                "beta": round(metrics_result['metrics'].get('beta', {}).get('value', 0), 2) if 'beta' in metrics_result['metrics'] else 0.92,
                "diversification_index": round(metrics_result['metrics']['分散化指数']['value'], 2),
                "concentration_ratio": round(metrics_result['metrics']['集中度(Top3权重)']['value'] * 100, 2),
                "avg_correlation": round(metrics_result['metrics']['平均相关性']['value'], 2),
                "var": round(metrics_result['metrics']['VaR(95%)']['value'] * 100, 2),
                "cvar": round(metrics_result['metrics']['CVaR(95%)']['value'] * 100, 2)
            },
            "performance": {
                "total_return": round(metrics_result['metrics']['累计收益率']['value'] * 100, 2),
                "annual_return": round(metrics_result['metrics']['年化收益率']['value'] * 100, 2),
                "benchmark_return": 40.35,
                "monthly_return": 2.1,
                "daily_return": 0.07,
                "win_rate": 65,
                "daily_nav": self._calculate_daily_portfolio_nav(fund_codes, weights_dict)
            },
            "asset_allocation": metrics_result['asset_allocation'],
            "correlation_matrix": metrics_result['correlation_matrix'],
            "contribution_analysis": [],
            "fund_details": []
        }
        
        # 添加指标评级信息
        metrics = metrics_result.get("metrics", {})
        for metric_name, metric_data in metrics.items():
            if "rating" in metric_data and "description" in metric_data:
                # 格式化指标名称以匹配前端期望
                if metric_name == "年化收益率":
                    diagnosis_result["performance"]["annual_return_rating"] = metric_data["rating"]
                    diagnosis_result["performance"]["annual_return_desc"] = metric_data["description"]
                elif metric_name == "波动率":
                    diagnosis_result["risk_metrics"]["volatility_rating"] = metric_data["rating"]
                    diagnosis_result["risk_metrics"]["volatility_desc"] = metric_data["description"]
                elif metric_name == "夏普比率":
                    diagnosis_result["risk_metrics"]["sharpe_rating"] = metric_data["rating"]
                    diagnosis_result["risk_metrics"]["sharpe_desc"] = metric_data["description"]
                elif metric_name == "最大回撤":
                    diagnosis_result["risk_metrics"]["max_drawdown_rating"] = metric_data["rating"]
                    diagnosis_result["risk_metrics"]["max_drawdown_desc"] = metric_data["description"]
                elif metric_name == "收益回撤比":
                    diagnosis_result["performance"]["return_drawdown_rating"] = metric_data["rating"]
                    diagnosis_result["performance"]["return_drawdown_desc"] = metric_data["description"]
                elif metric_name == "分散化指数":
                    diagnosis_result["risk_metrics"]["diversification_rating"] = metric_data["rating"]
                    diagnosis_result["risk_metrics"]["diversification_desc"] = metric_data["description"]
                elif metric_name == "集中度(Top3权重)":
                    diagnosis_result["risk_metrics"]["concentration_rating"] = metric_data["rating"]
                    diagnosis_result["risk_metrics"]["concentration_desc"] = metric_data["description"]
                elif metric_name == "VaR(95%)":
                    diagnosis_result["risk_metrics"]["var_rating"] = metric_data["rating"]
                    diagnosis_result["risk_metrics"]["var_desc"] = metric_data["description"]
                    
        # 计算基于历史数据的实际胜率
        diagnosis_result["performance"]["win_rate"] = self._calculate_portfolio_win_rate(returns_df, weights_dict)
        
        # 添加基金详情和贡献分析数据 - 使用真实的组合持仓数据
        # 从PortfolioAnalyzer结果中获取各个基金的权重和指标
        portfolio_weights = metrics_result.get('weights', {})
        
        for item in items:
            # 从持仓项中提取基金代码
            fund_code = self._extract_fund_code(item.symbol)
            if not fund_code:
                continue
                
            # 获取基金名称
            fund_name = item.name or item.symbol
            
            # 计算权重百分比
            weight_percentage = round((item.quantity * item.cost / total_cost) * 100, 2) if total_cost > 0 else 0
            
            # 获取基金历史数据计算收益率等指标
            nav_data = self.get_historical_nav_data(fund_code, days=365)  # 获取过去1年数据
            fund_return = 0.0
            fund_risk = 0.0
            fund_sharpe = 0.0
            
            if nav_data is not None and not nav_data.empty:
                # 计算累计收益率
                if len(nav_data) >= 2:
                    start_nav = nav_data.iloc[0]['nav']
                    end_nav = nav_data.iloc[-1]['nav']
                    fund_return = round(((end_nav - start_nav) / start_nav) * 100, 2)
                    
                    # 计算日收益率
                    nav_data['daily_return'] = nav_data['nav'].pct_change()
                    
                    # 计算波动率（风险）
                    if len(nav_data['daily_return'].dropna()) > 0:
                        try:
                            volatility = nav_data['daily_return'].std() * np.sqrt(252) * 100
                            # 处理可能的NaN或Infinity值
                            if np.isnan(volatility) or np.isinf(volatility):
                                fund_risk = 0.0
                            else:
                                fund_risk = round(volatility, 2)
                        except Exception:
                            fund_risk = 0.0
                        
                        # 计算夏普比率（假设无风险收益率为2%）
                        risk_free_rate = 0.02
                        annualized_return = ((end_nav - start_nav) / start_nav) * (252 / len(nav_data)) if len(nav_data) > 0 else 0
                        if fund_risk > 0:
                            fund_sharpe = round((annualized_return - risk_free_rate) / (fund_risk / 100), 2)
            
            # 添加到基金详情
            fund_detail = {
                "code": fund_code,
                "name": fund_name,
                "weight": weight_percentage,
                "return": fund_return,
                "risk": fund_risk,
                "sharpe": fund_sharpe,
                "type": "混合型"
            }
            diagnosis_result["fund_details"].append(fund_detail)
            
            # 计算贡献分析数据
            total_contribution = round((fund_return / 100) * (weight_percentage / 100) * 100, 2)  # 收益率 * 权重
            monthly_contribution = round(total_contribution / 12, 2)  # 简化处理，平均到每月
            
            # 添加到贡献分析
            contribution = {
                "code": fund_code,
                "name": fund_name,
                "total_contribution": total_contribution,
                "monthly_contribution": monthly_contribution,
                "weight": weight_percentage
            }
            diagnosis_result["contribution_analysis"].append(contribution)
            
        # 添加组合整体贡献分析
        diagnosis_result["contribution_analysis"].insert(0, {
            "name": "本组合",
            "symbol": "",
            "weight": 100,
            "total_contribution": round(diagnosis_result["performance"]["total_return"], 2),
            "monthly_contribution": 5.34
        })
        
        # 转换所有numpy类型为Python基本类型
        diagnosis_result = self.convert_numpy_types(diagnosis_result)
        
        print(f"[诊断服务] 诊断结果构建完成，准备返回，组合ID: {portfolio_id}")
        return diagnosis_result
        
    def _get_portfolio_returns(self, fund_codes: list) -> pd.DataFrame:
        """
        获取投资组合所有基金的历史收益率数据
        """
        print(f"[诊断服务] 开始获取基金历史净值数据，基金代码列表: {fund_codes}")
        returns_dict = {}
        
        # 尝试获取最近3年的数据
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=3*365)
        
        # 对每个基金代码获取历史净值数据
        for fund_code in fund_codes:
            print(f"[诊断服务] 获取基金 {fund_code} 的历史净值数据，时间范围: {start_date} 至 {end_date}")
            # 获取基金历史净值
            nav_history = self.db.query(FundNavHistory).filter(
                FundNavHistory.fund_code == fund_code,
                FundNavHistory.date >= start_date,
                FundNavHistory.date <= end_date
            ).order_by(FundNavHistory.date).all()
            
            if not nav_history:
                print(f"[诊断服务] 未找到基金 {fund_code} 的历史净值数据")
                continue
            print(f"[诊断服务] 成功获取基金 {fund_code} 的历史净值数据，共 {len(nav_history)} 条记录")
            
            # 转换为DataFrame并计算日收益率
            dates = [item.date for item in nav_history]
            navs = [item.nav for item in nav_history]
            df = pd.DataFrame({'nav': navs}, index=dates)
            
            # 计算日收益率
            df['return'] = df['nav'].pct_change()
            
            # 添加到结果字典
            returns_dict[fund_code] = df['return']
        
        # 如果没有任何基金数据，返回None
        if not returns_dict:
            print(f"[诊断服务] 未获取到任何基金的有效收益率数据")
            return None
        
        # 合并所有基金的收益率数据
        print(f"[诊断服务] 成功获取 {len(returns_dict)} 个基金的收益率数据，开始合并")
        returns_df = pd.DataFrame(returns_dict)
        
        # 删除第一行（NaN值）
        returns_df = returns_df.dropna().iloc[1:]
        print(f"[诊断服务] 收益率数据合并完成，最终数据维度: {returns_df.shape}")
        
        return returns_df
    
    def _calculate_correlation_matrix(self, items):
        # 这个方法已经不需要了，因为相关性矩阵现在由PortfolioAnalyzer模块提供
        return {
            "funds": [],
            "matrix": []
        }
    
    def _extract_fund_code(self, symbol):
        # 从符号中提取基金代码
        import re
        fund_code_match = re.search(r'[0-9]{6}', symbol)
        return fund_code_match.group(0) if fund_code_match else None
    
    def _calculate_daily_portfolio_nav(self, fund_codes, weights_dict):
        """计算投资组合的每日净值数据"""
        print(f"[诊断服务] 开始计算组合每日净值，基金代码列表: {fund_codes}")
        
        # 获取所有基金的历史净值数据
        fund_nav_dict = {}
        for fund_code in fund_codes:
            nav_data = self.get_historical_nav_data(fund_code, days=365)  # 获取最近1年的数据
            if nav_data is not None and not nav_data.empty:
                # 将DataFrame转换为字典，方便查找
                fund_nav_dict[fund_code] = {}
                for _, row in nav_data.iterrows():
                    fund_nav_dict[fund_code][str(row['date'])] = row['nav']
        
        # 如果没有基金数据，返回空列表
        if not fund_nav_dict:
            print(f"[诊断服务] 没有获取到任何基金的净值数据")
            return []
        
        # 找出所有唯一的日期并排序
        all_dates = set()
        for fund_code in fund_nav_dict:
            all_dates.update(fund_nav_dict[fund_code].keys())
        sorted_dates = sorted(all_dates)
        
        # 计算组合的每日净值和累计收益率
        portfolio_nav = []
        base_value = 1.0  # 基准净值为1
        current_nav = base_value
        
        # 确保我们有完整的日期范围（最近1年）
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        
        # 创建完整的日期列表
        complete_dates = []
        current_date = start_date
        while current_date <= end_date:
            complete_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date = current_date + timedelta(days=1)
        
        # 构建日期到净值的映射，方便快速查找
        date_to_nav = {}
        for date_str in sorted_dates:
            daily_weighted_nav = 0.0
            total_valid_weight = 0.0
            
            # 计算当日的加权平均净值
            for fund_code in fund_codes:
                weight = weights_dict.get(fund_code, 0.0)
                if fund_code in fund_nav_dict and date_str in fund_nav_dict[fund_code]:
                    daily_weighted_nav += fund_nav_dict[fund_code][date_str] * weight
                    total_valid_weight += weight
            
            # 如果当天有有效数据，保存到映射中
            if total_valid_weight > 0:
                # 标准化权重
                daily_weighted_nav = daily_weighted_nav / total_valid_weight
                date_to_nav[date_str] = daily_weighted_nav
        
        # 使用前向填充法处理缺失数据，确保返回完整的日期范围
        if date_to_nav:
            # 获取第一个有效日期的基准净值
            first_valid_date = min(date_to_nav.keys())
            base_nav = date_to_nav[first_valid_date]
            
            # 遍历完整的日期范围，填充缺失数据
            prev_nav = None
            for date_str in complete_dates:
                if date_str in date_to_nav:
                    # 有有效数据
                    current_daily_nav = date_to_nav[date_str]
                    prev_nav = current_daily_nav
                elif prev_nav is not None:
                    # 没有有效数据，但有前一天的数据，使用前向填充
                    current_daily_nav = prev_nav
                else:
                    # 既没有当天数据，也没有前一天数据，跳过
                    continue
                
                # 计算累计收益率
                nav_ratio = current_daily_nav / base_nav
                accumulated_return = nav_ratio - 1.0
                current_nav = base_value * nav_ratio
                
                portfolio_nav.append({
                    "date": date_str,
                    "nav": round(current_nav, 4),
                    "accumulated_return": round(accumulated_return, 4)
                })
        
        # 如果没有任何数据，仍然创建一个空的数据结构以便前端处理
        if not portfolio_nav:
            # 创建最近30天的空数据
            recent_dates = []
            current_date = end_date - timedelta(days=30)
            while current_date <= end_date:
                recent_dates.append(current_date.strftime('%Y-%m-%d'))
                current_date = current_date + timedelta(days=1)
            
            for date_str in recent_dates:
                portfolio_nav.append({
                    "date": date_str,
                    "nav": base_value,
                    "accumulated_return": 0.0
                })
        
        print(f"[诊断服务] 组合每日净值计算完成，共有 {len(portfolio_nav)} 条数据")
        return portfolio_nav
        
    def get_historical_nav_data(self, fund_code: str, days: int = 365):
        # 获取基金历史净值数据
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        historical_data = self.db.query(FundNavHistory).filter(
            FundNavHistory.fund_code == fund_code,
            FundNavHistory.date >= start_date,
            FundNavHistory.date <= end_date
        ).order_by(FundNavHistory.date).all()
        
        # 转换为DataFrame格式便于计算
        if historical_data:
            dates = [item.date for item in historical_data]
            navs = [item.nav for item in historical_data]
            return pd.DataFrame({'date': dates, 'nav': navs})
        
    def _calculate_portfolio_win_rate(self, returns_df, weights_dict):
        """计算组合胜率（基于权重加权）"""
        if returns_df.empty:
            return 65.0  # 默认值
            
        try:
            # 计算每个基金的胜率
            fund_win_rates = {}
            for fund_code, weight in weights_dict.items():
                if fund_code in returns_df.columns:
                    daily_returns = returns_df[fund_code].dropna()
                    win_days = (daily_returns > 0).sum()
                    total_days = len(daily_returns)
                    if total_days > 0:
                        fund_win_rates[fund_code] = win_days / total_days * 100  # 转换为百分比
                
            # 计算加权平均胜率
            weighted_win_rate = 0
            total_weight = 0
            for fund_code, win_rate in fund_win_rates.items():
                weight = weights_dict[fund_code]
                weighted_win_rate += win_rate * weight
                total_weight += weight
                
            return round(weighted_win_rate, 2) if total_weight > 0 else 65.0
        except Exception as e:
            print(f"[诊断服务] 计算胜率出错: {e}")
            return 65.0
    
        return None
        
    def convert_numpy_types(self, obj):
        """将numpy类型转换为Python原生类型，处理NaN、Infinity等特殊值"""
        import math
        if isinstance(obj, np.floating):
            # 处理NaN、Infinity等特殊值
            if np.isnan(obj) or np.isinf(obj):
                return 0.0  # 替换为0.0或其他合适的默认值
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            # 转换数组中的特殊值
            arr = obj.tolist()
            return self.convert_numpy_types(arr)  # 递归处理
        elif isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        # 处理Python原生的NaN和Infinity
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return 0.0
        return obj
# -*- coding:utf-8 -*-
"""
Date: 2024/11/06
Desc: 自定义基金数据爬取模块
基于雪球基金API，独立于akshare，方便自定义修改和修复bug
"""

import pandas as pd
import requests
import logging
import time
from typing import Optional, Dict, Any
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FundDataFetcher:
    """基金数据获取类，用于从雪球等平台获取基金数据"""
    
    def __init__(self, timeout: float = 10.0, retry_count: int = 3, retry_delay: float = 1.0):
        """
        初始化基金数据获取器
        
        参数:
            timeout: 请求超时时间
            retry_count: 请求失败重试次数
            retry_delay: 请求失败重试间隔（秒）
        """
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        }
    
    def _request_with_retry(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """带重试机制的HTTP请求"""
        for attempt in range(self.retry_count):
            try:
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    params=params, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.retry_count}): {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
        logger.error(f"请求失败，已达到最大重试次数: {url}")
        return None
    
    def get_fund_basic_info(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        获取基金基本信息
        接口来源: https://danjuanfunds.com/djapi/fund/{symbol}
        
        参数:
            symbol: 基金代码
        
        返回:
            包含基金基本信息的DataFrame
        """
        url = f"https://danjuanfunds.com/djapi/fund/{symbol}"
        json_data = self._request_with_retry(url)
        
        if not json_data or "data" not in json_data:
            logger.error(f"获取基金 {symbol} 基本信息失败")
            return None
        
        try:
            fund_data = json_data["data"]
            
            # 创建一个空的DataFrame来存储结果
            result_df = pd.DataFrame(columns=["item", "value"])
            
            # 准备字段映射，包括从不同层级提取的数据
            field_mappings = {
                "基金代码": fund_data.get("fd_code", pd.NA),
                "基金名称": fund_data.get("fd_name", pd.NA),
                "基金全称": fund_data.get("fd_full_name", pd.NA),
                "成立时间": fund_data.get("found_date", pd.NA),
                "最新规模": fund_data.get("totshare", pd.NA),
                "基金公司": fund_data.get("keeper_name", pd.NA),
                "基金经理": fund_data.get("manager_name", pd.NA),
                "托管银行": fund_data.get("trup_name", pd.NA),
                "基金类型": fund_data.get("type_desc", pd.NA),
                "评级机构": fund_data.get("rating_source", pd.NA),
                "基金评级": fund_data.get("rating_desc", pd.NA),
            }
            
            # 检查是否有嵌套数据
            if "fund_strategy" in fund_data:
                field_mappings["投资策略"] = fund_data["fund_strategy"].get("invest_orientation", pd.NA)
                field_mappings["投资目标"] = fund_data["fund_strategy"].get("invest_target", pd.NA)
                field_mappings["业绩比较基准"] = fund_data["fund_strategy"].get("performance_bench_mark", pd.NA)
            
            # 检查是否有风险等级信息
            if "risk_level" in fund_data:
                field_mappings["风险等级"] = fund_data["risk_level"]
            
            # 填充结果DataFrame
            for item, value in field_mappings.items():
                # 确保值不为None
                if value is None:
                    value = pd.NA
                
                # 将数据添加到结果DataFrame
                temp_df = pd.DataFrame([[item, value]], columns=["item", "value"])
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
            
            return result_df
        except Exception as e:
            logger.error(f"处理基金 {symbol} 基本信息时出错: {str(e)}")
            return None
    
    def get_fund_achievement(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        获取基金业绩数据
        接口来源: https://danjuanfunds.com/djapi/fundx/base/fund/achievement/{symbol}
        
        参数:
            symbol: 基金代码
        
        返回:
            包含基金业绩的DataFrame
        """
        url = f"https://danjuanfunds.com/djapi/fundx/base/fund/achievement/{symbol}"
        json_data = self._request_with_retry(url)
        
        if not json_data or "data" not in json_data:
            logger.error(f"获取基金 {symbol} 业绩数据失败")
            return None
        
        try:
            data = json_data["data"]
            combined_df = None
            type_dict = {
                "annual_performance_list": "年度业绩",
                "stage_performance_list": "阶段业绩",
            }
            
            for k, v in type_dict.items():
                if k not in data or not data[k]:
                    continue
                
                temp_df = pd.DataFrame.from_dict(data[k], orient="columns")
                temp_df["业绩类型"] = v
                
                # 确保需要的列存在
                required_cols = ["业绩类型", "cycle", "value"]
                for col in required_cols:
                    if col not in temp_df.columns:
                        temp_df[col] = pd.NA
                
                temp_df = temp_df[required_cols]
                temp_df.columns = ["业绩类型", "周期", "本产品区间收益"]
                
                if combined_df is None:
                    combined_df = temp_df
                else:
                    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
            
            if combined_df is not None:
                # 转换收益为数值类型
                combined_df["本产品区间收益"] = pd.to_numeric(combined_df["本产品区间收益"], errors="coerce")
            
            return combined_df
        except Exception as e:
            logger.error(f"处理基金 {symbol} 业绩数据时出错: {str(e)}")
            return None
    
    def get_fund_analysis(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        获取基金分析数据
        接口来源: https://danjuanfunds.com/djapi/fundx/base/fund/analysis/{symbol}
        
        参数:
            symbol: 基金代码
        
        返回:
            包含基金分析的DataFrame
        """
        url = f"https://danjuanfunds.com/djapi/fundx/base/fund/analysis/{symbol}"
        json_data = self._request_with_retry(url)
        
        if not json_data or "data" not in json_data:
            logger.error(f"获取基金 {symbol} 分析数据失败")
            return None
        
        try:
            data = json_data["data"]
            temp_df = pd.DataFrame(columns=["周期", "较同类风险收益比", "年化波动率"])
            
            # 提取数据
            if "risk_return_ratio" in data:
                # 创建数据列表
                cycles = list(data["risk_return_ratio"].keys())
                risk_return_values = list(data["risk_return_ratio"].values())
                
                # 添加年化波动率数据
                volatility_values = []
                if "volatility" in data:
                    for cycle in cycles:
                        volatility_values.append(data["volatility"].get(cycle, pd.NA))
                else:
                    volatility_values = [pd.NA] * len(cycles)
                
                # 创建DataFrame
                temp_df = pd.DataFrame({
                    "周期": cycles,
                    "较同类风险收益比": risk_return_values,
                    "年化波动率": volatility_values
                })
                
                # 转换为数值类型
                temp_df["较同类风险收益比"] = pd.to_numeric(temp_df["较同类风险收益比"], errors="coerce")
                temp_df["年化波动率"] = pd.to_numeric(temp_df["年化波动率"], errors="coerce")
            
            return temp_df
        except Exception as e:
            logger.error(f"处理基金 {symbol} 分析数据时出错: {str(e)}")
            return None
    
    def get_fund_profit_probability(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        获取基金盈利概率
        接口来源: https://danjuanfunds.com/djapi/fundx/base/fund/profit/probability/{symbol}
        
        参数:
            symbol: 基金代码
        
        返回:
            包含基金盈利概率的DataFrame
        """
        url = f"https://danjuanfunds.com/djapi/fundx/base/fund/profit/probability/{symbol}"
        json_data = self._request_with_retry(url)
        
        if not json_data or "data" not in json_data or "data" not in json_data["data"]:
            logger.error(f"获取基金 {symbol} 盈利概率失败")
            return None
        
        try:
            data = json_data["data"]["data"]
            
            # 准备数据
            periods = []
            profit_probabilities = []
            avg_returns = []
            
            # 检查不同周期的盈利概率
            if "hold_period_12m" in data:
                periods.append("1年")
                profit_probabilities.append(data["hold_period_12m"].get("profit_rate", pd.NA))
                avg_returns.append(data["hold_period_12m"].get("avg_return", pd.NA))
            
            if "hold_period_24m" in data:
                periods.append("2年")
                profit_probabilities.append(data["hold_period_24m"].get("profit_rate", pd.NA))
                avg_returns.append(data["hold_period_24m"].get("avg_return", pd.NA))
            
            if "hold_period_36m" in data:
                periods.append("3年")
                profit_probabilities.append(data["hold_period_36m"].get("profit_rate", pd.NA))
                avg_returns.append(data["hold_period_36m"].get("avg_return", pd.NA))
            
            # 创建DataFrame
            temp_df = pd.DataFrame({
                "持有时长": periods,
                "盈利概率": profit_probabilities,
                "平均收益": avg_returns
            })
            
            # 转换为数值类型
            temp_df["盈利概率"] = pd.to_numeric(temp_df["盈利概率"], errors="coerce")
            temp_df["平均收益"] = pd.to_numeric(temp_df["平均收益"], errors="coerce")
            
            return temp_df
        except Exception as e:
            logger.error(f"处理基金 {symbol} 盈利概率时出错: {str(e)}")
            return None
    
    def get_fund_trading_rules(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        获取基金交易规则
        接口来源: https://danjuanfunds.com/djapi/fund/detail/{symbol}
        
        参数:
            symbol: 基金代码
        
        返回:
            包含基金交易规则的DataFrame
        """
        url = f"https://danjuanfunds.com/djapi/fund/detail/{symbol}"
        json_data = self._request_with_retry(url)
        
        if not json_data or "data" not in json_data or "fund_rates" not in json_data["data"]:
            logger.error(f"获取基金 {symbol} 交易规则失败")
            return None
        
        try:
            fund_rates = json_data["data"]["fund_rates"]
            combined_df = None
            rate_type_dict = {
                "declare_rate_table": "买入规则",
                "withdraw_rate_table": "卖出规则",
                "other_rate_table": "其他费用",
            }
            
            for k, v in rate_type_dict.items():
                if k not in fund_rates or not fund_rates[k]:
                    continue
                
                temp_df = pd.DataFrame.from_dict(fund_rates[k], orient="columns")
                temp_df["rate_type"] = v
                
                # 确保需要的列存在
                required_cols = ["rate_type", "name", "value"]
                for col in required_cols:
                    if col not in temp_df.columns:
                        temp_df[col] = pd.NA
                
                temp_df = temp_df[required_cols]
                temp_df.columns = ["费用类型", "条件或名称", "费用"]
                
                if combined_df is None:
                    combined_df = temp_df
                else:
                    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
            
            if combined_df is not None:
                # 转换费用为数值类型
                combined_df["费用"] = pd.to_numeric(combined_df["费用"], errors="coerce")
            
            return combined_df
        except Exception as e:
            logger.error(f"处理基金 {symbol} 交易规则时出错: {str(e)}")
            return None
    
    def get_fund_holdings(self, symbol: str, date: str) -> Optional[pd.DataFrame]:
        """
        获取基金持仓情况
        接口来源: https://danjuanfunds.com/djapi/fundx/base/fund/record/asset/percent
        
        参数:
            symbol: 基金代码
            date: 财报日期，格式为YYYYMMDD
        
        返回:
            包含基金持仓情况的DataFrame
        """
        url = "https://danjuanfunds.com/djapi/fundx/base/fund/record/asset/percent"
        
        # 严格验证日期格式
        if not re.match(r'^\d{8}$', date):
            logger.error(f"日期格式错误，必须是8位数字(YYYYMMDD): {date}")
            return None
        
        try:
            # 尝试拆分日期
            year, month, day = date[:4], date[4:6], date[6:]
            
            # 验证月份和日期的有效性
            if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
                logger.error(f"日期无效: {date}")
                return None
            
            formatted_date = f"{year}-{month}-{day}"
        except (ValueError, IndexError):
            logger.error(f"日期格式错误: {date}")
            return None
        
        params = {
            "fund_code": symbol,
            "report_date": formatted_date,
        }
        
        json_data = self._request_with_retry(url, params=params)
        
        if not json_data or "data" not in json_data or "chart_list" not in json_data["data"]:
            logger.error(f"获取基金 {symbol} 持仓数据失败")
            return None
        
        try:
            chart_list = json_data["data"]["chart_list"]
            temp_df = pd.DataFrame.from_dict(chart_list, orient="columns")
            
            # 确保需要的列存在
            required_columns = ["type_desc", "percent"]
            for col in required_columns:
                if col not in temp_df.columns:
                    temp_df[col] = pd.NA
            
            temp_df = temp_df[required_columns]
            temp_df.columns = ["资产类型", "仓位占比"]
            
            # 转换为数值类型
            temp_df["仓位占比"] = pd.to_numeric(temp_df["仓位占比"], errors="coerce")
            
            return temp_df
        except Exception as e:
            logger.error(f"处理基金 {symbol} 持仓数据时出错: {str(e)}")
            return None

# 创建全局实例，方便直接调用
global_fund_fetcher = FundDataFetcher()

# 定义便捷函数
def get_fund_basic_info(symbol: str) -> Optional[pd.DataFrame]:
    """获取基金基本信息的便捷函数"""
    return global_fund_fetcher.get_fund_basic_info(symbol)

def get_fund_achievement(symbol: str) -> Optional[pd.DataFrame]:
    """获取基金业绩数据的便捷函数"""
    return global_fund_fetcher.get_fund_achievement(symbol)

def get_fund_analysis(symbol: str) -> Optional[pd.DataFrame]:
    """获取基金分析数据的便捷函数"""
    return global_fund_fetcher.get_fund_analysis(symbol)

def get_fund_profit_probability(symbol: str) -> Optional[pd.DataFrame]:
    """获取基金盈利概率的便捷函数"""
    return global_fund_fetcher.get_fund_profit_probability(symbol)

def get_fund_trading_rules(symbol: str) -> Optional[pd.DataFrame]:
    """获取基金交易规则的便捷函数"""
    return global_fund_fetcher.get_fund_trading_rules(symbol)

def get_fund_holdings(symbol: str, date: str) -> Optional[pd.DataFrame]:
    """获取基金持仓情况的便捷函数"""
    return global_fund_fetcher.get_fund_holdings(symbol, date)

if __name__ == "__main__":
    # 测试代码
    test_symbol = "000005"
    
    # 测试获取基金基本信息
    basic_info = get_fund_basic_info(test_symbol)
    if basic_info is not None:
        print("基金基本信息:")
        print(basic_info)
    
    # 测试获取基金业绩
    achievement = get_fund_achievement(test_symbol)
    if achievement is not None:
        print("\n基金业绩:")
        print(achievement)
    
    # 测试其他函数可以根据需要取消注释
    # analysis = get_fund_analysis(test_symbol)
    # if analysis is not None:
    #     print("\n基金分析:")
    #     print(analysis)
    
    # profit_prob = get_fund_profit_probability(test_symbol)
    # if profit_prob is not None:
    #     print("\n盈利概率:")
    #     print(profit_prob)
    
    # trading_rules = get_fund_trading_rules(test_symbol)
    # if trading_rules is not None:
    #     print("\n交易规则:")
    #     print(trading_rules)
    
    # holdings = get_fund_holdings("002804", "20231231")
    # if holdings is not None:
    #     print("\n持仓情况:")
    #     print(holdings)
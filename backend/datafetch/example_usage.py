# -*- coding:utf-8 -*-
"""
Date: 2024/11/06
Desc: 基金数据获取模块使用示例
展示如何使用自定义的fund_data_fetch模块获取和处理基金数据
"""

import sys
import os
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义的基金数据获取模块
from datafetch.fund_data_fetch import (
    get_fund_basic_info,
    get_fund_achievement,
    get_fund_analysis,
    get_fund_profit_probability,
    get_fund_trading_rules,
    get_fund_holdings,
    FundDataFetcher
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FundDataProcessor:
    """基金数据处理类，用于处理和使用从API获取的基金数据"""
    
    def __init__(self):
        # 创建自定义配置的基金数据获取器实例
        self.fetcher = FundDataFetcher(
            timeout=15.0,  # 增加超时时间以应对网络不稳定情况
            retry_count=5,  # 增加重试次数
            retry_delay=1.5  # 增加重试间隔
        )
    
    def fetch_and_process_fund_basic_info(self, fund_code: str) -> dict:
        """
        获取并处理基金基本信息
        包含健壮的数据处理逻辑，解决可能出现的KeyError问题
        """
        logger.info(f"获取基金 {fund_code} 的基本信息...")
        
        # 获取基金基本信息
        df = self.fetcher.get_fund_basic_info(fund_code)
        
        if df is None or df.empty:
            logger.warning(f"未获取到基金 {fund_code} 的基本信息")
            return {}
        
        try:
            # 将DataFrame转换为字典便于处理
            info_dict = {}
            for _, row in df.iterrows():
                item = row.get('item')
                value = row.get('value')
                if pd.notna(item) and pd.notna(value):
                    info_dict[item] = value
            
            # 处理关键数据，确保即使某些字段缺失也不会导致错误
            processed_info = {
                'fund_code': fund_code,
                'fund_name': info_dict.get('基金名称', ''),
                'full_name': info_dict.get('基金全称', ''),
                'establish_date': self._parse_date(info_dict.get('成立时间', '')),
                'latest_scale': self._parse_float(info_dict.get('最新规模', 0)),
                'company': info_dict.get('基金公司', ''),
                'manager': info_dict.get('基金经理', ''),
                'custodian_bank': info_dict.get('托管银行', ''),
                'fund_type': info_dict.get('基金类型', ''),
                'rating_agency': info_dict.get('评级机构', ''),
                'rating': info_dict.get('基金评级', ''),
                'investment_strategy': info_dict.get('投资策略', ''),
                'investment_goal': info_dict.get('投资目标', ''),
                'benchmark': info_dict.get('业绩比较基准', ''),
                # 确保即使'最新规模'字段不存在也不会抛出KeyError
                'has_latest_scale': '最新规模' in info_dict
            }
            
            logger.info(f"成功获取并处理基金 {fund_code} 的基本信息")
            return processed_info
            
        except Exception as e:
            logger.error(f"处理基金 {fund_code} 基本信息时出错: {str(e)}")
            # 返回空字典而不是抛出异常
            return {}
    
    def batch_fetch_fund_info(self, fund_codes: list, delay: float = 0.5) -> list:
        """
        批量获取基金信息
        
        参数:
            fund_codes: 基金代码列表
            delay: 每只基金之间的延迟时间（秒），避免请求过于频繁
        
        返回:
            处理后的基金信息列表
        """
        results = []
        total = len(fund_codes)
        
        for i, fund_code in enumerate(fund_codes, 1):
            logger.info(f"处理基金 {fund_code} ({i}/{total})")
            fund_info = self.fetch_and_process_fund_basic_info(fund_code)
            if fund_info:
                results.append(fund_info)
            
            # 如果不是最后一个基金，添加延迟
            if i < total:
                time.sleep(delay)
        
        logger.info(f"批量处理完成，成功获取 {len(results)}/{total} 只基金的信息")
        return results
    
    def _parse_date(self, date_str: str) -> str:
        """解析日期字符串，处理各种可能的格式"""
        if not date_str or pd.isna(date_str):
            return ''
        
        try:
            # 尝试多种日期格式
            date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d', '%Y年%m月%d日']
            
            for fmt in date_formats:
                try:
                    # 解析成功后返回标准格式
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            # 如果所有格式都解析失败，返回原始字符串
            logger.warning(f"无法解析日期格式: {date_str}")
            return date_str
        except Exception as e:
            logger.error(f"解析日期时出错: {str(e)}")
            return ''
    
    def _parse_float(self, float_str: str) -> float:
        """解析浮点数，处理可能的非数字情况"""
        if float_str is None or pd.isna(float_str):
            return 0.0
        
        try:
            # 尝试直接转换
            if isinstance(float_str, (int, float)):
                return float(float_str)
            
            # 处理字符串，去除可能的非数字字符
            if isinstance(float_str, str):
                # 去除千位分隔符和单位
                clean_str = float_str.replace(',', '').strip()
                # 去除常见单位
                for unit in ['亿元', '万元', '亿', '万']:
                    if clean_str.endswith(unit):
                        clean_str = clean_str[:-len(unit)].strip()
                        # 转换单位（如果需要）
                        if unit in ['亿元', '亿']:
                            multiplier = 100000000
                        elif unit in ['万元', '万']:
                            multiplier = 10000
                        else:
                            multiplier = 1
                        return float(clean_str) * multiplier
                
                return float(clean_str)
        except Exception as e:
            logger.error(f"解析浮点数时出错: {str(e)}")
            return 0.0

# 修复sys.modules中的pandas引用，避免在某些环境中出现导入问题
import pandas as pd
import time

def example_basic_usage():
    """基础使用示例"""
    print("===== 基础使用示例 =====")
    
    # 使用便捷函数直接获取数据
    fund_code = "000005"
    
    # 获取基金基本信息
    print(f"\n1. 获取基金 {fund_code} 的基本信息:")
    basic_info = get_fund_basic_info(fund_code)
    if basic_info is not None:
        print(basic_info)
    else:
        print("未能获取到基金基本信息")
    
    # 获取基金业绩
    print(f"\n2. 获取基金 {fund_code} 的业绩数据:")
    achievement = get_fund_achievement(fund_code)
    if achievement is not None:
        print(achievement)
    else:
        print("未能获取到基金业绩数据")
    
    # 获取基金持仓（需要指定日期）
    print("\n3. 获取基金 002804 在2023年底的持仓情况:")
    holdings = get_fund_holdings("002804", "20231231")
    if holdings is not None:
        print(holdings)
    else:
        print("未能获取到基金持仓数据")


def example_advanced_usage():
    """高级使用示例"""
    print("\n===== 高级使用示例 =====")
    
    # 创建处理器实例
    processor = FundDataProcessor()
    
    # 单只基金数据处理（包含KeyError防护）
    fund_code = "000005"
    print(f"\n1. 处理基金 {fund_code} 的基本信息（包含错误处理）:")
    fund_info = processor.fetch_and_process_fund_basic_info(fund_code)
    print(f"处理结果: {fund_info}")
    print(f"是否包含'最新规模'字段: {fund_info.get('has_latest_scale', False)}")
    
    # 批量处理基金数据
    print("\n2. 批量处理多只基金数据:")
    fund_codes = ["000005", "000001", "000002", "000003", "000004"]
    batch_results = processor.batch_fetch_fund_info(fund_codes, delay=0.3)
    
    print(f"\n批量处理结果摘要:")
    for fund in batch_results:
        print(f"基金代码: {fund.get('fund_code')}, 基金名称: {fund.get('fund_name')}, 最新规模: {fund.get('latest_scale')}")


def example_custom_config():
    """自定义配置示例"""
    print("\n===== 自定义配置示例 =====")
    
    # 创建自定义配置的获取器
    custom_fetcher = FundDataFetcher(
        timeout=20.0,  # 更长的超时时间
        retry_count=3,  # 更少的重试次数
        retry_delay=2.0  # 更长的重试间隔
    )
    
    fund_code = "000005"
    print(f"\n使用自定义配置获取基金 {fund_code} 的基本信息:")
    custom_info = custom_fetcher.get_fund_basic_info(fund_code)
    if custom_info is not None:
        print(custom_info)
    else:
        print("未能获取到基金基本信息")

if __name__ == "__main__":
    # 运行所有示例
    try:
        example_basic_usage()
        example_advanced_usage()
        example_custom_config()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"示例运行出错: {str(e)}")
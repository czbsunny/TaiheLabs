import os
import sys
import time
import logging
import os
import concurrent.futures
from datetime import datetime
from functools import partial

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak
import pandas as pd

# 导入数据库和模型
from database.init_db import SessionLocal, init_db
from models.fund import Fund
from models.fund_nav_history import FundNavHistory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("init_fund_nav.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_fund_data(fund_code, force_update):
    """获取单只基金的净值数据
    
    参数:
        fund_code: 基金代码
        force_update: 是否强制更新数据
        
    返回:
        tuple: (是否成功, 处理的记录数, 错误信息)
    """
    try:
        # 1. 为每个基金创建独立的数据库会话进行检查
        check_db = SessionLocal()
        try:
            # 检查该基金是否已经有净值数据，且不是强制更新模式
            if not force_update:
                existing_records = check_db.query(FundNavHistory).filter(FundNavHistory.fund_code == fund_code).count()
                if existing_records > 0:
                    # 检查最新记录的日期，判断是否需要更新
                    latest_record = check_db.query(FundNavHistory).filter(
                        FundNavHistory.fund_code == fund_code
                    ).order_by(FundNavHistory.date.desc()).first()
                    
                    # 如果最新记录是今天或者昨天的数据，认为数据已是最新
                    # 考虑到基金数据可能会有延迟，设置2天的缓冲期
                    logger.info(f"基金 {fund_code} 的净值数据已是最新（最新记录日期: {latest_record.date}），跳过...")
                    return True, 0, "跳过处理"
        finally:
            check_db.close()
        
        # 2. 并发获取三种数据
        nav_df = None
        accumulated_nav_df = None
        return_rate_df = None
        
        # 使用函数局部作用域存储结果
        def _get_nav_data():
            try:
                # 减少延时时间以提高效率，但保留一定延时避免API请求过于频繁
                time.sleep(0.2)  # 从0.5秒减少到0.2秒
                return ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
            except Exception as e:
                logger.warning(f"获取基金 {fund_code} 单位净值数据失败: {str(e)}")
                return None
        
        def _get_accumulated_nav_data():
            try:
                time.sleep(0.2)  # 减少延时
                return ak.fund_open_fund_info_em(symbol=fund_code, indicator="累计净值走势")
            except Exception as e:
                logger.warning(f"获取基金 {fund_code} 累计净值数据失败: {str(e)}")
                return None
        
        def _get_return_rate_data():
            try:
                time.sleep(0.2)  # 减少延时
                return ak.fund_open_fund_info_em(symbol=fund_code, indicator="累计收益率走势", period="成立来")
            except Exception as e:
                logger.warning(f"获取基金 {fund_code} 累计收益率数据失败: {str(e)}")
                return None
        
        # 使用线程池并发获取数据
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_nav = executor.submit(_get_nav_data)
            future_acc_nav = executor.submit(_get_accumulated_nav_data)
            future_return_rate = executor.submit(_get_return_rate_data)
            
            nav_df = future_nav.result()
            accumulated_nav_df = future_acc_nav.result()
            return_rate_df = future_return_rate.result()
        
        # 3. 处理数据（但不立即存储，返回数据供批量处理）
        processed_data = {
            'fund_code': fund_code,
            'nav_df': nav_df,
            'accumulated_nav_df': accumulated_nav_df,
            'return_rate_df': return_rate_df
        }
        
        return True, processed_data, "成功获取数据"
        
    except Exception as e:
        logger.error(f"处理基金 {fund_code} 时出错: {str(e)}")
        return False, 0, str(e)

def process_batch_data(db, batch_results):
    """批量处理并存储多个基金的数据
    
    参数:
        db: 数据库会话
        batch_results: 包含多个基金数据的列表
        
    返回:
        tuple: (成功数量, 失败数量, 跳过数量, 处理的记录总数)
    """
    success_count = 0
    failed_count = 0
    skipped_count = 0
    total_records = 0
    failed_funds = []
    
    try:
        # 验证数据库连接状态
        try:
            # 执行简单查询测试连接
            db.execute("SELECT 1")
        except Exception as conn_err:
            logger.error(f"数据库连接检查失败，重新创建连接: {str(conn_err)}")
            # 如果连接已断开，重新获取连接
            db.rollback()
            db.close()
            db = SessionLocal()
        
        for result in batch_results:
            if result is None:
                continue
                
            success, data, message = result
            # 安全地获取fund_code，避免NoneType错误
            if isinstance(data, dict) and 'fund_code' in data:
                fund_code = data['fund_code']
            elif isinstance(data, tuple) and len(data) >= 3 and isinstance(data[2], str) and data[2] == "跳过处理":
                # 处理跳过的情况
                fund_code = None
            else:
                fund_code = None
            
            if success:
                if isinstance(data, dict) and 'fund_code' in data:
                    try:
                        # 处理基金数据
                        processed_records = 0
                        
                        # 优先处理单位净值数据
                        if data['nav_df'] is not None and not data['nav_df'].empty:
                            processed_records += process_nav_data(db, data['fund_code'], data['nav_df'], is_nav=True)
                        
                        # 处理累计净值数据
                        if data['accumulated_nav_df'] is not None and not data['accumulated_nav_df'].empty:
                            processed_records += process_nav_data(db, data['fund_code'], data['accumulated_nav_df'], is_accumulated_nav=True)
                        
                        # 处理累计收益率数据
                        if data['return_rate_df'] is not None and not data['return_rate_df'].empty:
                            processed_records += process_nav_data(db, data['fund_code'], data['return_rate_df'], is_return_rate=True)
                        
                        if processed_records > 0:
                            logger.info(f"成功处理基金 {data['fund_code']} 的 {processed_records} 条净值数据")
                            success_count += 1
                            total_records += processed_records
                        else:
                            logger.warning(f"基金 {data['fund_code']} 没有可处理的净值数据")
                            failed_count += 1
                            failed_funds.append(data['fund_code'])
                    except Exception as fund_err:
                        logger.error(f"处理基金 {data['fund_code']} 数据时出错: {str(fund_err)}")
                        failed_count += 1
                        failed_funds.append(data['fund_code'])
                        # 出错时回滚当前基金的事务，但继续处理其他基金
                        try:
                            db.rollback()
                        except Exception as rb_err:
                            logger.error(f"回滚事务失败: {str(rb_err)}")
                else:
                    # 跳过处理的情况
                    skipped_count += 1
            else:
                if fund_code:
                    failed_count += 1
                    failed_funds.append(fund_code)
                    logger.warning(f"基金 {fund_code} 处理失败: {message}")
                else:
                    failed_count += 1
                    logger.warning(f"未知基金处理失败: {message}")
        
        # 提交当前批次的事务
        try:
            db.commit()
        except Exception as commit_err:
            logger.error(f"提交事务失败: {str(commit_err)}")
            # 尝试回滚
            try:
                db.rollback()
            except Exception as rb_err:
                logger.error(f"回滚事务失败: {str(rb_err)}")
            # 标记当前批次所有基金为失败
            failed_count = len(batch_results)
            success_count = 0
            skipped_count = 0
            failed_funds = [result[1]['fund_code'] if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict) and 'fund_code' in result[1] else f"未知基金_{i}" for i, result in enumerate(batch_results)]
        
    except Exception as e:
        logger.error(f"批量处理数据时出错: {str(e)}")
        try:
            db.rollback()
        except Exception as rb_err:
            logger.error(f"回滚事务失败: {str(rb_err)}")
        # 将当前批次的所有基金标记为失败
        for result in batch_results:
            if result and result[0]:
                fund_code = result[1]['fund_code'] if isinstance(result[1], dict) and 'fund_code' in result[1] else None
                if fund_code:
                    failed_count += 1
                    failed_funds.append(fund_code)
                    success_count = max(0, success_count - 1)
                    skipped_count = max(0, skipped_count - 1)
    
    return success_count, failed_count, skipped_count, total_records, failed_funds

def init_fund_nav_history(force_update=False):
    """初始化基金净值历史数据
    使用akshare获取基金的单位净值、累计净值和累计收益率数据，存储到数据库
    
    参数:
        force_update: 是否强制更新所有数据，即使已经存在数据
    """
    # 初始化数据库
    init_db()
    db = SessionLocal()
    
    try:
        # 记录开始时间
        start_time = time.time()
        logger.info("开始初始化基金净值历史数据...")
        
        if force_update:
            logger.info("强制更新模式已启用，将重新获取所有基金的净值数据")
        
        # 1. 获取所有已存在的基金代码、类型和名称
        logger.info("正在获取数据库中已存在的基金代码、类型和名称...")
        fund_data = db.query(Fund.fund_code, Fund.fund_type, Fund.company).all()
        total_funds = len(fund_data)
        
        if total_funds == 0:
            logger.warning("数据库中没有基金记录，请先运行init_fund.py初始化基金基础信息")
            return
        
        # 过滤掉不需要获取净值的基金
        # 需要跳过的基金: 货币型-普通货币、货币型-浮动净值、空类型和name为空的基金
        filtered_fund_codes = []
        skipped_types_count = 0
        skipped_empty_count = 0
        skipped_company_empty_count = 0
        
        for fund_code, fund_type, company in fund_data:
            if fund_type == '货币型-普通货币' or fund_type == '货币型-浮动净值':
                skipped_types_count += 1
            elif fund_type == '':
                skipped_empty_count += 1
            elif not company or company.strip() == '':
                skipped_company_empty_count += 1
            else:
                filtered_fund_codes.append(fund_code)
        
        filtered_total = len(filtered_fund_codes)
        
        logger.info(f"成功获取{total_funds}条基金记录")
        logger.info(f"过滤掉{skipped_types_count}条货币型基金，{skipped_empty_count}条空类型基金，{skipped_company_empty_count}条名称为空的基金")
        logger.info(f"实际需要处理的基金数量: {filtered_total}")
        
        if filtered_total == 0:
            logger.warning("没有需要处理的基金记录")
            return
        
        # 2. 批量处理基金数据，每批次20只基金
        # 批量大小调整考虑：
        # 1. 系统资源：当前系统配置能够支持更大批次的并发处理
        # 2. API调用限制：每批次20只基金，每只基金3个并发请求，总并发数60
        # 3. 数据库性能：采用了每批次独立会话，事务隔离性较好
        # 4. 错误恢复：单个基金出错不会影响批次内其他基金
        batch_size = 50
        success_count = 0
        failed_count = 0
        skipped_count = 0
        total_records = 0
        failed_funds = []
        
        # 分批处理基金
        for batch_start in range(0, filtered_total, batch_size):
            batch_end = min(batch_start + batch_size, filtered_total)
            batch_funds = filtered_fund_codes[batch_start:batch_end]
            
            # 记录进度
            progress = batch_end / filtered_total * 100
            logger.info(f"进度: {batch_end}/{filtered_total} ({progress:.2f}%)")
            logger.info(f"正在处理第{batch_start//batch_size + 1}批基金，共{len(batch_funds)}只基金")
            
            # 批量获取基金数据
            batch_results = []
            # 调整线程池大小为批量大小的1.5倍，确保并发效率
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size * 2) as executor:
                # 提交任务并保存future对象
                # 注意：不传递共享的数据库连接给线程池中的任务
                futures = {executor.submit(get_fund_data, fund_code, force_update): fund_code for fund_code in batch_funds}
                
                # 收集结果
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        fund_code = futures[future]
                        logger.error(f"获取基金 {fund_code} 数据时发生异常: {str(e)}")
                        batch_results.append((False, 0, str(e)))
            
            # 为每批次创建独立的数据库会话进行数据处理
            batch_db = SessionLocal()
            try:
                # 处理当前批次的数据
                batch_success, batch_failed, batch_skipped, batch_records, batch_failed_funds = process_batch_data(batch_db, batch_results)
            except Exception as e:
                logger.error(f"处理批次时发生数据库错误: {str(e)}")
                batch_success, batch_failed, batch_skipped, batch_records, batch_failed_funds = 0, len(batch_funds), 0, 0, batch_funds
            finally:
                batch_db.close()
            
            # 更新统计信息
            success_count += batch_success
            failed_count += batch_failed
            skipped_count += batch_skipped
            total_records += batch_records
            failed_funds.extend(batch_failed_funds)
            
            # 每处理完一批后，短暂休息以避免API请求过于频繁
            if batch_end < total_funds:
                # 优化：根据批次处理情况动态调整休息时间
                rest_time = 0.2 if batch_success > 0 else 0.5
                logger.info(f"批次处理完成，休息{rest_time}秒...")
                time.sleep(rest_time)
        
        # 最后提交剩余的事务（如果有的话）
        db.commit()
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        # 输出统计信息
        logger.info("基金净值历史数据初始化完成！")
        logger.info(f"总耗时: {duration:.2f}秒")
        logger.info(f"成功处理: {success_count}条基金")
        logger.info(f"处理失败: {failed_count}条基金")
        if not force_update:
            logger.info(f"跳过处理: {skipped_count}条基金")
        if failed_funds:
            logger.info(f"失败的基金代码: {', '.join(failed_funds)}")
        
    except Exception as e:
        logger.error(f"初始化基金净值数据时发生严重错误: {str(e)}")
        db.rollback()
    finally:
        db.close()

def process_nav_data(db, fund_code, data_df, is_nav=False, is_accumulated_nav=False, is_return_rate=False):
    """处理并存储基金净值相关数据（优化版本）
    
    参数:
        db: 数据库会话
        fund_code: 基金代码
        data_df: 包含净值数据的DataFrame
        is_nav: 是否为单位净值数据
        is_accumulated_nav: 是否为累计净值数据
        is_return_rate: 是否为累计收益率数据
        
    返回:
        处理的记录数
    """
    processed_count = 0
    
    try:
        # 确保DataFrame有正确的列名
        if is_nav:
            # 单位净值数据列名: 净值日期, 单位净值, 日增长率
            if '净值日期' in data_df.columns and ('单位净值' in data_df.columns or '单位净值走势' in data_df.columns):
                # 标准化列名
                if '单位净值走势' in data_df.columns:
                    data_df = data_df.rename(columns={'单位净值走势': '单位净值'})
                
                # 转换日期格式
                data_df['净值日期'] = pd.to_datetime(data_df['净值日期'], errors='coerce').dt.date
                
                # 过滤掉无效日期
                data_df = data_df.dropna(subset=['净值日期'])
                
                if not data_df.empty:
                    # 批量获取现有记录，避免逐行查询（性能优化关键）
                    dates = data_df['净值日期'].tolist()
                    existing_records = db.query(FundNavHistory).filter(
                        FundNavHistory.fund_code == fund_code,
                        FundNavHistory.date.in_(dates)
                    ).all()
                    
                    # 创建日期到记录的映射，便于快速查找
                    date_to_record = {(record.fund_code, record.date): record for record in existing_records}
                    
                    # 准备要添加的新记录
                    new_records = []
                    
                    # 使用向量化操作替代逐行迭代
                    for _, row in data_df.iterrows():
                        date_key = (fund_code, row['净值日期'])
                        
                        if date_key not in date_to_record:
                            # 创建新记录
                            nav_record = FundNavHistory(
                                fund_code=fund_code,
                                date=row['净值日期'],
                                created_at=datetime.now(),
                                updated_at=datetime.now()
                            )
                            new_records.append(nav_record)
                            date_to_record[date_key] = nav_record
                        else:
                            nav_record = date_to_record[date_key]
                        
                        # 更新单位净值
                        if '单位净值' in row and pd.notna(row['单位净值']):
                            nav_record.nav = float(row['单位净值'])
                        
                        # 更新日增长率
                        if '日增长率' in row and pd.notna(row['日增长率']):
                            # 处理百分比字符串
                            growth_rate = row['日增长率']
                            if isinstance(growth_rate, str) and '%' in growth_rate:
                                growth_rate = float(growth_rate.replace('%', ''))
                            nav_record.daily_growth_rate = float(growth_rate)
                        
                        processed_count += 1
                    
                    # 批量添加新记录（性能优化）
                    if new_records:
                        db.add_all(new_records)
                
        elif is_accumulated_nav:
            # 累计净值数据列名: 净值日期, 累计净值
            if '净值日期' in data_df.columns and ('累计净值' in data_df.columns or '累计净值走势' in data_df.columns):
                # 标准化列名
                if '累计净值走势' in data_df.columns:
                    data_df = data_df.rename(columns={'累计净值走势': '累计净值'})
                
                # 转换日期格式
                data_df['净值日期'] = pd.to_datetime(data_df['净值日期'], errors='coerce').dt.date
                
                # 过滤掉无效日期
                data_df = data_df.dropna(subset=['净值日期'])
                
                if not data_df.empty:
                    # 批量获取现有记录
                    dates = data_df['净值日期'].tolist()
                    existing_records = db.query(FundNavHistory).filter(
                        FundNavHistory.fund_code == fund_code,
                        FundNavHistory.date.in_(dates)
                    ).all()
                    
                    # 创建日期到记录的映射
                    date_to_record = {(record.fund_code, record.date): record for record in existing_records}
                    
                    # 准备要添加的新记录
                    new_records = []
                    
                    # 处理数据
                    for _, row in data_df.iterrows():
                        date_key = (fund_code, row['净值日期'])
                        
                        if date_key not in date_to_record:
                            # 创建新记录
                            nav_record = FundNavHistory(
                                fund_code=fund_code,
                                date=row['净值日期'],
                                created_at=datetime.now(),
                                updated_at=datetime.now()
                            )
                            new_records.append(nav_record)
                            date_to_record[date_key] = nav_record
                        else:
                            nav_record = date_to_record[date_key]
                        
                        # 更新累计净值
                        if '累计净值' in row and pd.notna(row['累计净值']):
                            nav_record.accumulated_nav = float(row['累计净值'])
                        
                        processed_count += 1
                    
                    # 批量添加新记录
                    if new_records:
                        db.add_all(new_records)
                
        elif is_return_rate:
            # 累计收益率数据列名: 日期, 累计收益率
            if '日期' in data_df.columns and ('累计收益率' in data_df.columns or '累计收益率走势' in data_df.columns):
                # 标准化列名
                if '累计收益率走势' in data_df.columns:
                    data_df = data_df.rename(columns={'累计收益率走势': '累计收益率'})
                
                # 转换日期格式
                data_df['日期'] = pd.to_datetime(data_df['日期'], errors='coerce').dt.date
                
                # 过滤掉无效日期
                data_df = data_df.dropna(subset=['日期'])
                
                if not data_df.empty:
                    # 批量获取现有记录
                    dates = data_df['日期'].tolist()
                    existing_records = db.query(FundNavHistory).filter(
                        FundNavHistory.fund_code == fund_code,
                        FundNavHistory.date.in_(dates)
                    ).all()
                    
                    # 创建日期到记录的映射
                    date_to_record = {(record.fund_code, record.date): record for record in existing_records}
                    
                    # 准备要添加的新记录
                    new_records = []
                    
                    # 处理数据
                    for _, row in data_df.iterrows():
                        date_key = (fund_code, row['日期'])
                        
                        if date_key not in date_to_record:
                            # 创建新记录
                            nav_record = FundNavHistory(
                                fund_code=fund_code,
                                date=row['日期'],
                                created_at=datetime.now(),
                                updated_at=datetime.now()
                            )
                            new_records.append(nav_record)
                            date_to_record[date_key] = nav_record
                        else:
                            nav_record = date_to_record[date_key]
                        
                        # 更新累计收益率
                        if '累计收益率' in row and pd.notna(row['累计收益率']):
                            # 处理百分比字符串
                            return_rate = row['累计收益率']
                            if isinstance(return_rate, str) and '%' in return_rate:
                                return_rate = float(return_rate.replace('%', ''))
                            nav_record.accumulated_return_rate = float(return_rate)
                        
                        processed_count += 1
                    
                    # 批量添加新记录
                    if new_records:
                        db.add_all(new_records)
                
    except Exception as e:
        logger.error(f"处理基金 {fund_code} 的净值数据时出错: {str(e)}")
        # 回滚当前事务
        try:
            db.rollback()
        except:
            pass
    
    return processed_count

if __name__ == "__main__":
    # 解析命令行参数，支持强制更新
    import argparse
    parser = argparse.ArgumentParser(description='初始化基金净值历史数据')
    parser.add_argument('--force', action='store_true', help='强制更新所有基金净值数据')
    args = parser.parse_args()
    
    init_fund_nav_history(force_update=True)
import os
import sys
import time
import logging
import argparse
from datetime import datetime
import concurrent.futures
from functools import partial

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义的基金数据获取模块
from datafetch.fund_data_fetch import get_fund_basic_info

# 导入数据库和模型
from database.init_db import SessionLocal, init_db
from models.fund import Fund

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("init_fund.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_single_fund(row, force_update=False):
    """处理单只基金的数据
    
    参数:
        row: 基金的基本信息行
        force_update: 是否强制更新已存在的基金信息
        
    返回:
        tuple: (是否成功, 基金代码, 数据/错误信息)
    """
    fund_code = row['基金代码']
    fund_type = row['基金类型']
    
    # 每个线程创建自己的数据库会话
    thread_db = SessionLocal()
    result = None
    
    try:
        try:
            # 获取基金详细信息
            try:
                # 注意：频繁调用接口可能会被限制，这里添加延时
                time.sleep(0.5)  # 减少延时以提高并发效率
                
                fund_info_df = get_fund_basic_info(symbol=fund_code)
                logger.debug(f"成功获取基金 {fund_code} 的详细信息，数据框形状: {fund_info_df.shape if fund_info_df is not None else 'None'}")
            except Exception as e:
                logger.warning(f"获取基金 {fund_code} 详细信息失败: {str(e)}")
                # 继续使用基本信息，详细信息设为None
                fund_info_df = None

            # 检查基金是否已存在
            existing_fund = thread_db.query(Fund).filter(Fund.fund_code == fund_code).first()
            
            if existing_fund:
                if force_update:
                    logger.debug(f"强制更新基金: {fund_code}")
                    # 更新基金类型和更新时间
                    existing_fund.fund_type = fund_type
                    existing_fund.updated_at = datetime.now()
                    
                    # 如果有详细信息，则更新详细信息
                    if fund_info_df is not None and not fund_info_df.empty:
                        update_fund_details(existing_fund, fund_info_df)
                    else:
                        logger.warning(f"跳过基金 {fund_code} 的详细信息更新，因为fund_info_df无效")
                    
                    # 在当前线程中提交更新
                    thread_db.commit()
                    result = (True, fund_code, "updated")
                else:
                    logger.debug(f"基金 {fund_code} 已存在，跳过处理")
                    result = (False, fund_code, "基金已存在")
            else:
                # 创建新基金对象
                fund = Fund(
                    fund_code=fund_code,
                    fund_type=fund_type,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )

                # 如果有详细信息，则更新详细信息
                if fund_info_df is not None and not fund_info_df.empty:
                    update_fund_details(fund, fund_info_df)
                else:
                    logger.warning(f"跳过基金 {fund_code} 的详细信息更新，因为fund_info_df无效")

                # 添加到当前线程的会话中并提交
                thread_db.add(fund)
                thread_db.commit()
                logger.debug(f"已处理基金: {fund_code}")
                result = (True, fund_code, "added")
                
        except Exception as e:
            # 发生异常时回滚事务
            thread_db.rollback()
            logger.error(f"处理基金 {fund_code} 时出错: {str(e)}")
            result = (False, fund_code, str(e))
        
    finally:
        # 确保关闭会话
        thread_db.close()
        
    return result

def init_fund_data(force_update=False):
    """初始化基金基础信息
    使用自定义datafetch模块获取所有基金代码，然后更新完整的基金基础信息
    
    参数:
        force_update: 是否强制更新已存在的基金信息
    """
    # 初始化数据库
    init_db()
    db = SessionLocal()
    
    try:
        # 记录开始时间
        start_time = time.time()
        logger.info("开始初始化基金基础信息...")
        if force_update:
            logger.info("模式: 强制更新所有基金信息")
        else:
            logger.info("模式: 只添加新基金")
        
        # 1. 获取所有基金代码和基本信息
        logger.info("正在获取所有基金代码...")
        try:
            # 这里暂时保留akshare获取基金列表的功能
            import akshare as ak
            fund_name_em_df = ak.fund_name_em()
            logger.info(f"成功获取{len(fund_name_em_df)}条基金基本信息")
        except Exception as e:
            logger.error(f"获取基金代码失败: {str(e)}")
            return
        
        # 2. 创建或更新基金基础信息
        # 2.1 先一次性获取数据库中所有已存在的基金代码，优化性能
        logger.info("正在加载数据库中已存在的基金代码...")
        existing_fund_codes = set(
            [fund.fund_code for fund in db.query(Fund.fund_code).all()]
        )
        logger.info(f"数据库中已存在 {len(existing_fund_codes)} 条基金记录")
        
        # 2.2 确定要处理的基金
        # 强制更新模式下处理所有基金
        # 非强制更新模式下，处理所有基金但只添加不存在的基金（不更新已存在的基金）
        process_funds_df = fund_name_em_df
        total_process_funds = len(process_funds_df)
        
        if force_update:
            logger.info(f"强制更新模式：将处理所有 {total_process_funds} 条基金记录")
            logger.info(f"  - 对于已存在的 {len(existing_fund_codes)} 条基金记录：将进行更新操作")
            logger.info(f"  - 对于不存在的 {total_process_funds - len(existing_fund_codes)} 条基金记录：将进行新增操作")
        else:
            # 计算需要新增的基金数量
            new_funds_count = len(fund_name_em_df[~fund_name_em_df['基金代码'].isin(existing_fund_codes)])
            logger.info(f"标准模式：将处理所有 {total_process_funds} 条基金记录")
            logger.info(f"  - 对于已存在的 {len(existing_fund_codes)} 条基金记录：将跳过处理")
            logger.info(f"  - 对于不存在的 {new_funds_count} 条基金记录：将进行新增操作")
            if new_funds_count == 0:
                logger.info("没有新基金需要添加")
        
        # 2.3 分批处理基金数据，每批次100只基金
        batch_size = 100
        processed_count = 0
        success_count = 0
        failed_count = 0
        updated_count = 0
        skipped_count = 0
        failed_funds = []
        
        for batch_start in range(0, total_process_funds, batch_size):
            batch_end = min(batch_start + batch_size, total_process_funds)
            batch_funds = process_funds_df.iloc[batch_start:batch_end]
            batch_count = len(batch_funds)
            
            # 记录进度
            progress = batch_end / total_process_funds * 100
            logger.info(f"进度: {batch_end}/{total_process_funds} ({progress:.2f}%)")
            logger.info(f"正在处理第{batch_start//batch_size + 1}批基金，共{batch_count}只基金")
            
            # 使用线程池并发处理基金数据
            batch_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(20, batch_count)) as executor:  # 限制最大线程数为20
                # 提交任务并保存future对象 - 不再传递数据库会话
                process_func = partial(process_single_fund, force_update=force_update)
                future_to_fund = {executor.submit(process_func, row): row['基金代码'] for _, row in batch_funds.iterrows()}
                
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_fund):
                    fund_code = future_to_fund[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"获取基金 {fund_code} 数据时发生异常: {str(e)}")
                        batch_results.append((False, fund_code, str(e)))
            
            # 处理批次结果 - 不需要数据库操作，因为每个线程已经处理了自己的会话
            batch_success = 0
            batch_failed = 0
            batch_updated = 0
            batch_skipped = 0
            
            for success, fund_code, data in batch_results:
                if success:
                    if data == "updated":
                        batch_updated += 1
                    elif data == "added":
                        batch_success += 1
                else:
                    if data == "基金已存在":
                        batch_skipped += 1
                    else:
                        batch_failed += 1
                        failed_funds.append(fund_code)
                        logger.warning(f"基金 {fund_code} 处理失败: {data}")
            
            success_count += batch_success
            failed_count += batch_failed
            updated_count += batch_updated
            skipped_count += batch_skipped
            processed_count += batch_count
            
            logger.info(f"批次处理完成：新增 {batch_success} 只，更新 {batch_updated} 只，跳过 {batch_skipped} 只，失败 {batch_failed} 只")
            
            # 批次间短暂休息，避免API请求过于频繁
            if batch_end < total_process_funds:
                logger.info(f"批次处理完成，休息2秒...")
                time.sleep(0.5)
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        # 输出统计信息
        logger.info("基金基础信息初始化完成！")
        logger.info(f"总耗时: {duration:.2f}秒")
        logger.info(f"成功处理: {success_count}条")
        if force_update:
            logger.info(f"成功更新: {updated_count}条")
        logger.info(f"处理失败: {failed_count}条")
        if skipped_count > 0:
            logger.info(f"跳过处理: {skipped_count}条")
        if failed_funds:
            logger.info(f"失败的基金代码: {', '.join(failed_funds[:50])}{'...' if len(failed_funds) > 50 else ''}")
        
    except Exception as e:
        logger.error(f"初始化基金数据时发生严重错误: {str(e)}")
    finally:
        db.close()

def update_fund_details(fund, fund_info_df):
    """更新基金的详细信息
    
    参数:
        fund: Fund对象
        fund_info_df: 包含基金详细信息的DataFrame
    """
    try:
        import pandas as pd
        # 将DataFrame转换为字典便于处理
        info_dict = fund_info_df.set_index('item').to_dict()['value']
        
        # 更新基金详细信息
        updated_fields = []
        if '基金名称' in info_dict:
            fund.name = info_dict['基金名称']
        if '基金全称' in info_dict:
            fund.full_name = info_dict['基金全称']
        # 使用get方法安全地获取'最新规模'，避免KeyError
        latest_scale = info_dict.get('最新规模')
        if latest_scale is not None and pd.notna(latest_scale):
            fund.latest_scale = latest_scale
        if '成立时间' in info_dict:
            try:
                fund.establish_date = datetime.strptime(info_dict['成立时间'], '%Y-%m-%d').date()
                updated_fields.append(f"establish_date={info_dict['成立时间']}")
            except ValueError:
                logger.warning(f"基金 {fund.fund_code} 的成立时间格式不正确: {info_dict['成立时间']}")
        if '基金公司' in info_dict:
            fund.company = info_dict['基金公司']
        if '基金经理' in info_dict:
            fund.manager = info_dict['基金经理']
        if '托管银行' in info_dict:
            fund.custodian_bank = info_dict['托管银行']
        if '评级机构' in info_dict:
            fund.rating_agency = info_dict['评级机构']
        if '基金评级' in info_dict:
            fund.rating = info_dict['基金评级']
        if '投资策略' in info_dict:
            fund.investment_strategy = info_dict['投资策略']
        if '投资目标' in info_dict:
            fund.investment_goal = info_dict['投资目标']
        if '业绩比较基准' in info_dict:
            fund.benchmark = info_dict['业绩比较基准']

        if '风险等级' in info_dict:
            fund.risk_level = info_dict['风险等级']
        # 更新最后修改时间
        fund.updated_at = datetime.now()
        
    except Exception as e:
        logger.error(f"更新基金 {fund.fund_code} 详细信息时出错: {str(e)}")

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='初始化或更新基金基础信息')
    parser.add_argument('--force-update', action='store_true', help='强制更新已存在的基金信息')
    args = parser.parse_args()
    
    init_fund_data(force_update=True)
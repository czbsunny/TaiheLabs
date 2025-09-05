from typing import Optional, Dict, Any, List, Tuple
import logging
import time
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from core.fund_processor import FundProcessor
from core.fund_data_source import FundDataSource, MockFundDataSource
from models.fund import Fund

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FundSyncService:
    """基金数据同步服务
    负责从各种数据源同步基金信息到本地数据库
    """
    
    def __init__(self, db: Session, data_sources: Optional[List[FundDataSource]] = None):
        """初始化基金同步服务
        
        参数:
            db: 数据库会话
            data_sources: 基金数据源列表，可选，如果不提供则使用模拟数据源
        """
        self.db = db
        self.fund_processor = FundProcessor(db)
        
        # 如果没有提供数据源，则使用模拟数据源
        self.data_sources = data_sources or [MockFundDataSource()]
        
        # 记录同步状态
        self.sync_status = {
            'last_synced': None,
            'current_sync': False,
            'success_count': 0,
            'fail_count': 0,
            'failed_funds': []
        }
    
    def sync_single_fund(self, fund_code: str, force_update: bool = False) -> bool:
        """同步单个基金信息
        
        参数:
            fund_code: 基金代码
            force_update: 是否强制更新，即使数据已存在
        
        返回:
            同步结果，True表示成功，False表示失败
        """
        logger.info(f"开始同步基金 {fund_code}")
        
        try:
            # 检查基金是否已存在且不是强制更新
            existing_fund = self.fund_processor.get_fund_info(fund_code)
            if existing_fund and not force_update:
                logger.info(f"基金 {fund_code} 已存在且无需强制更新，跳过同步")
                return True
            
            # 从第一个可用的数据源获取数据
            fund_basic_info = None
            fund_detail_info = None
            
            for data_source in self.data_sources:
                try:
                    fund_basic_info = data_source.get_fund_basic_info(fund_code)
                    if fund_basic_info:
                        fund_detail_info = data_source.get_fund_detail_info(fund_code)
                        break  # 成功获取数据，跳出循环
                except Exception as e:
                    logger.error(f"从数据源获取基金 {fund_code} 信息失败: {str(e)}")
                    continue
            
            if not fund_basic_info:
                logger.error(f"所有数据源均无法获取基金 {fund_code} 的基本信息")
                return False
            
            # 合并基本信息和详细信息
            merged_data = {**fund_basic_info}
            if fund_detail_info:
                merged_data.update(fund_detail_info)
            
            # 创建或更新完整的基金信息
            fund_result = self.fund_processor.create_or_update_fund(
                fund_code=fund_code,
                fund_data=merged_data
            )
            
            if not fund_result:
                logger.error(f"创建或更新基金 {fund_code} 信息失败")
                return False
            
            logger.info(f"基金 {fund_code} 同步成功")
            return True
        except Exception as e:
            logger.error(f"同步基金 {fund_code} 时发生异常: {str(e)}")
            return False
    
    def sync_multiple_funds(self, fund_codes: List[str], force_update: bool = False) -> Dict[str, Any]:
        """同步多个基金信息
        
        参数:
            fund_codes: 基金代码列表
            force_update: 是否强制更新，即使数据已存在
        
        返回:
            同步结果统计
        """
        logger.info(f"开始同步 {len(fund_codes)} 个基金")
        
        start_time = time.time()
        results = {
            'total': len(fund_codes),
            'success': 0,
            'fail': 0,
            'failed_funds': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration': None
        }
        
        for fund_code in fund_codes:
            success = self.sync_single_fund(fund_code, force_update)
            if success:
                results['success'] += 1
            else:
                results['fail'] += 1
                results['failed_funds'].append(fund_code)
            
            # 避免请求过于频繁
            time.sleep(0.5)
        
        # 计算同步时间
        end_time = time.time()
        results['end_time'] = datetime.now().isoformat()
        results['duration'] = round(end_time - start_time, 2)
        
        logger.info(f"同步完成: 成功 {results['success']} 个，失败 {results['fail']} 个，耗时 {results['duration']} 秒")
        return results
    
    def sync_all_funds(self, force_update: bool = False) -> Dict[str, Any]:
        """同步所有基金信息
        
        参数:
            force_update: 是否强制更新，即使数据已存在
        
        返回:
            同步结果统计
        """
        logger.info("开始同步所有基金信息")
        
        # 标记当前正在同步
        self.sync_status['current_sync'] = True
        self.sync_status['success_count'] = 0
        self.sync_status['fail_count'] = 0
        self.sync_status['failed_funds'] = []
        
        try:
            # 获取所有需要同步的基金代码
            # 在实际应用中，这里可能需要从配置或其他地方获取需要同步的基金列表
            # 这里仅作为示例，使用模拟的基金代码列表
            fund_codes_to_sync = ["005827", "161725"]  # 实际应用中应该替换为真实的基金代码列表
            
            # 对于已有基金，如果不是强制更新，只同步不存在的基金
            if not force_update:
                existing_funds = self.db.query(Fund.fund_code).all()
                existing_codes = [fund[0] for fund in existing_funds]
                # 只同步不存在的基金
                fund_codes_to_sync = [code for code in fund_codes_to_sync if code not in existing_codes]
                
            logger.info(f"发现 {len(fund_codes_to_sync)} 个基金需要同步")
            
            # 同步基金
            results = self.sync_multiple_funds(fund_codes_to_sync, force_update)
            
            # 更新同步状态
            self.sync_status['last_synced'] = datetime.now()
            self.sync_status['success_count'] = results['success']
            self.sync_status['fail_count'] = results['fail']
            self.sync_status['failed_funds'] = results['failed_funds']
            
            return results
        finally:
            # 无论成功失败，都标记同步结束
            self.sync_status['current_sync'] = False
    
    def sync_recently_updated_funds(self, days: int = 1) -> Dict[str, Any]:
        """同步最近更新的基金信息
        
        参数:
            days: 天数，同步最近几天更新的基金
        
        返回:
            同步结果统计
        """
        logger.info(f"开始同步最近 {days} 天更新的基金信息")
        
        # 在实际应用中，这里应该从数据源获取最近更新的基金列表
        # 这里仅作为示例，使用模拟数据
        # 实际实现时，可能需要调用数据源的特定API来获取最近更新的基金列表
        
        # 模拟获取最近更新的基金代码
        # 实际应用中应该替换为真实的逻辑
        recently_updated_funds = ["005827", "161725"]  # 示例基金代码
        
        # 同步这些基金
        results = self.sync_multiple_funds(recently_updated_funds, force_update=True)
        
        logger.info(f"同步最近更新的基金完成: 成功 {results['success']} 个，失败 {results['fail']} 个")
        return results
    
    def get_sync_status(self) -> Dict[str, Any]:
        """获取同步状态信息
        
        返回:
            同步状态字典
        """
        status = self.sync_status.copy()
        # 将datetime对象转换为字符串
        if status['last_synced']:
            status['last_synced'] = status['last_synced'].isoformat()
        return status
    
    def retry_failed_syncs(self) -> Dict[str, Any]:
        """重试失败的同步任务
        
        返回:
            重试结果统计
        """
        failed_funds = self.sync_status.get('failed_funds', [])
        
        if not failed_funds:
            logger.info("没有失败的同步任务需要重试")
            return {
                'total': 0,
                'success': 0,
                'fail': 0,
                'failed_funds': []
            }
        
        logger.info(f"开始重试 {len(failed_funds)} 个失败的同步任务")
        
        # 重试同步
        results = self.sync_multiple_funds(failed_funds, force_update=True)
        
        # 更新失败列表，只保留这次仍然失败的基金
        self.sync_status['failed_funds'] = results['failed_funds']
        
        logger.info(f"重试完成: 成功 {results['success']} 个，失败 {results['fail']} 个")
        return results
    
    def schedule_regular_sync(self, interval_hours: int = 24) -> None:
        """安排定期同步任务
        
        参数:
            interval_hours: 同步间隔（小时）
        
        注意：这是一个阻塞方法，会一直运行
        在实际应用中，应该在单独的线程或进程中运行
        """
        logger.info(f"启动定期同步任务，间隔 {interval_hours} 小时")
        
        try:
            while True:
                # 执行同步
                try:
                    self.sync_all_funds()
                except Exception as e:
                    logger.error(f"定期同步任务执行失败: {str(e)}")
                
                # 等待下一次同步
                wait_seconds = interval_hours * 3600
                logger.info(f"等待 {wait_seconds} 秒后进行下一次同步")
                time.sleep(wait_seconds)
        except KeyboardInterrupt:
            logger.info("定期同步任务被用户中断")
        except Exception as e:
            logger.error(f"定期同步任务发生异常: {str(e)}")


# 示例用法
if __name__ == "__main__":
    # 这里是示例用法，实际应用中应该通过依赖注入获取数据库会话
    from database.init_db import SessionLocal
    
    # 获取数据库会话
    db = SessionLocal()
    
    try:
        # 创建同步服务实例
        sync_service = FundSyncService(db)
        
        # 示例1：同步单个基金
        print("示例1：同步单个基金")
        result = sync_service.sync_single_fund("005827")
        print(f"同步结果: {'成功' if result else '失败'}")
        
        # 示例2：同步多个基金
        print("\n示例2：同步多个基金")
        results = sync_service.sync_multiple_funds(["005827", "161725"])
        print(f"同步结果统计: {results}")
        
        # 示例3：获取同步状态
        print("\n示例3：获取同步状态")
        status = sync_service.get_sync_status()
        print(f"同步状态: {status}")
        
        # 注意：示例4会阻塞执行，在实际应用中应该在单独的线程中运行
        # 示例4：启动定期同步
        # print("\n示例4：启动定期同步")
        # sync_service.schedule_regular_sync(interval_hours=24)
        
    finally:
        # 关闭数据库会话
        db.close()
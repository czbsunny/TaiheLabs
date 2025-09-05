from typing import Optional, Dict, Any, List, Tuple
import logging
from datetime import date
from sqlalchemy.orm import Session
from core.fund_processor import FundProcessor
from services.fund_sync_service import FundSyncService
from models.fund import Fund, FundTag, FundCategory, FundCategoryMapping

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FundService:
    """基金服务层
    整合所有基金处理功能，为API层提供统一的服务接口
    """
    
    def __init__(self, db: Session, data_sources: Optional[List[FundDataSource]] = None):
        """初始化基金服务
        
        参数:
            db: 数据库会话
            data_sources: 基金数据源列表，可选
        """
        self.db = db
        self.fund_processor = FundProcessor(db)
        self.sync_service = FundSyncService(db, data_sources)
    
    # ---------- 基础信息管理 ----------
    
    def get_fund_info(self, fund_code: str, include_extended: bool = True) -> Optional[Dict[str, Any]]:
        """获取基金信息
        
        参数:
            fund_code: 基金代码
            include_extended: 是否包含扩展信息
        
        返回:
            基金信息字典，如果未找到则返回None
        """
        # 获取基本信息
        basic_fund = self.fund_processor.get_fund_info(fund_code)
        if not basic_fund:
            return None
        
        # 转换为字典
        fund_dict = {
            'fund_code': basic_fund.fund_code,
            'name': basic_fund.name,
            'fund_type': basic_fund.fund_type,
            'created_at': basic_fund.created_at.isoformat() if basic_fund.created_at else None,
            'short_name': getattr(basic_fund, 'short_name', None),
            'company': getattr(basic_fund, 'company', None),
            'manager': getattr(basic_fund, 'manager', None),
            'issue_date': getattr(basic_fund, 'issue_date', None).isoformat() if getattr(basic_fund, 'issue_date', None) else None,
            'establish_date': getattr(basic_fund, 'establish_date', None).isoformat() if getattr(basic_fund, 'establish_date', None) else None,
            'purchase_fee_rate': getattr(basic_fund, 'purchase_fee_rate', None),
            'redemption_fee_rate': getattr(basic_fund, 'redemption_fee_rate', None),
            'management_fee_rate': getattr(basic_fund, 'management_fee_rate', None),
            'custodian_fee_rate': getattr(basic_fund, 'custodian_fee_rate', None),
            'total_asset': getattr(basic_fund, 'total_asset', None),
            'share_size': getattr(basic_fund, 'share_size', None),
            'update_date': getattr(basic_fund, 'update_date', None).isoformat() if getattr(basic_fund, 'update_date', None) else None,
            'risk_level': getattr(basic_fund, 'risk_level', None),
            'benchmark': getattr(basic_fund, 'benchmark', None),
            'investment_scope': getattr(basic_fund, 'investment_scope', None),
            'investment_strategy': getattr(basic_fund, 'investment_strategy', None),
            'updated_at': getattr(basic_fund, 'updated_at', None).isoformat() if getattr(basic_fund, 'updated_at', None) else None
        }
        
        # 如果需要包含扩展信息
        if include_extended:
            # 添加标签
            fund_dict['tags'] = self.fund_processor.get_fund_tags(fund_code)
            
            # 添加分类
            fund_dict['categories'] = self.fund_processor.get_fund_categories(fund_code)
        
        return fund_dict
    
    def create_fund(self, fund_code: str, name: str, fund_type: Optional[str] = None, 
                   extended_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """创建新基金
        
        参数:
            fund_code: 基金代码
            name: 基金名称
            fund_type: 基金类型，可选
            extended_info: 扩展信息，可选
        
        返回:
            创建的基金信息字典，如果失败则返回None
        """
        try:
            # 创建基本基金信息
            basic_fund = self.fund_processor.create_fund(fund_code, name, fund_type)
            if not basic_fund:
                logger.error(f"创建基金基本信息失败: {fund_code}")
                return None
            
            # 如果提供了扩展信息，创建扩展基金信息
            if extended_info:
                extended_fund = self.fund_processor_extended.create_or_update_fund_extended(fund_code, extended_info)
                if not extended_fund:
                    logger.warning(f"创建基金扩展信息失败: {fund_code}")
            
            # 返回完整的基金信息
            return self.get_fund_info(fund_code)
        except Exception as e:
            logger.error(f"创建基金时发生异常: {str(e)}")
            return None
    
    def update_fund(self, fund_code: str, fund_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新基金信息
        
        参数:
            fund_code: 基金代码
            fund_data: 包含要更新的基金信息的字典
        
        返回:
            更新后的基金信息字典，如果失败则返回None
        """
        try:
            # 检查基金是否存在
            existing_fund = self.fund_processor.get_fund_info(fund_code)
            if not existing_fund:
                logger.error(f"基金不存在: {fund_code}")
                return None
            
            # 提取基本信息更新
            basic_update = {
                'name': fund_data.get('name', existing_fund.name),
                'fund_type': fund_data.get('fund_type', existing_fund.fund_type)
            }
            
            # 更新基本信息
            updated_basic = self.fund_processor.create_fund(fund_code, **basic_update)
            if not updated_basic:
                logger.error(f"更新基金基本信息失败: {fund_code}")
                return None
            
            # 更新扩展信息
            # 排除基本信息字段
            extended_data = {k: v for k, v in fund_data.items() if k not in ['name', 'fund_type']}
            if extended_data:
                updated_extended = self.fund_processor_extended.create_or_update_fund_extended(fund_code, extended_data)
                if not updated_extended:
                    logger.warning(f"更新基金扩展信息失败: {fund_code}")
            
            # 返回更新后的基金信息
            return self.get_fund_info(fund_code)
        except Exception as e:
            logger.error(f"更新基金时发生异常: {str(e)}")
            return None
    
    def delete_fund(self, fund_code: str) -> bool:
        """删除基金信息
        
        参数:
            fund_code: 基金代码
        
        返回:
            删除结果，True表示成功，False表示失败
        """
        try:
            # 检查基金是否存在
            existing_fund = self.fund_processor.get_fund_info(fund_code)
            if not existing_fund:
                logger.warning(f"基金不存在: {fund_code}")
                return False
            
            # 删除扩展信息
            extended_fund = self.fund_processor_extended.get_fund_extended_info(fund_code)
            if extended_fund:
                self.db.delete(extended_fund)
            
            # 删除标签
            tags = self.db.query(FundTag).filter(FundTag.fund_code == fund_code).all()
            for tag in tags:
                self.db.delete(tag)
            
            # 删除分类映射（这里需要导入FundCategoryMapping）
            from models.fund_extended import FundCategoryMapping
            mappings = self.db.query(FundCategoryMapping).filter(FundCategoryMapping.fund_code == fund_code).all()
            for mapping in mappings:
                self.db.delete(mapping)
            
            # 删除基本信息
            self.db.delete(existing_fund)
            
            self.db.commit()
            logger.info(f"基金已成功删除: {fund_code}")
            return True
        except Exception as e:
            logger.error(f"删除基金时发生异常: {str(e)}")
            self.db.rollback()
            return False
    
    # ---------- 批量操作 ----------
    
    def batch_process_funds(self, funds_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量处理基金信息
        
        参数:
            funds_data: 包含多个基金信息的列表
        
        返回:
            处理结果列表
        """
        return self.fund_processor_extended.batch_process_funds(funds_data)
    
    def export_funds(self, fund_codes: List[str] = None) -> List[Dict[str, Any]]:
        """导出基金信息
        
        参数:
            fund_codes: 基金代码列表，如果为None则导出所有基金
        
        返回:
            基金信息列表
        """
        return self.fund_processor_extended.export_funds_to_dict(fund_codes)
    
    # ---------- 标签管理 ----------
    
    def add_fund_tag(self, fund_code: str, tag_name: str, tag_type: Optional[str] = None) -> bool:
        """为基金添加标签
        
        参数:
            fund_code: 基金代码
            tag_name: 标签名称
            tag_type: 标签类型，可选
        
        返回:
            添加结果，True表示成功，False表示失败
        """
        return self.fund_processor_extended.add_fund_tag(fund_code, tag_name, tag_type)
    
    def remove_fund_tag(self, fund_code: str, tag_name: str) -> bool:
        """移除基金的标签
        
        参数:
            fund_code: 基金代码
            tag_name: 标签名称
        
        返回:
            移除结果，True表示成功，False表示失败
        """
        return self.fund_processor_extended.remove_fund_tag(fund_code, tag_name)
    
    def get_fund_tags(self, fund_code: str) -> List[Dict[str, str]]:
        """获取基金的所有标签
        
        参数:
            fund_code: 基金代码
        
        返回:
            标签列表
        """
        return self.fund_processor_extended.get_fund_tags(fund_code)
    
    # ---------- 分类管理 ----------
    
    def add_fund_to_category(self, fund_code: str, category_id: int) -> bool:
        """将基金添加到分类
        
        参数:
            fund_code: 基金代码
            category_id: 分类ID
        
        返回:
            添加结果，True表示成功，False表示失败
        """
        return self.fund_processor_extended.add_fund_to_category(fund_code, category_id)
    
    def remove_fund_from_category(self, fund_code: str, category_id: int) -> bool:
        """将基金从分类中移除
        
        参数:
            fund_code: 基金代码
            category_id: 分类ID
        
        返回:
            移除结果，True表示成功，False表示失败
        """
        try:
            from models.fund_extended import FundCategoryMapping
            mapping = self.db.query(FundCategoryMapping).filter(
                FundCategoryMapping.fund_code == fund_code,
                FundCategoryMapping.category_id == category_id
            ).first()
            
            if mapping:
                self.db.delete(mapping)
                self.db.commit()
                return True
            
            return False
        except Exception as e:
            logger.error(f"从分类中移除基金时发生异常: {str(e)}")
            self.db.rollback()
            return False
    
    def get_fund_categories(self, fund_code: str) -> List[Dict[str, Any]]:
        """获取基金的所有分类
        
        参数:
            fund_code: 基金代码
        
        返回:
            分类列表
        """
        return self.fund_processor_extended.get_fund_categories(fund_code)
    
    def create_fund_category(self, category_name: str, category_type: Optional[str] = None,
                           parent_id: Optional[int] = None, description: Optional[str] = None) -> Optional[int]:
        """创建基金分类
        
        参数:
            category_name: 分类名称
            category_type: 分类类型，可选
            parent_id: 父分类ID，可选
            description: 分类描述，可选
        
        返回:
            创建的分类ID，如果失败则返回None
        """
        return self.fund_processor_extended.create_fund_category(category_name, category_type, parent_id, description)
    
    def get_all_categories(self) -> List[Dict[str, Any]]:
        """获取所有基金分类
        
        返回:
            分类列表
        """
        try:
            categories = self.db.query(FundCategory).all()
            return [{
                'id': category.id,
                'name': category.category_name,
                'type': category.category_type,
                'parent_id': category.parent_id,
                'description': category.description
            } for category in categories]
        except Exception as e:
            logger.error(f"获取所有分类时发生异常: {str(e)}")
            return []
    
    # ---------- 版本管理 ----------
    
    def create_version(self, fund_code: str, changed_fields: Dict[str, Any], created_by: Optional[str] = None) -> bool:
        """创建基金信息版本
        
        参数:
            fund_code: 基金代码
            changed_fields: 变更的字段字典
            created_by: 操作人，可选
        
        返回:
            创建结果，True表示成功，False表示失败
        """
        return self.fund_processor_extended.create_version(fund_code, changed_fields, created_by)
    
    def get_fund_versions(self, fund_code: str) -> List[Dict[str, Any]]:
        """获取基金的所有版本
        
        参数:
            fund_code: 基金代码
        
        返回:
            版本列表
        """
        return self.fund_processor_extended.get_fund_versions(fund_code)
    
    # ---------- 数据同步 ----------
    
    def sync_single_fund(self, fund_code: str, force_update: bool = False) -> bool:
        """同步单个基金信息
        
        参数:
            fund_code: 基金代码
            force_update: 是否强制更新，即使数据已存在
        
        返回:
            同步结果，True表示成功，False表示失败
        """
        return self.sync_service.sync_single_fund(fund_code, force_update)
    
    def sync_multiple_funds(self, fund_codes: List[str], force_update: bool = False) -> Dict[str, Any]:
        """同步多个基金信息
        
        参数:
            fund_codes: 基金代码列表
            force_update: 是否强制更新，即使数据已存在
        
        返回:
            同步结果统计
        """
        return self.sync_service.sync_multiple_funds(fund_codes, force_update)
    
    def sync_all_funds(self, force_update: bool = False) -> Dict[str, Any]:
        """同步所有基金信息
        
        参数:
            force_update: 是否强制更新，即使数据已存在
        
        返回:
            同步结果统计
        """
        return self.sync_service.sync_all_funds(force_update)
    
    def get_sync_status(self) -> Dict[str, Any]:
        """获取同步状态信息
        
        返回:
            同步状态字典
        """
        return self.sync_service.get_sync_status()
    
    # ---------- 查询和搜索 ----------
    
    def search_funds(self, keyword: str, fund_type: Optional[str] = None, 
                    page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """搜索基金
        
        参数:
            keyword: 搜索关键词
            fund_type: 基金类型，可选
            page: 页码，默认为1
            page_size: 每页数量，默认为20
        
        返回:
            包含搜索结果的字典
        """
        try:
            # 构建查询
            query = self.db.query(Fund)
            
            # 关键词搜索
            if keyword:
                query = query.filter(Fund.name.ilike(f"%{keyword}%"))
            
            # 基金类型过滤
            if fund_type:
                query = query.filter(Fund.fund_type == fund_type)
            
            # 计算总数
            total = query.count()
            
            # 分页
            start = (page - 1) * page_size
            funds = query.offset(start).limit(page_size).all()
            
            # 构建结果
            results = {
                'total': total,
                'page': page,
                'page_size': page_size,
                'funds': []
            }
            
            # 转换为字典并添加基本信息
            for fund in funds:
                fund_dict = {
                    'fund_code': fund.fund_code,
                    'name': fund.name,
                    'fund_type': fund.fund_type,
                    'company': fund.company,
                    'manager': fund.manager
                }
                
                results['funds'].append(fund_dict)
            
            return results
        except Exception as e:
            logger.error(f"搜索基金时发生异常: {str(e)}")
            return {
                'total': 0,
                'page': page,
                'page_size': page_size,
                'funds': []
            }
    
    def get_funds_by_category(self, category_id: int, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """获取指定分类下的基金
        
        参数:
            category_id: 分类ID
            page: 页码，默认为1
            page_size: 每页数量，默认为20
        
        返回:
            包含基金列表的字典
        """
        try:
            # 构建查询
            query = self.db.query(Fund).join(
                FundCategoryMapping, Fund.fund_code == FundCategoryMapping.fund_code
            ).filter(FundCategoryMapping.category_id == category_id)
            
            # 计算总数
            total = query.count()
            
            # 分页
            start = (page - 1) * page_size
            funds = query.offset(start).limit(page_size).all()
            
            # 构建结果
            results = {
                'total': total,
                'page': page,
                'page_size': page_size,
                'funds': []
            }
            
            # 转换为字典
            for fund in funds:
                results['funds'].append({
                    'fund_code': fund.fund_code,
                    'name': fund.name,
                    'fund_type': fund.fund_type,
                    'company': fund.company,
                    'manager': fund.manager
                })
            
            return results
        except Exception as e:
            logger.error(f"获取分类下的基金时发生异常: {str(e)}")
            return {
                'total': 0,
                'page': page,
                'page_size': page_size,
                'funds': []
            }
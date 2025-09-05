from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from database.init_db import get_db
from services.fund_service import FundService
from core.fund_data_source import MockFundDataSource

# 创建路由
fund_router = APIRouter(prefix="/api/fund", tags=["fund"])

# 依赖项：获取基金服务实例
def get_fund_service(db: Session = Depends(get_db)):
    # 使用模拟数据源，实际应用中可以替换为真实数据源
    data_source = MockFundDataSource()
    return FundService(db, [data_source])

# ---------- 基础信息管理 ----------

@fund_router.get("/{fund_code}", response_model=Optional[Dict[str, Any]])
def get_fund_by_code(
    fund_code: str,
    include_extended: bool = Query(default=True, description="是否包含扩展信息"),
    fund_service: FundService = Depends(get_fund_service)
):
    """根据基金代码获取基金信息"""
    fund_info = fund_service.get_fund_info(fund_code, include_extended)
    if not fund_info:
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")
    return fund_info

@fund_router.post("/", response_model=Optional[Dict[str, Any]])
def create_new_fund(
    fund_data: Dict[str, Any] = Body(..., description="基金信息"),
    fund_service: FundService = Depends(get_fund_service)
):
    """创建新基金"""
    # 检查必要字段
    fund_code = fund_data.get("fund_code") or fund_data.get("code")
    name = fund_data.get("name")
    
    if not fund_code or not name:
        raise HTTPException(status_code=400, detail="基金代码和名称是必需的")
    
    # 提取基金类型
    fund_type = fund_data.get("fund_type")
    
    # 提取扩展信息
    extended_info = {
        k: v for k, v in fund_data.items()
        if k not in ["fund_code", "code", "name", "fund_type"]
    }
    
    # 创建基金
    created_fund = fund_service.create_fund(fund_code, name, fund_type, extended_info)
    
    if not created_fund:
        raise HTTPException(status_code=500, detail="创建基金失败")
    
    return created_fund

@fund_router.put("/{fund_code}", response_model=Optional[Dict[str, Any]])
def update_fund_info(
    fund_code: str,
    fund_data: Dict[str, Any] = Body(..., description="要更新的基金信息"),
    fund_service: FundService = Depends(get_fund_service)
):
    """更新基金信息"""
    updated_fund = fund_service.update_fund(fund_code, fund_data)
    
    if not updated_fund:
        # 检查是否是因为基金不存在
        if not fund_service.get_fund_info(fund_code, include_extended=False):
            raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")
        raise HTTPException(status_code=500, detail="更新基金信息失败")
    
    return updated_fund

@fund_router.delete("/{fund_code}", response_model=Dict[str, str])
def delete_fund_by_code(
    fund_code: str,
    fund_service: FundService = Depends(get_fund_service)
):
    """删除基金"""
    success = fund_service.delete_fund(fund_code)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在或删除失败")
    
    return {"message": f"基金 {fund_code} 已成功删除"}

# ---------- 批量操作 ----------

@fund_router.post("/batch", response_model=List[Dict[str, Any]])
def batch_process_funds(
    funds_data: List[Dict[str, Any]] = Body(..., description="多个基金信息"),
    fund_service: FundService = Depends(get_fund_service)
):
    """批量处理基金信息"""
    results = fund_service.batch_process_funds(funds_data)
    return results

@fund_router.get("/export", response_model=List[Dict[str, Any]])
def export_funds(
    fund_codes: Optional[List[str]] = Query(default=None, description="基金代码列表，如果为空则导出所有基金"),
    fund_service: FundService = Depends(get_fund_service)
):
    """导出基金信息"""
    return fund_service.export_funds(fund_codes)

# ---------- 标签管理 ----------

@fund_router.post("/{fund_code}/tags", response_model=Dict[str, str])
def add_tag_to_fund(
    fund_code: str,
    tag_name: str = Query(..., description="标签名称"),
    tag_type: Optional[str] = Query(default=None, description="标签类型"),
    fund_service: FundService = Depends(get_fund_service)
):
    """为基金添加标签"""
    # 检查基金是否存在
    if not fund_service.get_fund_info(fund_code, include_extended=False):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")
    
    success = fund_service.add_fund_tag(fund_code, tag_name, tag_type)
    
    if not success:
        raise HTTPException(status_code=500, detail="添加标签失败")
    
    return {"message": f"标签 '{tag_name}' 已成功添加到基金 {fund_code}"}

@fund_router.delete("/{fund_code}/tags/{tag_name}", response_model=Dict[str, str])
def remove_tag_from_fund(
    fund_code: str,
    tag_name: str,
    fund_service: FundService = Depends(get_fund_service)
):
    """移除基金的标签"""
    # 检查基金是否存在
    if not fund_service.get_fund_info(fund_code, include_extended=False):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")
    
    success = fund_service.remove_fund_tag(fund_code, tag_name)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"标签 '{tag_name}' 不存在或移除失败")
    
    return {"message": f"标签 '{tag_name}' 已成功从基金 {fund_code} 移除"}

@fund_router.get("/{fund_code}/tags", response_model=List[Dict[str, str]])
def get_fund_tags_api(
    fund_code: str,
    fund_service: FundService = Depends(get_fund_service)
):
    """获取基金的所有标签"""
    # 检查基金是否存在
    if not fund_service.get_fund_info(fund_code, include_extended=False):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")
    
    return fund_service.get_fund_tags(fund_code)

# ---------- 分类管理 ----------

@fund_router.post("/{fund_code}/categories/{category_id}", response_model=Dict[str, str])
def add_fund_to_category_api(
    fund_code: str,
    category_id: int,
    fund_service: FundService = Depends(get_fund_service)
):
    """将基金添加到分类"""
    # 检查基金是否存在
    if not fund_service.get_fund_info(fund_code, include_extended=False):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")
    
    success = fund_service.add_fund_to_category(fund_code, category_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"分类 {category_id} 不存在或添加失败")
    
    return {"message": f"基金 {fund_code} 已成功添加到分类 {category_id}"}

@fund_router.delete("/{fund_code}/categories/{category_id}", response_model=Dict[str, str])
def remove_fund_from_category_api(
    fund_code: str,
    category_id: int,
    fund_service: FundService = Depends(get_fund_service)
):
    """将基金从分类中移除"""
    # 检查基金是否存在
    if not fund_service.get_fund_info(fund_code, include_extended=False):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")
    
    success = fund_service.remove_fund_from_category(fund_code, category_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"基金未在分类 {category_id} 中或移除失败")
    
    return {"message": f"基金 {fund_code} 已成功从分类 {category_id} 移除"}

@fund_router.get("/{fund_code}/categories", response_model=List[Dict[str, Any]])
def get_fund_categories_api(
    fund_code: str,
    fund_service: FundService = Depends(get_fund_service)
):
    """获取基金的所有分类"""
    # 检查基金是否存在
    if not fund_service.get_fund_info(fund_code, include_extended=False):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")
    
    return fund_service.get_fund_categories(fund_code)

@fund_router.post("/categories", response_model=Dict[str, Any])
def create_fund_category_api(
    category_name: str = Query(..., description="分类名称"),
    category_type: Optional[str] = Query(default=None, description="分类类型"),
    parent_id: Optional[int] = Query(default=None, description="父分类ID"),
    description: Optional[str] = Query(default=None, description="分类描述"),
    fund_service: FundService = Depends(get_fund_service)
):
    """创建基金分类"""
    category_id = fund_service.create_fund_category(category_name, category_type, parent_id, description)
    
    if not category_id:
        raise HTTPException(status_code=500, detail="创建分类失败或分类已存在")
    
    return {
        "id": category_id,
        "name": category_name,
        "type": category_type,
        "parent_id": parent_id,
        "description": description,
        "message": f"分类 '{category_name}' 已成功创建"
    }

@fund_router.get("/categories", response_model=List[Dict[str, Any]])
def get_all_categories_api(
    fund_service: FundService = Depends(get_fund_service)
):
    """获取所有基金分类"""
    return fund_service.get_all_categories()

# ---------- 版本管理 ----------

@fund_router.post("/{fund_code}/versions", response_model=Dict[str, str])
def create_fund_version(
    fund_code: str,
    changed_fields: Dict[str, Any] = Body(..., description="变更的字段字典"),
    created_by: Optional[str] = Query(default=None, description="操作人"),
    fund_service: FundService = Depends(get_fund_service)
):
    """创建基金信息版本"""
    # 检查基金是否存在
    if not fund_service.get_fund_info(fund_code, include_extended=False):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")
    
    success = fund_service.create_version(fund_code, changed_fields, created_by)
    
    if not success:
        raise HTTPException(status_code=500, detail="创建版本失败")
    
    return {"message": f"基金 {fund_code} 的版本已成功创建"}

@fund_router.get("/{fund_code}/versions", response_model=List[Dict[str, Any]])
def get_fund_versions_api(
    fund_code: str,
    fund_service: FundService = Depends(get_fund_service)
):
    """获取基金的所有版本"""
    # 检查基金是否存在
    if not fund_service.get_fund_info(fund_code, include_extended=False):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")
    
    return fund_service.get_fund_versions(fund_code)

# ---------- 数据同步 ----------

@fund_router.post("/{fund_code}/sync", response_model=Dict[str, str])
def sync_single_fund_api(
    fund_code: str,
    force_update: bool = Query(default=False, description="是否强制更新"),
    fund_service: FundService = Depends(get_fund_service)
):
    """同步单个基金信息"""
    success = fund_service.sync_single_fund(fund_code, force_update)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"同步基金 {fund_code} 失败")
    
    return {"message": f"基金 {fund_code} 同步成功"}

@fund_router.post("/batch/sync", response_model=Dict[str, Any])
def sync_multiple_funds_api(
    fund_codes: List[str] = Body(..., description="基金代码列表"),
    force_update: bool = Query(default=False, description="是否强制更新"),
    fund_service: FundService = Depends(get_fund_service)
):
    """同步多个基金信息"""
    results = fund_service.sync_multiple_funds(fund_codes, force_update)
    
    if results.get("fail", 0) > 0:
        # 部分或全部失败，返回详细结果
        return results
    
    return results

@fund_router.post("/sync/all", response_model=Dict[str, Any])
def sync_all_funds_api(
    force_update: bool = Query(default=False, description="是否强制更新"),
    fund_service: FundService = Depends(get_fund_service)
):
    """同步所有基金信息"""
    results = fund_service.sync_all_funds(force_update)
    return results

@fund_router.get("/sync/status", response_model=Dict[str, Any])
def get_sync_status_api(
    fund_service: FundService = Depends(get_fund_service)
):
    """获取同步状态信息"""
    return fund_service.get_sync_status()

# ---------- 查询和搜索 ----------

@fund_router.get("/search", response_model=Dict[str, Any])
def search_funds_api(
    keyword: Optional[str] = Query(default=None, description="搜索关键词"),
    fund_type: Optional[str] = Query(default=None, description="基金类型"),
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页数量"),
    fund_service: FundService = Depends(get_fund_service)
):
    """搜索基金"""
    if not keyword and not fund_type:
        raise HTTPException(status_code=400, detail="至少提供关键词或基金类型之一")
    
    return fund_service.search_funds(keyword, fund_type, page, page_size)

@fund_router.get("/categories/{category_id}/funds", response_model=Dict[str, Any])
def get_funds_by_category_api(
    category_id: int,
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页数量"),
    fund_service: FundService = Depends(get_fund_service)
):
    """获取指定分类下的基金"""
    return fund_service.get_funds_by_category(category_id, page, page_size)

# ---------- 健康检查 ----------

@fund_router.get("/health", response_model=Dict[str, str])
def fund_api_health_check():
    """基金API健康检查"""
    return {"status": "ok", "message": "Fund API is running normally"}
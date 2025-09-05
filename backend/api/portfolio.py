from fastapi import APIRouter, Request, Form, Depends, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.orm import Session
from services.portfolio_service import PortfolioService
from services.user_service import UserService
from services.diagnosis_service import DiagnosisService
from database.init_db import get_db
from pathlib import Path
import re

# 使用绝对路径配置模板目录
base_dir = Path(__file__).resolve().parent.parent.parent  # 项目根目录
templates = Jinja2Templates(directory=str(base_dir / "frontend"))

portfolio_router = APIRouter()

@portfolio_router.get("/{user_id}")
def portfolio_dashboard(request: Request, user_id: int, portfolio_id: int = Query(None), db: Session = Depends(get_db)):
    portfolio_service = PortfolioService(db)
    user_service = UserService(db)
    portfolios = portfolio_service.get_user_portfolios(user_id)
    current_user = user_service.get_user(user_id)
    return templates.TemplateResponse("dashboard.html", {"request": request, "portfolios": portfolios, "user_id": user_id, "current_user": current_user, "selected_portfolio_id": portfolio_id})

@portfolio_router.post("/create")
async def create_portfolio(request: Request, db: Session = Depends(get_db)):
    try:
        # Get form data
        form_data = await request.form()
        print(f"Received create portfolio request with form data: {dict(form_data)}")
        
        # Validate required fields
        if not form_data.get('user_id'):
            return JSONResponse(
                status_code=422,
                content={"detail": "Missing required field: user_id", "received_data": dict(form_data)}
            )
            
        # Accept both 'name' and 'portfolio_name' fields to handle frontend inconsistency
        portfolio_name = form_data.get('name') or form_data.get('portfolio_name')
        if not portfolio_name:
            return JSONResponse(
                status_code=422,
                content={"detail": "Missing required field: name", "received_data": dict(form_data)}
            )
        
        # Try to convert user_id to int
        try:
            user_id = int(form_data.get('user_id'))
        except ValueError:
            return JSONResponse(
                status_code=422,
                content={"detail": f"Invalid user_id format. Expected integer, got: {form_data.get('user_id')}", "received_data": dict(form_data)}
            )
        
        description = form_data.get('description', '')  # Default to empty string
        
        # Create portfolio
        portfolio_service = PortfolioService(db)
        portfolio_service.create_portfolio(user_id, portfolio_name, description)
        
        # Success - redirect back to dashboard
        return RedirectResponse(url=f"/api/portfolio/{user_id}", status_code=303)
        
    except Exception as e:
        # Catch any unexpected errors
        print(f"Unexpected error in create_portfolio: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

@portfolio_router.post("/delete")
def delete_portfolio_form(user_id: int = Form(...), portfolio_id: int = Form(...), db: Session = Depends(get_db)):
    """Legacy form-based delete endpoint"""
    portfolio_service = PortfolioService(db)
    portfolio_service.delete_portfolio(portfolio_id)
    return RedirectResponse(url=f"/api/portfolio/{user_id}", status_code=303)

@portfolio_router.delete("/delete/{portfolio_id}")
def delete_portfolio(portfolio_id: int, db: Session = Depends(get_db)):
    """New RESTful delete endpoint that matches frontend usage"""
    portfolio_service = PortfolioService(db)
    success = portfolio_service.delete_portfolio(portfolio_id)
    if success:
        return JSONResponse(status_code=200, content={"success": True, "message": "Portfolio deleted successfully"})
    else:
        return JSONResponse(status_code=404, content={"success": False, "message": "Portfolio not found"})

@portfolio_router.get("/diagnosis/{portfolio_id}")
def get_portfolio_diagnosis(portfolio_id: int, db: Session = Depends(get_db)):
    """
    获取组合诊断结果
    """
    print(f"[诊断请求] 收到组合诊断请求，组合ID: {portfolio_id}")
    diagnosis_service = DiagnosisService(db)
    diagnosis_result = diagnosis_service.get_portfolio_diagnosis(portfolio_id)
    
    print(f"[诊断结果] 组合ID: {portfolio_id}, 诊断结果: {diagnosis_result}")
    
    if not diagnosis_result:
        print(f"[诊断错误] 组合ID: {portfolio_id}, 未找到组合")
        return JSONResponse(status_code=404, content={"success": False, "message": "Portfolio not found"})
    
    return JSONResponse(status_code=200, content={"success": True, "data": diagnosis_result})

@portfolio_router.get("/diagnosis/page/{portfolio_id}")
def diagnosis_page(request: Request, portfolio_id: int, db: Session = Depends(get_db)):
    """
    返回组合诊断页面
    """
    portfolio_service = PortfolioService(db)
    portfolio_data = portfolio_service.get_portfolio_items(portfolio_id)
    user_id = portfolio_data["portfolio"].user_id if portfolio_data else 1  # 默认值为1
    return templates.TemplateResponse("diagnosis.html", {"request": request, "portfolio_id": portfolio_id, "user_id": user_id})

@portfolio_router.get("/diagnosis.html")
def diagnosis_page_with_query_param(request: Request, portfolio_id: int = Query(...), db: Session = Depends(get_db)):
    """
    通过查询参数接收portfolio_id的诊断页面路由
    兼容前端通过查询参数传递portfolio_id的情况
    """
    portfolio_service = PortfolioService(db)
    portfolio_data = portfolio_service.get_portfolio_items(portfolio_id)
    user_id = portfolio_data["portfolio"].user_id if portfolio_data else 1  # 默认值为1
    return templates.TemplateResponse("diagnosis.html", {"request": request, "portfolio_id": portfolio_id, "user_id": user_id})

@portfolio_router.get("/{portfolio_id}/items")
def view_portfolio_items(request: Request, portfolio_id: int, db: Session = Depends(get_db)):
    portfolio_service = PortfolioService(db)
    portfolio_data = portfolio_service.get_portfolio_items(portfolio_id)
    if not portfolio_data:
        return {"error": "Portfolio not found"}
    # 获取用户ID用于返回链接
    user_id = portfolio_data["portfolio"].user_id
    return templates.TemplateResponse(
        "portfolio_items.html",
        {"request": request, "items": portfolio_data["items"], "portfolio_id": portfolio_id, "user_id": user_id}
    )

@portfolio_router.post("/{portfolio_id}/items/add")
def add_portfolio_item(portfolio_id: int, symbol: str = Form(...), quantity: float = Form(...),
                       cost: float = Form(...), name: str = Form(None), hold_amount: float = Form(None),
                       hold_profit: float = Form(None), fund_code: str = Form(None), db: Session = Depends(get_db)):
    print(f"API received: portfolio_id={portfolio_id}, symbol={symbol}, fund_code={fund_code}, quantity={quantity}")
    portfolio_service = PortfolioService(db)
    # 如果提供了fund_code，就使用它，否则使用symbol作为fallback
    # 这样确保基金代码栏只显示code而不是code+name的组合
    actual_symbol = fund_code if fund_code else symbol
    print(f"Using actual_symbol={actual_symbol}")
    portfolio_service.add_item(portfolio_id, actual_symbol, quantity, cost, name=name, hold_amount=hold_amount, hold_profit=hold_profit)
    # 获取用户ID以重定向回dashboard
    portfolio = portfolio_service.get_portfolio_items(portfolio_id)
    if portfolio:
        return RedirectResponse(url=f"/api/portfolio/{portfolio['portfolio'].user_id}", status_code=303)
    return RedirectResponse(url=f"/api/portfolio/{portfolio_id}/items", status_code=303)

@portfolio_router.post("/{portfolio_id}/items/delete")
def delete_portfolio_item(portfolio_id: int, item_id: int = Form(...), db: Session = Depends(get_db)):
    portfolio_service = PortfolioService(db)
    portfolio_service.delete_item(item_id)
    # 获取用户ID以重定向回dashboard
    portfolio = portfolio_service.get_portfolio_items(portfolio_id)
    if portfolio:
        return RedirectResponse(url=f"/api/portfolio/{portfolio['portfolio'].user_id}", status_code=303)
    return RedirectResponse(url=f"/api/portfolio/{portfolio_id}/items", status_code=303)

@portfolio_router.post("/{portfolio_id}/import-selected")
async def import_selected_funds(portfolio_id: int, request: Request, db: Session = Depends(get_db)):
    """
    导入用户选择的基金持仓记录
    """
    try:
        # 获取JSON数据
        data = await request.json()
        funds = data.get("funds", [])
        
        if not funds:
            return {"success": False, "message": "没有选择要导入的基金"}
        
        portfolio_service = PortfolioService(db)
        imported_count = 0
        
        # 遍历所有选中的基金并添加到投资组合
        for fund in funds:
            # 获取基金代码和名称
            fund_code = fund.get("code") or fund.get("fund_code") or ""
            # 尝试从基金名称中提取基金代码
            if not fund_code and fund.get("name", ""):
                code_match = re.search(r'[0-9]{6}', fund.get("name", ""))
                if code_match:
                    fund_code = code_match.group(0)
            
            fund_name = fund.get("name") or fund.get("fund_name") or ""
            
            # 构建symbol，使用代码和名称的组合
            symbol = f"{fund_code}" if fund_code else fund_name
            
            # 获取持仓信息 - 适配不同的数据结构
            shares = float(fund.get("shares", 0))
            purchase_price = float(fund.get("purchase_price", 0))
            hold_amount = float(fund.get("amount", 0) or fund.get("hold_amount", 0) or 0)
            
            # 处理持有收益，支持字符串格式如'+20.57'或'-10.23'
            hold_profit = 0
            profit_str = fund.get("profit", "")
            if profit_str:
                try:
                    # 直接转换为浮点数（Python的float()可以处理带有正负号的数字字符串）
                    hold_profit = float(profit_str)
                except ValueError:
                    # 如果转换失败，保持为0
                    pass
            
            # 计算成本和数量 - 增强逻辑以处理只有金额的情况
            cost = 0
            quantity = 0
            
            if shares > 0 and purchase_price > 0:
                # 如果有份额和购买价格，直接使用
                cost = purchase_price
                quantity = shares
            elif hold_amount > 0 and purchase_price > 0:
                # 如果有持仓金额和购买价格，计算数量
                cost = purchase_price
                quantity = hold_amount / purchase_price
            elif hold_amount > 0 and shares > 0:
                # 如果有持仓金额和份额，计算成本
                cost = hold_amount / shares
                quantity = shares
            elif hold_amount > 0:
                # 特殊处理：只有持仓金额的情况（从OCR识别结果）
                # 假设单位成本为1，实际数量等于金额
                # 这是一种简化处理，实际应用中可能需要更复杂的逻辑
                cost = 1.0
                quantity = hold_amount
            
            # 添加持仓项（确保有有效的代码/名称和数量）
            if (fund_code or fund_name) and quantity > 0:
                portfolio_service.add_item(
                    portfolio_id, 
                    symbol, 
                    quantity, 
                    cost,
                    name=fund_name,
                    hold_amount=hold_amount,
                    hold_profit=hold_profit
                )
                imported_count += 1
        
        return {"success": True, "imported_count": imported_count}
    except Exception as e:
        # 记录异常
        import traceback
        traceback.print_exc()
        return {"success": False, "message": str(e)}
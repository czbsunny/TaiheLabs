from fastapi import APIRouter, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from services.portfolio_service import PortfolioService
from database.init_db import get_db
from pathlib import Path

# 使用绝对路径配置模板目录
base_dir = Path(__file__).resolve().parent.parent.parent  # 项目根目录
templates = Jinja2Templates(directory=str(base_dir / "frontend"))

portfolio_router = APIRouter()

@portfolio_router.get("/{user_id}")
def portfolio_dashboard(request: Request, user_id: int, db: Session = Depends(get_db)):
    portfolio_service = PortfolioService(db)
    portfolios = portfolio_service.get_user_portfolios(user_id)
    return templates.TemplateResponse("dashboard.html", {"request": request, "portfolios": portfolios, "user_id": user_id})

@portfolio_router.post("/create")
def create_portfolio(user_id: int = Form(...), name: str = Form(...), description: str = Form(None), db: Session = Depends(get_db)):
    portfolio_service = PortfolioService(db)
    portfolio_service.create_portfolio(user_id, name, description)
    return RedirectResponse(url=f"/api/portfolio/{user_id}", status_code=303)

@portfolio_router.post("/delete")
def delete_portfolio(user_id: int = Form(...), portfolio_id: int = Form(...), db: Session = Depends(get_db)):
    portfolio_service = PortfolioService(db)
    portfolio_service.delete_portfolio(portfolio_id)
    return RedirectResponse(url=f"/api/portfolio/{user_id}", status_code=303)

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
                       cost: float = Form(...), db: Session = Depends(get_db)):
    portfolio_service = PortfolioService(db)
    portfolio_service.add_item(portfolio_id, symbol, quantity, cost)
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
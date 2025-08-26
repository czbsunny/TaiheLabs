from fastapi import APIRouter, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from services.portfolio_service import PortfolioService
# from backend.services.portfolio_item_service import PortfolioItemService
from database.init_db import get_db
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend目录上一级
templates = Jinja2Templates(directory=str(BASE_DIR / "frontend"))

portfolio_router = APIRouter()

@portfolio_router.get("/portfolios/{user_id}")
def portfolio_dashboard(request: Request, user_id: int, db: Session = Depends(get_db)):
    portfolios = PortfolioService.get_user_portfolios(db, user_id)
    return templates.TemplateResponse("dashboard.html", {"request": request, "portfolios": portfolios, "user_id": user_id})

@portfolio_router.post("/portfolios/create")
def create_portfolio(user_id: int = Form(...), name: str = Form(...), db: Session = Depends(get_db)):
    PortfolioService.create_portfolio(db, user_id, name)
    return RedirectResponse(url=f"/portfolios/{user_id}", status_code=303)

@portfolio_router.post("/portfolios/delete")
def delete_portfolio(user_id: int = Form(...), portfolio_id: int = Form(...), db: Session = Depends(get_db)):
    PortfolioService.delete_portfolio(db, portfolio_id)
    return RedirectResponse(url=f"/portfolios/{user_id}", status_code=303)

# @portfolio_router.get("/portfolio/{portfolio_id}/items")
# def view_portfolio_items(request: Request, portfolio_id: int, db: Session = Depends(get_db)):
#     items = PortfolioItemService.get_portfolio_items(db, portfolio_id)
#     return templates.TemplateResponse(
#         "portfolio_items.html",
#         {"request": request, "items": items, "portfolio_id": portfolio_id}
#     )

# @portfolio_router.post("/portfolio/{portfolio_id}/items/add")
# def add_portfolio_item(portfolio_id: int, symbol: str = Form(...), quantity: float = Form(...),
#                        cost: float = Form(...), db: Session = Depends(get_db)):
#     PortfolioItemService.add_item(db, portfolio_id, symbol, quantity, cost)
#     return RedirectResponse(url=f"/portfolio/{portfolio_id}/items", status_code=303)

# @portfolio_router.post("/portfolio/{portfolio_id}/items/delete")
# def delete_portfolio_item(portfolio_id: int, item_id: int = Form(...), db: Session = Depends(get_db)):
#     PortfolioItemService.delete_item(db, item_id)
#     return RedirectResponse(url=f"/portfolio/{portfolio_id}/items", status_code=303)
from backend.database import db_session
from backend.models import SummaryAccount, FundAccount, StockAccount

class SummaryService:
    """汇总账户服务"""
    
    @staticmethod
    def update_summary_balance(user_id):
        """更新用户的汇总账户余额"""
        # 查询用户的所有基金账户和股票账户
        fund_accounts = FundAccount.query.filter_by(user_id=user_id).all()
        stock_accounts = StockAccount.query.filter_by(user_id=user_id).all()
        
        # 计算总余额
        total_balance = sum(account.balance for account in fund_accounts + stock_accounts)
        
        # 获取或创建汇总账户
        summary_account = SummaryAccount.query.filter_by(user_id=user_id).first()
        
        if summary_account:
            # 更新余额
            summary_account.balance = total_balance
            # 更新关联的账户ID
            linked_ids = ','.join(str(account.id) for account in fund_accounts + stock_accounts)
            summary_account.linked_account_ids = linked_ids
            db_session.commit()
            return summary_account
        else:
            # 创建汇总账户
            summary_account = SummaryAccount._create_for_user(user_id)
            summary_account.balance = total_balance
            linked_ids = ','.join(str(account.id) for account in fund_accounts + stock_accounts)
            summary_account.linked_account_ids = linked_ids
            db_session.add(summary_account)
            db_session.commit()
            return summary_account
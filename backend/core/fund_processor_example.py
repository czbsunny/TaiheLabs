"""基金基础信息处理模块使用示例
这个文件展示了如何在项目中集成和使用FundProcessor类
"""

from sqlalchemy.orm import Session
from database.init_db import get_db
from core.fund_processor import FundProcessor
from api.portfolio import portfolio_router
from fastapi import Depends

# 示例1：在路由处理函数中使用FundProcessor
def process_fund_data(fund_name: str, db: Session = Depends(get_db)):
    """处理基金数据的示例函数"""
    # 初始化FundProcessor实例
    fund_processor = FundProcessor(db)
    
    # 解析基金名称，提取代码
    fund_info = fund_processor.parse_fund_from_text(fund_name)
    
    if fund_info['code']:
        # 检查基金是否已存在于数据库
        existing_fund = fund_processor.get_fund_info(fund_info['code'])
        if existing_fund:
            print(f"基金已存在: {existing_fund.name}({existing_fund.fund_code})")
        else:
            # 创建新的基金记录
            new_fund = fund_processor.create_fund(
                fund_code=fund_info['code'],
                name=fund_info['name']
            )
            print(f"已创建新基金: {new_fund.name}({new_fund.fund_code})")
    else:
        print(f"无法从名称中提取基金代码: {fund_info['name']}")
    
    return fund_info

# 示例2：如何在现有OCR处理逻辑中集成FundProcessor
# 假设这是现有的OCR处理函数
def existing_ocr_processing(ocr_result: dict, db: Session = Depends(get_db)):
    """现有OCR处理逻辑中集成FundProcessor"""
    fund_processor = FundProcessor(db)
    
    # 假设ocr_result包含识别出的基金信息
    portfolios = ocr_result.get('portfolios', [])
    
    for portfolio in portfolios:
        if 'funds' in portfolio:
            for fund in portfolio['funds']:
                # 使用FundProcessor处理基金信息
                fund_name = fund.get('name', '')
                
                # 解析基金信息
                parsed_fund = fund_processor.parse_fund_from_text(fund_name)
                
                # 获取或创建基金记录
                if parsed_fund['code']:
                    fund_info = fund_processor.get_fund_info(parsed_fund['code'])
                    if not fund_info:
                        # 如果基金不存在，创建新记录
                        fund_info = fund_processor.create_fund(
                            fund_code=parsed_fund['code'],
                            name=parsed_fund['name']
                        )
                    
                    # 更新fund字典，确保使用标准化的名称
                    fund['name'] = fund_info.name
                    fund['code'] = fund_info.fund_code
                
                # 处理其他基金数据
                # ...
    
    return ocr_result

# 示例3：替换现有代码中的基金代码提取逻辑
# 假设diagnosis_service.py中的_extract_fund_code方法
# 可以被FundProcessor的extract_fund_code方法替代

# 原始代码
# def _extract_fund_code(self, symbol):
#     import re
#     fund_code_match = re.search(r'[0-9]{6}', symbol)
#     return fund_code_match.group(0) if fund_code_match else None

# 替换后的代码
# def _extract_fund_code(self, symbol):
#     fund_processor = FundProcessor(self.db)
#     return fund_processor.extract_fund_code(symbol)

# 示例4：在portfolio.py中集成FundProcessor
# 以下是如何在import_from_ocr方法中使用FundProcessor

# 原始代码
# # 尝试从基金名称中提取基金代码
# if not fund_code and fund.get('name', ''):
#     code_match = re.search(r'[0-9]{6}', fund.get('name', ''))
#     if code_match:
#         fund_code = code_match.group(0)

# 替换后的代码
# from core.fund_processor import FundProcessor
# 
# # 初始化FundProcessor
# fund_processor = FundProcessor(db)
# 
# # 尝试从基金名称中提取基金代码
# if not fund_code and fund.get('name', ''):
#     fund_info = fund_processor.parse_fund_from_text(fund.get('name', ''))
#     fund_code = fund_info.get('code')
#     # 同时获取标准化的基金名称
#     normalized_name = fund_info.get('name')

# 示例5：如何使用FundProcessor进行基金代码验证

def validate_fund_input(fund_code: str):
    """验证用户输入的基金代码"""
    fund_processor = FundProcessor()  # 验证功能不需要数据库
    
    if not fund_processor.validate_fund_code(fund_code):
        return False, "基金代码必须是6位数字"
    
    return True, "基金代码格式正确"

# 运行示例
if __name__ == '__main__':
    # 获取数据库会话
    db = next(get_db())
    
    # 示例1：处理基金名称
    result = process_fund_data("易方达消费精选 012808", db)
    print(f"处理结果: {result}")
    
    # 示例5：验证基金代码
    is_valid, message = validate_fund_input("012808")
    print(f"基金代码验证: {is_valid}, {message}")
    
    is_valid, message = validate_fund_input("abc123")
    print(f"基金代码验证: {is_valid}, {message}")
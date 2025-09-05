# 基金基础信息处理模块

## 概述
这个模块提供了基金基础信息的处理功能，包括基金代码提取、名称标准化、基金信息查询与创建等。它旨在将项目中分散的基金信息处理逻辑整合到一个统一的模块中，提高代码的复用性和可维护性。

## 文件结构
```
backend/core/
├── fund_processor.py      # 核心处理类FundProcessor的实现
├── fund_processor_example.py  # 使用示例
├── README.md              # 模块说明文档
└── ocr_handler.py         # 现有的OCR处理模块（保留）
```

## FundProcessor 类

### 主要功能
1. **基金代码提取**：从文本中提取6位数字的基金代码
2. **基金名称标准化**：处理基金名称中的空格、省略号等问题
3. **基金信息查询**：根据基金代码从数据库中查询基金信息
4. **基金记录创建**：创建或更新基金记录
5. **数字提取**：从文本中提取数字，支持普通数字和带符号数字
6. **基金信息解析**：从文本中解析基金信息（名称和代码）
7. **基金代码验证**：验证基金代码是否为有效的6位数字

### 方法说明

#### 初始化
```python
from core.fund_processor import FundProcessor
from database.init_db import get_db

db = next(get_db())
fund_processor = FundProcessor(db)  # 传入数据库会话，可选
```

#### 基金代码提取
```python
fund_code = fund_processor.extract_fund_code("易方达消费精选 012808")
# 返回: "012808"
```

#### 基金名称标准化
```python
normalized_name = fund_processor.normalize_name("易 方 达 消 费 精 选...")
# 返回: "易方达消费精选…"
```

#### 基金信息查询
```python
fund_info = fund_processor.get_fund_info("012808")
# 返回Fund对象或None
```

#### 基金记录创建
```python
new_fund = fund_processor.create_fund(
    fund_code="012808",
    name="易方达消费精选",
    fund_type="股票型"  # 可选
)
# 返回创建的Fund对象或None
```

#### 基金信息解析
```python
parsed_fund = fund_processor.parse_fund_from_text("易方达消费精选 012808")
# 返回: {'name': '易方达消费精选', 'code': '012808'}
```

#### 基金代码验证
```python
is_valid = fund_processor.validate_fund_code("012808")
# 返回: True
```

## 集成到现有代码

### 替换现有代码中的基金代码提取逻辑

在 `diagnosis_service.py` 中：

```python
# 原始代码
# def _extract_fund_code(self, symbol):
#     import re
#     fund_code_match = re.search(r'[0-9]{6}', symbol)
#     return fund_code_match.group(0) if fund_code_match else None

# 替换后的代码
from core.fund_processor import FundProcessor

def _extract_fund_code(self, symbol):
    fund_processor = FundProcessor(self.db)
    return fund_processor.extract_fund_code(symbol)
```

### 在OCR处理逻辑中集成

在 `ocr.py` 或相关OCR处理文件中：

```python
from core.fund_processor import FundProcessor

def process_ocr_result(ocr_result, db):
    fund_processor = FundProcessor(db)
    
    # 处理识别到的基金信息
    portfolios = ocr_result.get('portfolios', [])
    for portfolio in portfolios:
        # 使用FundProcessor处理基金信息
        # ...
    
    return ocr_result
```

### 在组合导入功能中集成

在 `portfolio.py` 的 `import_from_ocr` 方法中：

```python
from core.fund_processor import FundProcessor

@portfolio_router.post("/import-from-ocr")
async def import_from_ocr(request: Request, db: Session = Depends(get_db)):
    try:
        import_data = await request.json()
        portfolio_id = import_data.get('portfolio_id')
        ocr_portfolios = import_data.get('portfolios', [])
        
        # 初始化FundProcessor
        fund_processor = FundProcessor(db)
        portfolio_service = PortfolioService(db)
        added_count = 0
        
        # 遍历每个识别到的投资组合
        for portfolio in ocr_portfolios:
            # 遍历组合中的每个基金
            if 'funds' in portfolio:
                for fund in portfolio['funds']:
                    # 使用FundProcessor解析基金信息
                    fund_name = fund.get('name', '')
                    fund_info = fund_processor.parse_fund_from_text(fund_name)
                    
                    fund_code = fund.get('code', '') or fund.get('fund_code', '') or fund_info.get('code', '')
                    
                    # 其他处理逻辑
                    # ...
    
    # 其余代码保持不变
    # ...
```

## 单元测试
可以通过运行以下命令来执行单元测试：

```bash
cd /path/to/TaiheLabs
python -m backend.tests.test_fund_processor
```

## 设计思路
1. **统一化处理**：将分散在不同文件中的基金信息处理逻辑整合到一个统一的模块中
2. **可扩展性**：设计良好的接口，便于后续添加新功能
3. **错误处理**：包含完善的错误处理机制，确保模块的稳定性
4. **测试覆盖**：提供全面的单元测试，确保功能的正确性
5. **文档完善**：提供详细的文档和使用示例，方便其他开发人员使用

## 注意事项
1. 使用前确保已正确配置数据库连接
2. 在不需要数据库操作的场景下，可以不传入数据库会话
3. 基金代码验证只检查格式是否为6位数字，不检查代码是否真实存在
4. 标准化基金名称时会根据中文比例决定是否保留空格

## 未来优化方向
1. 添加基金名称模糊匹配功能，支持不同平台基金名称的统一处理
2. 集成基金数据API，自动获取和更新基金信息
3. 添加基金类型识别和分类功能
4. 优化基金代码提取算法，提高准确率
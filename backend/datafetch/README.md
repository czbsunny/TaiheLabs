# 基金数据获取模块 (datafetch)

这个模块提供了自定义的基金数据爬取和处理功能，独立于第三方库如akshare，方便用户自行修改和修复可能出现的bug。

## 目录结构

```
datafetch/
├── __init__.py         # 包初始化文件
├── fund_data_fetch.py  # 主要的基金数据获取实现
├── example_usage.py    # 使用示例代码
└── README.md           # 模块说明文档
```

## 功能概述

`fund_data_fetch.py` 模块提供了以下功能：

1. **基金基本信息获取** - 获取基金名称、成立时间、规模等基本信息
2. **基金业绩数据获取** - 获取年度和阶段业绩表现
3. **基金数据分析** - 获取风险收益比、波动率等分析指标
4. **盈利概率分析** - 获取历史任意时点买入的盈利概率数据
5. **交易规则获取** - 获取基金买入/卖出规则和费用信息
6. **基金持仓情况** - 获取基金在特定日期的资产配置情况

## 核心特性

1. **健壮的错误处理** - 添加了全面的异常捕获和错误处理机制
2. **请求重试机制** - 网络请求失败时自动重试
3. **灵活的配置选项** - 可自定义超时时间、重试次数和重试间隔
4. **数据完整性检查** - 确保即使API返回数据不完整也不会导致程序崩溃
5. **兼容akshare的接口设计** - 方便用户从akshare平滑迁移

## 安装依赖

这个模块依赖以下Python库：
- pandas
- requests

你可以使用pip安装这些依赖：

```bash
pip install pandas requests
```

## 基本使用方法

### 导入模块

```python
from datafetch.fund_data_fetch import (
    get_fund_basic_info,
    get_fund_achievement,
    get_fund_analysis,
    get_fund_profit_probability,
    get_fund_trading_rules,
    get_fund_holdings
)
```

### 获取基金基本信息

```python
# 获取单只基金的基本信息
df = get_fund_basic_info("000005")
print(df)
```

### 获取基金业绩数据

```python
df = get_fund_achievement("000005")
print(df)
```

### 获取基金持仓情况

```python
df = get_fund_holdings("002804", "20231231")  # 第二个参数是财报日期，格式为YYYYMMDD
print(df)
```

## 高级使用方法

### 创建自定义配置的实例

```python
from datafetch.fund_data_fetch import FundDataFetcher

# 创建自定义配置的获取器实例
fetcher = FundDataFetcher(
    timeout=15.0,    # 超时时间设为15秒
    retry_count=5,   # 最多重试5次
    retry_delay=1.5  # 重试间隔为1.5秒
)

# 使用自定义实例获取数据
df = fetcher.get_fund_basic_info("000005")
```

### 使用示例处理器类

`example_usage.py` 文件中提供了一个 `FundDataProcessor` 类，它封装了更高级的数据处理功能：

```python
from datafetch.example_usage import FundDataProcessor

processor = FundDataProcessor()

# 处理单只基金数据（包含错误处理）
fund_info = processor.fetch_and_process_fund_basic_info("000005")
print(fund_info)

# 批量处理多只基金数据
fund_codes = ["000005", "000001", "000002"]
batch_results = processor.batch_fetch_fund_info(fund_codes, delay=0.5)
```

## 运行示例

你可以直接运行示例文件来测试功能：

```bash
cd /Users/zhibiaochen/TaiheLabs/backend
python -m datafetch.example_usage
```

## 修复KeyError问题

本模块特别处理了在使用akshare时可能出现的 `KeyError: "['最新规模'] not in index"` 问题。

主要修复措施：
1. 在访问DataFrame的列之前先检查列是否存在
2. 使用 `get()` 方法安全地获取字典值
3. 添加数据完整性检查，确保即使某些字段缺失也能正常运行
4. 提供默认值，避免因缺失字段导致的程序崩溃

## 自定义和扩展

如果你需要修改或扩展这个模块，可以：

1. 在 `fund_data_fetch.py` 文件中添加新的API接口
2. 修改现有的数据处理逻辑
3. 调整请求参数和重试策略
4. 添加新的数据来源

## 注意事项

1. 请遵守相关网站的爬虫规则，不要过度频繁地发送请求
2. 批量请求时建议添加适当的延时，避免触发网站的反爬虫机制
3. 部分API可能会随着网站更新而变化，如有必要请及时更新代码
4. 如果遇到问题，请查看日志输出获取详细的错误信息

## 许可证

本模块仅供学习和研究使用。
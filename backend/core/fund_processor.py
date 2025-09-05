import re
from typing import Optional, Dict, Any, List, Tuple
import logging
import numpy as np
from sqlalchemy.orm import Session
from models.fund import Fund
import difflib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FundProcessor:
    """基金基础信息处理模块
    提供基金代码提取、名称标准化、基金信息查询等基础功能
    """
    
    def __init__(self, db: Optional[Session] = None):
        """初始化基金处理器
        
        参数:
            db: 数据库会话，可选，用于基金信息查询
        """
        self.db = db
        # 定义用于提取基金代码的正则表达式
        self.fund_code_pattern = re.compile(r'[0-9]{6}')
        # 普通数字和带符号数字的正则表达式
        self.num_plain_pattern = re.compile(r'(?<![+\-])\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b(?!%)')
        self.num_signed_pattern = re.compile(r'[+\-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?!%)')
        # 基金名称匹配的阈值设置
        self.exact_match_threshold = 1.0  # 完全匹配阈值
        self.high_match_threshold = 0.8   # 高相似度匹配阈值
        self.low_match_threshold = 0.6    # 低相似度匹配阈值
        
    def extract_fund_code(self, text: str) -> Optional[str]:
        """从文本中提取6位数字的基金代码
        
        参数:
            text: 包含基金代码的文本字符串
        
        返回:
            提取到的6位数字基金代码，如果未找到则返回None
        """
        if not text:
            return None
        
        match = self.fund_code_pattern.search(text)
        if match:
            return match.group(0)
        return None
    
    def normalize_name(self, name: str) -> str:
        """规范化基金名称，去除干扰字符，提高匹配准确性
        
        参数:
            name: 原始基金名称
        
        返回:
            规范化后的基金名称
        """
        if not name:
            return ''
        
        # 去除前后空格
        normalized = name.strip()
        
        # 转换为小写
        normalized = normalized.lower()
        
        # 移除常见的基金名称后缀和特殊字符
        suffixes = [
            '混合型', '股票型', '债券型', '货币市场', 'QDII',
            '指数型', 'ETF联接', 'LOF', 'FOF', '发起式',
            'A类', 'B类', 'C类', 'A', 'B', 'C',
            '前端', '后端', '分红', '净值', '累计',
            '（前端）', '（后端）', '(前端)', '(后端)',
            '人民币', '美元', '欧元', '份额'
        ]
        
        for suffix in suffixes:
            # 处理中英文括号
            suffix_variants = [suffix, f'（{suffix}）', f'({suffix})']
            for variant in suffix_variants:
                normalized = normalized.replace(variant, '')
        
        # 去除所有非中文字符、数字和英文字母
        normalized = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', normalized)
        
        # 去除连续空格
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def first_match(self, pattern: re.Pattern, text: str) -> Optional[str]:
        """查找正则表达式在文本中的第一个匹配
        
        参数:
            pattern: 正则表达式模式
            text: 要搜索的文本
        
        返回:
            第一个匹配的字符串，如果没有匹配则返回None
        """
        m = pattern.search(text.replace(' ', ''))
        return m.group(0) if m else None
    
    def strip_commas(self, value: Any) -> Any:
        """移除数字中的逗号
        
        参数:
            value: 可能包含逗号的数字或字符串
        
        返回:
            移除逗号后的结果
        """
        return value.replace(',', '') if isinstance(value, str) else value
    
    def extract_number(self, text: str, signed: bool = False) -> Optional[str]:
        """从文本中提取数字
        
        参数:
            text: 包含数字的文本
            signed: 是否提取带正负号的数字
        
        返回:
            提取到的数字字符串，如果未找到则返回None
        """
        if not text:
            return None
        
        pattern = self.num_signed_pattern if signed else self.num_plain_pattern
        match = self.first_match(pattern, text)
        if match:
            return self.strip_commas(match)
        return None
    
    def get_fund_info(self, fund_code: str) -> Optional[Fund]:
        """根据基金代码查询基金信息
        
        参数:
            fund_code: 基金代码
        
        返回:
            Fund对象，如果未找到则返回None
        """
        if not self.db or not fund_code:
            return None
        
        return self.db.query(Fund).filter(Fund.fund_code == fund_code).first()
    
    def create_fund(self, fund_code: str, name: str, fund_type: Optional[str] = None) -> Optional[Fund]:
        """创建新的基金记录
        
        参数:
            fund_code: 基金代码
            name: 基金名称
            fund_type: 基金类型，可选
        
        返回:
            创建的Fund对象，如果数据库会话不存在则返回None
        """
        if not self.db or not fund_code or not name:
            return None
        
        # 检查基金是否已存在
        existing_fund = self.get_fund_info(fund_code)
        if existing_fund:
            # 更新现有基金信息
            existing_fund.name = name
            if fund_type:
                existing_fund.fund_type = fund_type
        else:
            # 创建新基金
            existing_fund = Fund(fund_code=fund_code, name=name, fund_type=fund_type)
            self.db.add(existing_fund)
        
        try:
            self.db.commit()
            self.db.refresh(existing_fund)
            return existing_fund
        except Exception as e:
            logger.error(f"Failed to create or update fund: {str(e)}")
            self.db.rollback()
            return None
    
    def parse_fund_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中解析基金信息
        
        参数:
            text: 可能包含基金信息的文本
        
        返回:
            包含基金代码和名称的字典
        """
        result = {
            'name': None,
            'code': None
        }
        
        if not text:
            return result
        
        # 规范化文本
        normalized_text = self.normalize_name(text)
        
        # 提取基金代码
        fund_code = self.extract_fund_code(normalized_text)
        if fund_code:
            # 从文本中移除代码部分
            name_part = normalized_text.replace(fund_code, '').replace('…', '').replace('...', '').strip(' .，,')
            name_part = self.remove_inner_spaces(name_part)
            result['code'] = fund_code
            result['name'] = name_part
        else:
            result['name'] = normalized_text
        
        return result
    
    def remove_inner_spaces(self, text: str) -> str:
        """去掉文本中间的所有空格（保留中文、数字、符号）
        
        参数:
            text: 需要处理的文本
        
        返回:
            处理后的文本
        """
        if not text:
            return ''
        # 去掉首尾空格并删除中间所有空格
        return text.replace(' ', '').strip()
    
    def validate_fund_code(self, fund_code: str) -> bool:
        """验证基金代码是否为有效的6位数字
        
        参数:
            fund_code: 待验证的基金代码
        
        返回:
            验证结果，True表示有效，False表示无效
        """
        if not fund_code:
            return False
        
        return bool(self.fund_code_pattern.fullmatch(fund_code))
        
    def match_fund_by_name(self, fund_name: str, top_n: int = 3) -> List[Tuple[Fund, float]]:
        """通过基金名称匹配最相似的基金
        
        参数:
            fund_name: 基金名称
            top_n: 返回的最相似基金数量
        
        返回:
            包含(Fund对象, 相似度得分)的元组列表，按相似度降序排列
        """
        if not self.db or not fund_name:
            return []
        
        # 规范化输入的基金名称
        normalized_name = self.normalize_name(fund_name)
        
        # 查询所有基金记录（实际应用中可以考虑添加类型过滤等优化）
        all_funds = self.db.query(Fund).all()
        
        # 如果没有基金数据，返回空列表
        if not all_funds:
            return []
        
        # 计算相似度并存储结果
        matched_funds = []
        for fund in all_funds:
            # 规范化数据库中的基金名称
            db_fund_name = self.normalize_name(fund.name)
            
            # 计算字符串相似度（使用difflib的SequenceMatcher）
            similarity = difflib.SequenceMatcher(None, normalized_name, db_fund_name).ratio()
            
            # 添加到结果列表
            matched_funds.append((fund, similarity))
        
        # 按相似度降序排序
        matched_funds.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前top_n个结果
        return matched_funds[:top_n]
        
    def find_best_matching_fund(self, fund_name: str) -> Optional[Tuple[Fund, float]]:
        """查找与给定基金名称最匹配的基金
        
        参数:
            fund_name: 基金名称
        
        返回:
            包含(Fund对象, 相似度得分)的元组，如果没有找到足够相似的基金则返回None
        """
        matched_funds = self.match_fund_by_name(fund_name, top_n=1)
        
        # 检查是否有足够相似的基金
        if matched_funds and matched_funds[0][1] >= self.low_match_threshold:
            return matched_funds[0]
        
        return None
        
    def enhance_fund_info_from_name(self, fund_name: str) -> Optional[Dict[str, Any]]:
        """通过基金名称增强基金信息，添加匹配到的基金代码等
        
        参数:
            fund_name: 基金名称
        
        返回:
            包含基金信息的字典，如果匹配失败则返回None
        """
        # 尝试直接提取基金代码
        fund_code = self.extract_fund_code(fund_name)
        if fund_code:
            # 如果名称中包含代码，直接查询该代码的基金信息
            fund_info = self.get_fund_info(fund_code)
            if fund_info:
                return {
                    'code': fund_code,
                    'name': fund_info.name,
                    'fund_type': fund_info.fund_type,
                    'match_type': 'exact_code',
                    'match_score': 1.0
                }
        
        # 通过名称匹配查找基金
        best_match = self.find_best_matching_fund(fund_name)
        if best_match:
            fund, score = best_match
            match_type = 'exact_name' if score >= self.exact_match_threshold else \
                         'high_similarity' if score >= self.high_match_threshold else 'low_similarity'
            
            return {
                'code': fund.fund_code,
                'name': fund.name,
                'fund_type': fund.fund_type,
                'match_type': match_type,
                'match_score': round(score, 3)
            }
        
        return None
    
    def create_or_update_fund(self, fund_code: str, fund_data: Dict[str, Any]) -> Optional[Fund]:
        """创建或更新包含所有字段的基金信息
        
        参数:
            fund_code: 基金代码
            fund_data: 包含基金所有字段信息的字典
        
        返回:
            创建或更新的Fund对象，如果失败则返回None
        """
        if not self.db or not fund_code or not fund_data:
            return None
        
        try:
            # 检查基金是否已存在
            existing_fund = self.get_fund_info(fund_code)
            if not existing_fund:
                # 创建新基金
                existing_fund = Fund(fund_code=fund_code)
                self.db.add(existing_fund)
            
            # 更新基金信息
            for key, value in fund_data.items():
                if hasattr(existing_fund, key):
                    setattr(existing_fund, key, value)
            
            self.db.commit()
            self.db.refresh(existing_fund)
            return existing_fund
        except Exception as e:
            logger.error(f"Failed to create or update complete fund info: {str(e)}")
            self.db.rollback()
            return None
            
    def batch_enhance_fund_info(self, fund_names: List[str]) -> List[Dict[str, Any]]:
        """批量通过基金名称增强基金信息
        
        参数:
            fund_names: 基金名称列表
        
        返回:
            包含每个基金增强信息的字典列表
        """
        results = []
        
        for name in fund_names:
            enhanced_info = self.enhance_fund_info_from_name(name)
            if enhanced_info:
                results.append({
                    'original_name': name,
                    **enhanced_info
                })
            else:
                results.append({
                    'original_name': name,
                    'name': name,
                    'code': None,
                    'match_type': 'no_match',
                    'match_score': 0.0
                })
        
        return results
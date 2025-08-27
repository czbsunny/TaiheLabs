# OCR处理器 - 集中管理OCR识别的核心功能
import cv2
import numpy as np
import easyocr
import re
import logging
import os
import tempfile
import datetime
from typing import List, Dict, Any, Optional, Tuple
from fastapi import UploadFile

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRHandler:
    def __init__(self, languages: List[str] = ['ch_sim', 'en'], gpu: bool = False):
        """初始化OCR处理器
        
        参数:
            languages: 识别的语言列表，默认为中文简体和英文
            gpu: 是否使用GPU加速，默认为False
        """
        self.reader = easyocr.Reader(languages, gpu=gpu)
        logger.info(f"OCR处理器初始化完成，语言: {languages}, GPU: {gpu}")
    
    def ensure_all_native_types(self, data: Any) -> Any:
        """递归确保所有数据都是可JSON序列化的Python原生类型
        
        参数:
            data: 需要转换的数据（可以是dict、list、tuple或单个值）
        
        返回:
            转换后的Python原生类型数据
        """
        # 处理字典
        if isinstance(data, dict):
            return {key: self.ensure_all_native_types(value) for key, value in data.items()}
        # 处理列表
        elif isinstance(data, list):
            return [self.ensure_all_native_types(item) for item in data]
        # 处理元组
        elif isinstance(data, tuple):
            return tuple(self.ensure_all_native_types(item) for item in data)
        # 处理None值
        elif data is None:
            return ""
        # 处理numpy整数和浮点数
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()
        # 处理numpy数组
        elif isinstance(data, np.ndarray):
            return data.tolist()
        # 处理numpy其他类型
        elif hasattr(data, 'item'):
            return data.item()
        # 其他类型保持不变，但确保是Python原生类型
        else:
            if isinstance(data, (str, int, float, bool)):
                return data
            try:
                # 尝试转换为字符串
                return str(data)
            except:
                return "[不可转换]"
    
    def normalize_ocr_result(self, ocr_out: List[Any]) -> List[Dict[str, Any]]:
        """统一OCR输出格式
        
        接受 easyocr 的原始输出 [(bbox, text, score), ...] 或已经是 [{'text','bbox'}]，统一转成标准结构
        
        参数:
            ocr_out: OCR原始输出结果
        
        返回:
            标准化的OCR结果列表
        """
        norm = []
        for r in ocr_out:
            if isinstance(r, dict) and 'text' in r and 'bbox' in r:
                pts = r['bbox']
                if not isinstance(pts, np.ndarray):
                    pts = np.array(pts, dtype=np.float32)
                norm.append({'text': str(r['text']).strip(), 'bbox': pts.astype(int)})
            else:
                # 兼容 easyocr 三元组
                try:
                    bbox, text, score = r
                except:
                    continue
                if text is None:
                    continue
                t = str(text).strip()
                if not t:
                    continue
                pts = np.array(bbox, dtype=np.float32)
                norm.append({'text': t, 'bbox': pts.astype(int)})
        return norm
    
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
    
    def cluster_lines_by_y(self, ocr_struct: List[Dict[str, Any]], min_gap: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """将OCR结果按y坐标聚类成行
        
        参数:
            ocr_struct: 标准化的OCR结果列表
            min_gap: 最小行间距，默认为None（自动计算）
        
        返回:
            按行分组的OCR结果
        """
        if not ocr_struct:
            return []
        heights = [it['bbox'][:, 1].max() - it['bbox'][:, 1].min() for it in ocr_struct]
        med_h = np.median(heights) if heights else 12
        gap = max(8, int(med_h * 0.8))
        if min_gap is not None:
            gap = min_gap

        items = []
        for it in ocr_struct:
            # 先规范化文本
            it['text'] = self.remove_inner_spaces(it['text'])
            yc = int(np.mean(it['bbox'][:, 1]))
            items.append((yc, it))

        items.sort(key=lambda x: x[0])

        lines = []
        current = []
        prev_y = None
        for yc, it in items:
            if prev_y is None or (yc - prev_y) <= gap:
                current.append(it)
            else:
                lines.append(current)
                current = [it]
            prev_y = yc
        if current:
            lines.append(current)

        # 每行内按 x 从左到右
        for line in lines:
            line.sort(key=lambda it: int(np.mean(it['bbox'][:, 0])))
        return lines
    
    def vertical_blocks_with_position(self, roi: np.ndarray, bin_thresh: int = 200, padding: int = 5) -> List[Tuple[int, int, int, int]]:
        """获取图像中的垂直分块位置
        
        参数:
            roi: 感兴趣区域图像
            bin_thresh: 二值化阈值
            padding: 边界填充像素数
        
        返回:
            垂直分块的位置列表 [(left, top, right, bottom), ...]
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY_INV)
        v_proj = np.sum(binary, axis=0)
        cols = np.where(v_proj > 0)[0]
        h, w = roi.shape[:2]

        if len(cols) == 0:
            return []
        col_splits, start = [], cols[0]
        for i in range(1, len(cols)):
            if cols[i] - cols[i-1] > w * 0.05:
                col_splits.append((start, cols[i-1]))
                start = cols[i]
        col_splits.append((start, cols[-1]))

        blocks = []
        for l, r in col_splits:
            left = max(0, l - padding)
            right = min(w - 1, r + padding)
            blocks.append((left, 0, right, h - 1))
        return blocks
    
    def first_match(self, pat: re.Pattern, s: str) -> Optional[str]:
        """查找正则表达式在文本中的第一个匹配
        
        参数:
            pat: 正则表达式模式
            s: 要搜索的文本
        
        返回:
            第一个匹配的字符串，如果没有匹配则返回None
        """
        m = pat.search(s.replace(' ', ''))
        return m.group(0) if m else None
    
    def strip_commas(self, x: Any) -> Any:
        """移除数字中的逗号"""
        return x.replace(',', '') if isinstance(x, str) else x
    
    def normalize_name(self, name: str) -> str:
        """规范化基金名称的空格与省略号等"""
        if not name:
            return name

        s = str(name).strip()

        # 合并任意空白为单个空格
        s = re.sub(r'\s+', ' ', s)

        # 统计中文字符比例（常用基本汉字范围）
        cjk_chars = re.findall(r'[\u4e00-\u9fff]', s)
        cjk_ratio = len(cjk_chars) / max(1, len(s))

        if cjk_ratio > 0.4:
            # 中文为主：去掉所有空格（避免 OCR 在汉字间插入空格）
            s = re.sub(r'\s+', '', s)
        else:
            # 混合或英文为主：仅去掉汉字之间的空格，保留英文单词间空格
            s = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', s)
            # 再合并多空格（以防）
            s = re.sub(r'\s+', ' ', s)

        # 统一省略号：多个点或中文省略号规范为单个 '…'
        s = re.sub(r'\.{2,}', '…', s)          # "..." -> "…"
        s = re.sub(r'\s*…\s*', '…', s)         # 去掉省略号周围空格

        # 去除开头/结尾的多余标点或空格
        s = s.strip(' .，,;；')

        # 如果末尾跟了 6 位数字（code），把名字和 code 之间的空格去掉
        s = re.sub(r'\s+(\d{6})$', r'\1', s)

        return s
    
    # ---------------- 基金类型判断方法 ----------------
    def is_alipay_block(self, ocr_struct: List[Dict[str, Any]]) -> bool:
        """判断OCR结果是否是支付宝基金截图"""
        lines = self.cluster_lines_by_y(ocr_struct)
        if len(lines) < 3:
            return False

        # 取最后4行，排除顶部干扰
        lines = lines[-4:]

        # 第1行必须只有一个文本框
        if len(lines[0]) != 1:
            return False

        # 第3行必须有至少3个文本框（持仓金额、昨日收益、持仓收益）
        if len(lines[2]) < 3:
            return False

        # 最后一行是否包含"占比"
        last_line_text = ' '.join([it['text'] for it in lines[-1]])
        if '占比' not in last_line_text:
            return False

        return True
    
    def is_alipay_simple_block(self, ocr_struct: List[Dict[str, Any]]) -> bool:
        """判断OCR结果是否是支付宝简版基金截图"""
        lines = self.cluster_lines_by_y(ocr_struct)
        if len(lines) < 3:
            return False

        # 取最后3行，排除顶部干扰
        lines = lines[-3:]

        # 第1行必须是两个文本框
        if len(lines[0]) > 2:
            return False

        line2 = [it['text'] for it in lines[1]]

        if not all(k in line2 for k in ['金额', '昨日收益', '持有收益']):
            return False

        # 第3行必须有至少3个文本框（持仓金额、昨日收益、持仓收益）
        if len(lines[2]) < 3:
            return False

        return True
    
    def is_tiantian_block(self, ocr_struct: List[Dict[str, Any]]) -> bool:
        """判断OCR结果是否是天天基金截图"""
        text_all = ' '.join([it['text'] for it in ocr_struct])
        return ('资产' in text_all) and ('昨日收益' in text_all) and ('持仓收益' in text_all)
    
    def detect_screenshot_type(self, lines: List[str], roi: np.ndarray, ocr_struct: List[Dict[str, Any]]) -> str:
        """根据文字内容判断截图类型"""
        lines_text = " ".join(lines)  # 合并成一行方便判断

        # 支付宝特征
        if self.is_alipay_block(ocr_struct):
            return "支付宝"

        if self.is_alipay_simple_block(ocr_struct):
            return '支付宝简版'
        
        # 天天基金特征
        if all(k in lines_text for k in ['资产', '昨日收益', '持仓收益']):
            return '天天基金'

        lines_clustered = self.cluster_lines_by_y(ocr_struct)
        blocks = self.vertical_blocks_with_position(roi)
        if len(blocks) == 3 and len(lines_clustered) <= 2:
            return "雪球"

        # 如果不匹配上述，可以返回未知
        return '其他'
    
    # ---------------- 三列边界构建方法 ----------------
    def build_three_column_boundaries_from_keywords(self, keyword_line_items: List[Dict[str, Any]], roi_width: int) -> List[Tuple[int, int]]:
        """从关键字行构建三列边界"""
        def has(s, k): return k in s
        # 可能被 OCR 切成多段：合并同一关键词的多个小框的 x 中心
        key_centers = {'资产': [], '昨日收益': [], '持仓收益': []}
        for it in keyword_line_items:
            t = it['text']
            xc = int(np.mean(it['bbox'][:, 0]))
            if has(t, '资产'):
                key_centers['资产'].append(xc)
            if has(t, '昨日收益'):
                key_centers['昨日收益'].append(xc)
            if '持仓收益' in t:  # "持仓收益/率" 也算
                key_centers['持仓收益'].append(xc)

        centers = []
        for k in ['资产', '昨日收益', '持仓收益']:
            if key_centers[k]:
                centers.append(int(np.median(key_centers[k])))
        # 不足 3 个时，退化为均分
        if len(centers) < 3:
            thirds = [roi_width // 3, 2 * roi_width // 3]
            centers = [thirds[0]//2, thirds[0]+(thirds[1]-thirds[0])//2, thirds[1]+(roi_width-thirds[1])//2]
        centers.sort()
        c1, c2, c3 = centers[:3]
        # 以中心点的中点作为分界
        b1 = (c1 + c2) // 2
        b2 = (c2 + c3) // 2
        return [(0, b1), (b1, b2), (b2, roi_width-1)]
    
    # ---------------- 信息提取方法 ----------------
    def extract_alipay_block_info(self, roi: np.ndarray, ocr_struct: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """提取支付宝基金截图中的信息"""
        # 聚类按行
        lines = self.cluster_lines_by_y(ocr_struct)

        if len(lines) < 3:  # 行数太少，排除
            return None

        # 只取最后 4 行
        lines = lines[-4:]

        # 第1行：基金名称，必须只有一个文本框
        if len(lines[0]) != 1:
            return None
        name = lines[0][0]['text']
        roi_w = roi.shape[1]
        # 第3行：数值行
        value_line = lines[2]
        n_cols = 4  # 等比例划分
        col_width = roi_w / n_cols
        col_ranges = [(int(i*col_width), int((i+1)*col_width)) for i in range(n_cols)]

        def which_col(xc):
            for idx, (L,R) in enumerate(col_ranges):
                if L <= xc <= R:
                    return idx
            return int(np.argmin([abs(xc - (L+R)//2) for (L,R) in col_ranges]))

        # 初始化
        hold_amount = hold_profit = None

        for it in value_line:
            xc = int(np.mean(it['bbox'][:,0]))
            idx = which_col(xc)
            text = it['text']
            if idx == 0:
                hold_amount = text
            elif idx == 2:
                hold_profit = text

        return {
            'name': name,
            'amount': hold_amount,
            'profit': hold_profit,
        }
    
    def extract_alipay_simple_block_info(self, roi: np.ndarray, ocr_struct: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """提取支付宝简版基金截图中的信息"""
        if not ocr_struct:
            return None

        lines = self.cluster_lines_by_y(ocr_struct)
        if len(lines) < 3:
            return None

        # 判断倒数第四行是否是"今日收益更新"，如果是则忽略
        fourth_last_line = lines[-4] if len(lines) >= 4 else None
        if fourth_last_line:
            texts = [t['text'].replace(' ', '') for t in fourth_last_line]
            if any('今日收益更新' in t for t in texts):
                lines_to_use = lines[-3:]
            else:
                lines_to_use = lines[-3:]
        else:
            lines_to_use = lines[-3:]

        # 倒数第三行第一个文本框是基金名称
        third_last_line = lines_to_use[0]
        name = third_last_line[0]['text'].replace(' ', '')

        # 最后一行是持仓金额、昨日收益、持有收益
        last_line = lines_to_use[-1]

        # ROI宽度用于水平分列
        roi_w = max(it['bbox'][:,0].max() for it in ocr_struct)
        one_third = roi_w / 3
        col_ranges = [(0, one_third), (one_third, 2*one_third), (2*one_third, roi_w)]

        def which_col(xc):
            for idx, (L, R) in enumerate(col_ranges):
                if L <= xc <= R:
                    return idx
            centers = [(L+R)/2 for (L,R) in col_ranges]
            return int(np.argmin([abs(xc - c) for c in centers]))

        col_texts = {0: [], 1: [], 2: []}
        for it in last_line:
            xc = int(np.mean(it['bbox'][:,0]))
            col_texts[which_col(xc)].append(it['text'].replace(' ', ''))

        amount = col_texts[0][0] if col_texts[0] else None
        hold_profit = col_texts[2][0] if col_texts[2] else None

        return {
            'name': name,
            'amount': amount,
            'profit': hold_profit
        }
    
    def extract_tiantianfund_block_info(self, roi: np.ndarray, ocr_out: List[Any]) -> Optional[Dict[str, Any]]:
        """提取天天基金截图中的信息"""
        ocr_struct = self.normalize_ocr_result(ocr_out)
        if not ocr_struct:
            return None

        lines = self.cluster_lines_by_y(ocr_struct)
        roi_w = roi.shape[1]

        # 1) 第一行：名称 + 6位代码（名称可能含 ... 或 …）
        name, code = None, None
        if lines:
            line1 = lines[0]
            line1_text = ' '.join([it['text'] for it in line1])
            m = self._fund_code.search(line1_text)
            if m:
                code = m.group(0)
                name = line1_text.replace(code, '').replace('…', '').replace('...', '').strip(' .，,')
                name = self.remove_inner_spaces(name)
            else:
                # 没抓到代码，也记录名称
                name = line1_text.replace('…', '').replace('...', '').strip(' .，,')
                name = self.remove_inner_spaces(name)

        # 2) 第二行：关键词所在行（资产/昨日收益/持仓收益/率）
        kw_line_idx = None
        for i, line in enumerate(lines[:4]):  # 通常在前几行
            joined = ' '.join([it['text'] for it in line])
            if ('资产' in joined) and ('昨日收益' in joined) and ('持仓收益' in joined):
                kw_line_idx = i
                break

        # 列边界
        if kw_line_idx is not None:
            col_ranges = self.build_three_column_boundaries_from_keywords(lines[kw_line_idx], roi_w)
        else:
            # 兜底：等宽三列
            one = roi_w // 3
            col_ranges = [(0, one), (one, 2*one), (2*one, roi_w-1)]

        def which_col(xc):
            for idx, (L, R) in enumerate(col_ranges):
                if L <= xc <= R:
                    return idx
            # 不在范围就选最近的
            centers = [ (L+R)//2 for (L,R) in col_ranges ]
            return int(np.argmin([abs(xc - c) for c in centers]))

        # 3) 第三行：数值（与第二行对齐）
        amount = hold_profit = None
        third_line = lines[kw_line_idx+1] if (kw_line_idx is not None and kw_line_idx+1 < len(lines)) else []
        # 先把第三行的文本按列分发
        col_texts = {0: [], 1: [], 2: []}
        for it in third_line:
            xc = int(np.mean(it['bbox'][:,0]))
            col_texts[which_col(xc)].append(it['text'])

        # 资产：优先取"非百分比、无正负"的数值
        cand_amount = None
        for t in col_texts[0]:
            cand_amount = self.first_match(self._num_plain, t)
            if cand_amount: break
        amount = self.strip_commas(cand_amount) if cand_amount else None

        # 持仓收益：优先取"带正负"的数值
        cand_h = None
        for t in col_texts[2]:
            cand_h = self.first_match(self._num_signed, t)
            if cand_h: break
        hold_profit = self.strip_commas(cand_h) if cand_h else None

        return {
            'name': name,
            'code': code,
            'amount': amount,
            'profit': hold_profit
        }
    
    def extract_xueqiu_block_info(self, roi: np.ndarray, ocr_result: List[Any]) -> Optional[Dict[str, Any]]:
        """提取雪球基金截图中的信息"""
        v_blocks = self.vertical_blocks_with_position(roi)
        if len(v_blocks) != 3:
            return None

        # 分配文本到三个垂直块
        block_texts = [[], [], []]
        for item in ocr_result:  # 正确处理readtext返回的元组格式
            # 计算bbox的中心点x坐标
            x_center = np.mean([point[0] for point in item["bbox"]])
            for i, (l, t, r, b) in enumerate(v_blocks):
                if l <= x_center <= r:
                    block_texts[i].append(item["text"])
                    break

        # 提取名称、金额、收益
        first_block_lines = block_texts[0]
        if not first_block_lines:
            return None
            
        # 第一块包含基金名称和持仓金额
        # 假设最后一行是金额，前面的是基金名称
        if len(first_block_lines) < 2:
            name = " ".join(first_block_lines)
            amount = None
        else:
            name = " ".join(first_block_lines[:-1])
            # 从最后一行提取金额
            amount_match = self._num_plain.search(first_block_lines[-1])
            amount = amount_match.group() if amount_match else None

        # 第三块包含持仓收益和收益率
        third_block_lines = block_texts[2]
        profit = None
        profit_rate = None
        
        for line in third_block_lines:
            # 提取持仓收益（带正负号的数值）
            if not profit:
                profit_match = self._num_signed.search(line)
                if profit_match:
                    profit = profit_match.group()

        return {
            'name': name,
            'amount': amount,
            'profit': profit,
        }
    
    # ---------------- 核心处理方法 ----------------
    def extract_text_blocks(self, image_path: str) -> List[List[Tuple[int, int, int, int]]]:
        """提取图像中的文本块，使用连通域分析和水平分块"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        min_area, max_area = 50, 5000
        text_blocks = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if min_area < area < max_area:
                text_blocks.append((x, y, w, h))
        text_blocks = sorted(text_blocks, key=lambda b: b[1])

        # 水平分块逻辑
        block_interval = 60
        blocks = []
        current_block = []
        prev_line_bottom = 0
        for b in text_blocks:
            x, y, w, h = b
            line_top = y
            line_bottom = y + h
            interval = line_top - prev_line_bottom if prev_line_bottom else 0
            if not current_block:
                current_block.append(b)
            else:
                if interval > block_interval:
                    blocks.append(current_block)
                    current_block = [b]
                else:
                    current_block.append(b)
            prev_line_bottom = line_bottom
        if current_block:
            blocks.append(current_block)
            
        return blocks
    
    def process_block(self, img: np.ndarray, block: List[Tuple[int, int, int, int]], save_blocks: bool = False, blocks_dir: str = None) -> Dict[str, Any]:
        """处理单个文本块，进行OCR识别和信息提取"""
        # 计算块的边界
        block_top = max(min(b[1] for b in block)-8, 0)  # 添加一些边距
        block_bottom = min(max(b[1]+b[3] for b in block)+8, img.shape[0])  # 添加一些边距
        block_left = 0
        block_right = img.shape[1]

        # 提取ROI
        roi = img[block_top:block_bottom, block_left:block_right]

        # OCR识别
        result = self.reader.readtext(roi)
        ocr_struct = self.normalize_ocr_result(result)
        
        # 初始化block_info为None
        block_info = None
        source = None
        
        # 提取文本行
        lines = [text for (_, text, _) in result if text.strip()]
        
        # 判断是否包含数字特征
        has_plain_number = False
        has_signed_number = False
        for line in lines:
            if re.fullmatch(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', line):
                has_plain_number = True
            if re.fullmatch(r'[+\-]\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b(?!%)', line):
                has_signed_number = True

        # 只有当同时包含普通数字和带符号数字时才进行后续处理
        if has_plain_number and has_signed_number:
            # 判断来源类型
            source = self.detect_screenshot_type(lines, roi, ocr_struct)
            
            # 根据类型提取信息
            if source == "支付宝":
                block_info = self.extract_alipay_block_info(roi, ocr_struct)
            elif source == "支付宝简版":
                block_info = self.extract_alipay_simple_block_info(roi, ocr_struct)
            elif source == "天天基金":
                # 准备OCR原始输出格式
                raw_ocr_results = []
                for item in ocr_struct:
                    raw_ocr_results.append({
                        "bbox": item['bbox'],
                        "text": item['text']
                    })
                block_info = self.extract_tiantianfund_block_info(roi, raw_ocr_results)
            elif source == "雪球":
                # 准备OCR原始输出格式
                raw_ocr_results = []
                for item in ocr_struct:
                    raw_ocr_results.append({
                        "bbox": item['bbox'],
                        "text": item['text']
                    })
                block_info = self.extract_xueqiu_block_info(roi, raw_ocr_results)
            
            if block_info:
                # 规范化基金名称
                if 'name' in block_info and block_info['name']:
                    block_info['name'] = self.normalize_name(block_info['name'])
                block_info['type'] = source
                
        # 保存ROI图像
        block_image_path = None
        if save_blocks and blocks_dir and roi.size > 0:
            try:
                # 生成唯一的文件名
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                block_image_path = os.path.join(blocks_dir, f'block_roi_{timestamp}.png')
                cv2.imwrite(block_image_path, roi)
                logger.info(f"已保存文本块图像到: {block_image_path}")
            except Exception as e:
                logger.error(f"保存文本块图像失败: {str(e)}")
        
        return {
            'has_financial_data': has_plain_number and has_signed_number,
            'block_info': block_info,
            'source': source,
            'roi': roi,
            'ocr_result': result,
            'block_image_path': block_image_path
        }
    
    async def recognize_image(self, file: UploadFile, save_blocks: bool = False) -> Dict[str, Any]:
        """异步处理上传的图片文件，进行OCR识别和信息提取
        
        Args:
            file: 上传的图片文件
            save_blocks: 是否保存提取的文本块图像
        
        Returns:
            包含识别结果的字典
        """
        try:
            # 检查文件类型
            content_type = file.content_type
            if content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
                raise ValueError("请上传有效的图片文件")
            
            # 保存上传的临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(await file.read())
                temp_file_path = temp_file.name
            
            logger.info(f"已保存临时图片文件: {temp_file_path}")
            
            # 创建保存文本块的目录
            blocks_dir = None
            if save_blocks:
                blocks_dir = os.path.join(os.path.dirname(temp_file_path), 'ocr_blocks')
                os.makedirs(blocks_dir, exist_ok=True)
                logger.info(f"创建文本块保存目录: {blocks_dir}")
            
            try:
                # 读取原始图片
                img = cv2.imread(temp_file_path)
                if img is None:
                    raise Exception("无法读取图片内容")
                
                # 提取文本块
                blocks = self.extract_text_blocks(temp_file_path)
                
                # 处理每个文本块并提取基金持仓信息
                portfolios = []
                saved_blocks_info = []
                
                for idx, block in enumerate(blocks):
                    processed_block = self.process_block(img, block, save_blocks, blocks_dir)
                    
                    if processed_block['has_financial_data'] and processed_block['block_info']:
                        portfolios.append(processed_block['block_info'])
                    
                    # 收集保存的文本块信息
                    if save_blocks and processed_block['block_image_path']:
                        saved_blocks_info.append({
                            'block_index': idx,
                            'image_path': processed_block['block_image_path'],
                            'has_financial_data': processed_block['has_financial_data'],
                            'source': processed_block['source']
                        })
                
                # 构建结构化数据
                structured_data = {
                    'total_portfolios': len(portfolios),
                    'portfolios': portfolios
                }
                
                # 构建最终结果
                result = {
                    "status": "success",
                    "message": "图片识别成功",
                    "structured_data": structured_data,
                    "content_blocks_count": len(blocks)
                }
                
                # 如果保存了文本块，添加相关信息
                if save_blocks:
                    result['saved_blocks_info'] = saved_blocks_info
                    result['blocks_directory'] = blocks_dir
                
                # 确保所有数据都是Python原生类型
                return self.ensure_all_native_types(result)
            finally:
                # 清理临时文件
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    logger.info(f"已删除临时图片文件: {temp_file_path}")
        except Exception as e:
            logger.error(f"图片识别失败: {str(e)}")
            return {
                "status": "error",
                "message": f"图片识别失败: {str(e)}"
            }
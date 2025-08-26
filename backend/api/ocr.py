from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import easyocr
import logging
from typing import List, Dict, Any
import cv2
import numpy as np
import os
import tempfile

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建OCR路由器
ocr_router = APIRouter()

reader = easyocr.Reader(['ch_sim', 'en'], gpu=False) 

def ensure_all_native_types(data):
    """
    递归确保所有数据都是可JSON序列化的Python原生类型
    将numpy类型转换为Python原生类型，None转换为空字符串
    
    参数:
        data: 需要转换的数据（可以是dict、list、tuple或单个值）
    
    返回:
        转换后的Python原生类型数据
    """
    # 处理字典
    if isinstance(data, dict):
        return {key: ensure_all_native_types(value) for key, value in data.items()}
    # 处理列表
    elif isinstance(data, list):
        return [ensure_all_native_types(item) for item in data]
    # 处理元组
    elif isinstance(data, tuple):
        return tuple(ensure_all_native_types(item) for item in data)
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

@ocr_router.post("/recognize-portfolio")
async def recognize_portfolio(
    file: UploadFile = File(...),
    save_blocks: bool = True
):
    """
    识别图片中的持仓组合内容，使用分块处理提高准确性
    
    参数:
        file: 包含持仓组合的图片文件
        save_blocks: 是否保存检测到的内容块为临时文件
    
    返回:
        识别出的持仓组合文本内容
    """
    try:
        # 检查文件类型
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="请上传有效的图片文件")
        
        # 保存上传的临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        logger.info(f"已保存临时图片文件: {temp_file_path}")
        
        try:
            # 读取原始图片用于分块处理
            original_image = cv2.imread(temp_file_path)
            if original_image is None:
                raise Exception("无法读取图片内容")
            
            # 检测内容块
            blocks_result = detect_content_blocks(temp_file_path, save_blocks=save_blocks)
            
            # 初始化总的识别文本结果
            recognized_texts = []
            saved_blocks_info = None
            blocks_dir = None
            content_blocks = []
            
            # 处理detect_content_blocks的返回结果
            if isinstance(blocks_result, dict):
                # 如果返回的是字典，说明保存了内容块
                content_blocks = blocks_result['blocks']
                saved_blocks_info = blocks_result['saved_blocks_info']
                blocks_dir = blocks_result['blocks_dir']
            else:
                # 传统的返回格式
                content_blocks = blocks_result
            
            # 如果成功检测到内容块，按块进行OCR识别
            if content_blocks:
                logger.info(f"成功检测到 {len(content_blocks)} 个内容块")
                
                # 按块识别文本
                for block_idx, block in enumerate(content_blocks):
                    block_texts = recognize_block_text(reader, original_image, block)
                    
                    # 为每个文本添加块信息
                    for text_item in block_texts:
                        text_item["block_index"] = block_idx
                        recognized_texts.append(text_item)
            else:
                # 如果没有检测到内容块，使用传统的整体OCR识别
                logger.info("未检测到明显的内容块，使用整体OCR识别")
                results = reader.readtext(temp_file_path)
                
                # 处理识别结果
                for (bbox, text, prob) in results:
                    recognized_texts.append({
                        "text": text,
                        "confidence": float(prob),  # 转换为Python原生float
                        "bbox": bbox,
                        "block_index": -1  # 表示整体识别
                    })
            
            # 尝试提取表格结构化数据
            portfolio_data = extract_portfolio_data(recognized_texts)
            
            # 增强结果，添加分块信息
            enhanced_result = {
                "status": "success",
                "message": "持仓组合识别成功",
                "raw_texts": recognized_texts,
                "structured_data": portfolio_data,
                "content_blocks_count": len(content_blocks) if content_blocks else 0
            }
            
            # 如果保存了内容块，添加相关信息到结果
            if saved_blocks_info:
                enhanced_result['saved_blocks_info'] = saved_blocks_info
                enhanced_result['blocks_dir'] = blocks_dir
                logger.info(f"内容块图片已保存到: {blocks_dir}")
            
            # 最终类型检查：确保所有数据都是Python原生类型
            print("===== 最终JSON响应前类型检查 =====")
            enhanced_result = ensure_all_native_types(enhanced_result)
            print("===== 最终类型检查完成 =====")
            
            return JSONResponse(enhanced_result)
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"已删除临时图片文件: {temp_file_path}")
    
    except Exception as e:
        logger.error(f"OCR识别失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR识别失败: {str(e)}")

def detect_content_blocks(image_path, save_blocks=True):
    """
    检测图片中的内容块，优化以适应三种不同风格的基金截图：
    1. 水平分隔线分块（图1）
    2. 框框分块（图2）
    3. 空白区域分块（图3）
    确保每个块与图片同宽并包含完整的基金数据
    
    参数:
        image_path: 图片路径
        save_blocks: 是否保存检测到的内容块为单独的图片文件
    
    返回:
        内容块的坐标列表，按从上到下的顺序排列
    """
    try:
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"无法读取图片: {image_path}")
            return []
        
        # 获取图片尺寸
        height, width = image.shape[:2]
        
        # 创建保存内容块的目录
        blocks_dir = None
        if save_blocks:
            blocks_dir = tempfile.mkdtemp(prefix="ocr_blocks_")
            logger.info(f"将在目录 {blocks_dir} 中保存内容块图片")
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值处理 - 调整参数以适应不同类型的基金截图
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 3)
        
        # 形态学操作，增强水平线条和垂直框线
        # 首先增强水平线条
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 20, 1))
        horizontal_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # 然后增强垂直线条
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 20))
        vertical_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel)
        
        # 结合水平和垂直线条信息
        combined = cv2.bitwise_or(horizontal_closed, vertical_closed)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选出合适大小的轮廓作为内容块
        blocks = []
        
        # 初始化标记图片用于可视化
        marked_image = image.copy()
        
        # 针对三种不同风格的基金截图进行专门处理
        # 1. 检测水平分隔线（风格1）
        horizontal_lines = []
        min_line_length = width // 2  # 线条长度至少为图片宽度的1/2
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # 如果轮廓的高度很小但宽度很大，认为是水平分隔线
            if h < 10 and w > min_line_length:
                horizontal_lines.append(y)
                # 在标记图片上绘制水平分隔线
                cv2.line(marked_image, (x, y), (x + w, y), (0, 0, 255), 2)
        
        # 如果检测到足够的水平分隔线，基于分隔线进行分块（风格1）
        if len(horizontal_lines) >= 2:
            logger.info(f"检测到 {len(horizontal_lines)} 条水平分隔线，使用分隔线分块策略")
            # 排序分隔线
            horizontal_lines.sort()
            
            # 在顶部和底部添加边界线
            horizontal_lines.insert(0, 0)
            horizontal_lines.append(height)
            
            # 根据分隔线创建块，确保每个块与图片同宽
            for i in range(len(horizontal_lines) - 1):
                top = horizontal_lines[i]
                bottom = horizontal_lines[i + 1]
                # 确保块的高度足够大（至少30像素）
                if bottom - top > 30:
                    blocks.append((0, top, width, bottom - top))
        else:
            # 2. 检测框框分块（风格2）
            logger.info("未检测到足够的水平分隔线，尝试检测框框分块")
            # 寻找矩形框
            rectangles = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # 筛选宽高比较合适的矩形框，宽度至少为图片宽度的80%
                if w > width * 0.8 and h > 30 and h < height * 0.5:
                    rectangles.append((x, y, w, h))
                    # 在标记图片上绘制矩形框
                    cv2.rectangle(marked_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # 如果检测到框框
            if len(rectangles) >= 2:
                logger.info(f"检测到 {len(rectangles)} 个框框，使用框框分块策略")
                # 根据y坐标排序
                rectangles.sort(key=lambda b: b[1])
                
                # 调整块的宽度为图片宽度
                for (x, y, w, h) in rectangles:
                    blocks.append((0, y, width, h))
            else:
                # 3. 空白区域分块（风格3）
                logger.info("未检测到框框，使用空白区域分块策略")
                # 计算合适的块高度阈值，基于图片内容估算
                # 分析灰度图的垂直投影，找到空白区域
                vertical_projection = np.sum(thresh, axis=1)
                
                # 寻找空白区域的中心点
                blank_areas = []
                min_blank_size = 25  # 降低最小空白区域高度，检测更多分块
                current_blank = 0
                for i, val in enumerate(vertical_projection):
                    if val < 800:  # 降低阈值，提高对空白区域的敏感度
                        current_blank += 1
                    else:
                        if current_blank >= min_blank_size:
                            # 记录空白区域的中心点
                            blank_center = i - current_blank // 2
                            blank_areas.append(blank_center)
                        current_blank = 0
                
                # 如果找到足够的空白区域
                if len(blank_areas) >= 2:
                    logger.info(f"检测到 {len(blank_areas)} 个空白区域，基于空白区域分块")
                    # 排序空白区域
                    blank_areas.sort()
                    
                    # 在顶部和底部添加边界线
                    split_lines = [0] + blank_areas + [height]
                    
                    # 根据空白区域创建块
                    for i in range(len(split_lines) - 1):
                        top = split_lines[i]
                        bottom = split_lines[i + 1]
                        # 确保块的高度足够大
                        if bottom - top > 30:
                            blocks.append((0, top, width, bottom - top))
                else:
                    # 备用方案：根据平均行高估算分块
                    logger.info("使用备用方案：基于平均行高估算分块")
                    # 计算合适的块高度阈值，基于图片高度估算
                    avg_block_height = height // 6  # 假设平均每个基金块占图片高度的1/6
                    min_block_height = avg_block_height // 2
                    max_block_height = avg_block_height * 2
                    
                    # 筛选合适的轮廓
                    filtered_contours = []
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        # 筛选高度在合理范围内，且宽度超过图片宽度70%的块
                        if min_block_height < h < max_block_height and w > width * 0.7:
                            filtered_contours.append((x, y, w, h))
                    
                    # 根据y坐标排序
                    filtered_contours.sort(key=lambda b: b[1])
                    
                    # 合并相邻且高度相近的轮廓
                    if filtered_contours:
                        merged_contours = [filtered_contours[0]]
                        for current in filtered_contours[1:]:
                            last = merged_contours[-1]
                            # 如果当前块与上一个块在垂直方向上接近，且高度相近，则合并
                            # 增加垂直间隔容忍度到30像素，避免基金名称和金额被分割
                            if (current[1] - (last[1] + last[3])) < 30 and abs(current[3] - last[3]) < 40:
                                # 合并后的块取最左、最上、最右、最下的坐标
                                new_x = min(last[0], current[0])
                                new_y = last[1]
                                new_w = max(last[0] + last[3], current[0] + current[3]) - new_x
                                new_h = current[1] + current[3] - new_y
                                merged_contours[-1] = (new_x, new_y, new_w, new_h)
                            else:
                                merged_contours.append(current)
                        
                        # 调整块的宽度为图片宽度
                        for (x, y, w, h) in merged_contours:
                            blocks.append((0, y, width, h))
        
        # 按y坐标排序，确保从上到下处理
        blocks.sort(key=lambda block: block[1])
        
        # 设置优化块为处理后的块
        optimized_blocks = blocks
        
        # 过滤掉过小的优化块 - 降低阈值提高分块数量
        optimized_blocks = [(x, y, w, h) for x, y, w, h in optimized_blocks if w > width * 0.2 and h > height * 0.03]
        
        # 确保块按y坐标排序
        optimized_blocks.sort(key=lambda block: block[1])
        
        # 在应用后备分块机制前，对当前块进行基于数值数量的预过滤
        filtered_blocks = []
        
        for block in optimized_blocks:
            x, y, w, h = block
            
            # 对于高度较大的块，直接保留
            if h > height * 0.15:
                filtered_blocks.append(block)
                continue
            
            try:
                # 提取块区域
                block_image = image[y:y+h, x:x+w]
                
                # 预处理块图像
                gray_block = cv2.cvtColor(block_image, cv2.COLOR_BGR2GRAY)
                _, thresh_block = cv2.threshold(gray_block, 150, 255, cv2.THRESH_BINARY_INV)
                
                # 检测块中的轮廓
                contours_block, _ = cv2.findContours(thresh_block, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 检测数值区域
                numeric_count = 0
                
                for cnt in contours_block:
                    x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
                    
                    # 跳过过小的区域
                    if w_cnt < 8 or h_cnt < 6:
                        continue
                    
                    # 数值特征：宽高比通常在1:1到3:1之间，且比较紧凑
                    aspect_ratio = w_cnt / h_cnt
                    if 0.8 <= aspect_ratio <= 3.0:
                        # 数值区域通常具有较高的面积密度（填充度）
                        area = cv2.contourArea(cnt)
                        rect_area = w_cnt * h_cnt
                        if rect_area > 0:
                            solidity = area / rect_area
                            if solidity > 0.4:
                                numeric_count += 1
                
                # 降低数值检测的阈值，提高分块数量
                # 对于高度较大的块，至少包含1个数值就保留
                # 对于高度较小的块，至少包含2个数值就保留
                if (h > height * 0.1 and numeric_count >= 1) or (h <= height * 0.1 and numeric_count >= 2):
                    filtered_blocks.append(block)
                    logger.debug(f"预过滤保留块 {x},{y},{w},{h}: 检测到 {numeric_count} 个数值区域")
            except Exception as e:
                # 如果处理出错，保留原始块
                filtered_blocks.append(block)
                logger.warning(f"预过滤数值检测出错: {str(e)}")
        
        # 使用过滤后的块
        optimized_blocks = filtered_blocks
        
        # 如果分块数量仍然不足，应用基于平均行高的后备分块机制
        if len(optimized_blocks) < 2:
            logger.info(f"当前分块数量不足（{len(optimized_blocks)}块），应用基于平均行高的后备分块机制")
            # 寻找可能包含文本的小轮廓
            text_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # 筛选可能包含文本的小轮廓（宽度和高度都大于10，高度小于50）
                if w > 10 and h > 10 and h < 50:
                    text_contours.append((x, y, w, h))
            
            # 根据y坐标排序
            text_contours.sort(key=lambda b: b[1])
            
            if text_contours:
                # 计算平均行高
                avg_line_height = sum([h for (x, y, w, h) in text_contours]) / len(text_contours)
                
                # 估算每个基金块包含的行数（基金名称及下方的持仓金额、当日收益、持有收益、累计收益）
                lines_per_block = 5  # 假设每个基金块包含约5行文本
                estimated_block_height = avg_line_height * lines_per_block
                
                # 针对基金列表特殊优化：识别基金条目之间的分隔线（基于行高特征）
                fund_separators = []
                for i in range(1, len(text_contours)):
                    prev_contour = text_contours[i-1]
                    curr_contour = text_contours[i]
                    
                    # 基金条目之间的垂直距离通常大于单个条目中的行间距
                    if curr_contour[1] - (prev_contour[1] + prev_contour[3]) > avg_line_height * 2:  # 增加垂直距离要求，减少过多分隔符
                        fund_separators.append((prev_contour[1] + prev_contour[3] + curr_contour[1]) // 2)
                
                # 基于行高和基金分隔符重新分块
                # 结合基金分隔符和行高特征进行智能分块
                current_block = text_contours[0]
                new_blocks = []
                separator_index = 0
                
                for i, contour in enumerate(text_contours[1:]):
                    cx, cy, cw, ch = current_block
                    bx, by, bw, bh = contour
                    
                    # 检查是否接近基金分隔符
                    near_separator = False
                    if separator_index < len(fund_separators) and abs(by - fund_separators[separator_index]) < avg_line_height * 0.7:  # 减少对分隔符附近分块的敏感度
                        near_separator = True
                        separator_index += 1
                    
                    # 如果当前轮廓与当前块在垂直方向上接近，属于同一块
                    # 增加垂直间隔容忍度到1.5倍行高，确保基金名称和金额在同一块中
                    if by - (cy + ch) < avg_line_height * 1.5 and not near_separator:
                        # 扩展当前块
                        new_x = min(cx, bx)
                        new_y = min(cy, by)
                        new_w = max(cx + cw, bx + bw) - new_x
                        new_h = max(cy + ch, by + bh) - new_y
                        current_block = (new_x, new_y, new_w, new_h)
                    else:
                        new_blocks.append(current_block)
                        current_block = contour
                
                # 添加最后一个块
                if current_block not in new_blocks:
                    new_blocks.append(current_block)
                
                # 合并相近的块（确保每个块包含完整的基金数据）
                if new_blocks:
                    merged = [new_blocks[0]]
                    for block in new_blocks[1:]:
                        last = merged[-1]
                        # 如果当前块与上一个块在垂直方向上接近，则合并
                        # 大幅提高垂直间隔容忍度，允许更大的间距合并
                        if block[1] - (last[1] + last[3]) < avg_line_height * 3:
                            # 合并块
                            new_x = min(last[0], block[0])
                            new_y = last[1]
                            new_w = max(last[0] + last[3], block[0] + block[3]) - new_x
                            new_h = block[1] + block[3] - new_y
                            merged[-1] = (new_x, new_y, new_w, new_h)
                        else:
                            merged.append(block)
                    
                    # 调整块的宽度为图片宽度，并只添加高度合理的块
                    for (x, y, w, h) in merged:
                        if h > estimated_block_height * 0.5:
                            # 检查是否已经存在相似的块
                            duplicate = False
                            for existing_block in optimized_blocks:
                                ex, ey, ew, eh = existing_block
                                # 扩大重复块的判断范围，更严格地避免重复添加
                                if abs(ey - y) < avg_line_height * 2 and abs(eh - h) < avg_line_height * 3:
                                    duplicate = True
                                    break
                            
                            if not duplicate:
                                optimized_blocks.append((0, y, width, h))
            
            # 再次排序
            optimized_blocks.sort(key=lambda block: block[1])
        
        # 在返回最终结果前，进行基于数值数量的过滤，确保只保留包含基金信息的块
        final_filtered_blocks = []
        
        for block in optimized_blocks:
            x, y, w, h = block
            
            # 对于高度较大的块，直接保留
            if h > height * 0.15:
                final_filtered_blocks.append(block)
                continue
            
            try:
                # 提取块区域
                block_image = image[y:y+h, x:x+w]
                
                # 预处理块图像
                gray_block = cv2.cvtColor(block_image, cv2.COLOR_BGR2GRAY)
                _, thresh_block = cv2.threshold(gray_block, 150, 255, cv2.THRESH_BINARY_INV)
                
                # 检测块中的轮廓
                contours_block, _ = cv2.findContours(thresh_block, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 检测数值区域的简单方法：查找小而窄的区域，通常数值字符比较紧凑
                numeric_count = 0
                
                for cnt in contours_block:
                    x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
                    
                    # 跳过过小的区域
                    if w_cnt < 8 or h_cnt < 6:
                        continue
                    
                    # 数值特征：宽高比通常在1:1到3:1之间，且比较紧凑
                    aspect_ratio = w_cnt / h_cnt
                    if 0.8 <= aspect_ratio <= 3.0:
                        # 数值区域通常具有较高的面积密度（填充度）
                        area = cv2.contourArea(cnt)
                        rect_area = w_cnt * h_cnt
                        if rect_area > 0:
                            solidity = area / rect_area
                            if solidity > 0.4:
                                numeric_count += 1
                
                # 降低数值检测阈值，提高分块数量
                # 对于高度较大的块，要求至少1个数值；对于高度较小的块，要求至少2个数值
                if (h > height * 0.1 and numeric_count >= 1) or (h <= height * 0.1 and numeric_count >= 2):
                    final_filtered_blocks.append(block)
                    logger.debug(f"保留块 {x},{y},{w},{h}: 检测到 {numeric_count} 个数值区域")
            except Exception as e:
                # 如果处理出错，保留原始块
                final_filtered_blocks.append(block)
                logger.warning(f"数值检测出错: {str(e)}")
        
        # 使用最终过滤后的块
        optimized_blocks = final_filtered_blocks
        
        logger.info(f"最终分块数量: {len(optimized_blocks)}")
        
        # 保存内容块为临时文件
        saved_blocks_info = []
        if save_blocks and blocks_dir and optimized_blocks:
            # 创建一个标记了所有内容块的完整图片
            marked_image = image.copy()
            
            for i, block in enumerate(optimized_blocks):
                x, y, w, h = block
                
                # 为每个块绘制矩形
                cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(marked_image, f'Block {i+1}', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # 保存每个块为单独的图片
                block_image = image[y:y+h, x:x+w]
                block_file_path = os.path.join(blocks_dir, f'block_{i+1}.png')
                cv2.imwrite(block_file_path, block_image)
                
                saved_blocks_info.append({
                    'block_index': i,
                    'file_path': block_file_path,
                    'coordinates': (x, y, w, h)
                })
                logger.info(f"已保存内容块 {i+1} 到: {block_file_path}")
            
            # 保存标记了内容块的完整图片
            marked_file_path = os.path.join(blocks_dir, 'marked_image.png')
            cv2.imwrite(marked_file_path, marked_image)
            logger.info(f"已保存标记了内容块的完整图片到: {marked_file_path}")
            
            # 添加完整标记图片的信息
            saved_blocks_info.append({
                'block_index': -1,
                'file_path': marked_file_path,
                'description': '标记了所有内容块的完整图片'
            })
            
            # 将保存的内容块信息记录到日志，方便用户查看
            logger.info(f"内容块图片保存目录: {blocks_dir}")
            logger.info(f"共有 {len(optimized_blocks)} 个内容块被保存")
        
        # 如果保存了内容块，返回块信息和坐标
        if saved_blocks_info:
            return {
                'blocks': optimized_blocks,
                'saved_blocks_info': saved_blocks_info,
                'blocks_dir': blocks_dir
            }
        
        return optimized_blocks
    except Exception as e:
        logger.error(f"图片分块处理失败: {str(e)}")
        return []


def recognize_block_text(reader, image, block):
    """
    识别指定块中的文本
    
    参数:
        reader: EasyOCR读取器实例
        image: 原始图片
        block: 块的坐标(x, y, w, h)
    
    返回:
        识别出的文本列表
    """
    x, y, w, h = block
    # 提取块区域
    block_image = image[y:y+h, x:x+w]
    # 识别文本
    results = reader.readtext(block_image)
    # 处理识别结果
    recognized_texts = []
    for (bbox, text, prob) in results:
        # 调整坐标，使其相对于整个图片
        adjusted_bbox = [(point[0] + x, point[1] + y) for point in bbox]
        recognized_texts.append({
            "text": text,
            "confidence": float(prob),
            "bbox": adjusted_bbox
        })
    return recognized_texts


def extract_portfolio_data(recognized_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从OCR识别结果中提取结构化的持仓组合数据
    遵循简单规则：每个块最前面的一串是基金名称，基金名称下面是持仓金额
    如果不满足这种条件的，视为无效的基金信息
    
    参数:
        recognized_texts: OCR识别的文本结果，包含块信息
    
    返回:
        结构化的持仓组合数据，包含基金名称、持仓金额等
    """
    portfolio_items = []
    
    # 过滤低置信度的识别结果
    high_confidence_texts = [item for item in recognized_texts if item.get("confidence", 0) > 0.5]
    
    # 按块分组文本
    texts_by_block = {}
    for item in high_confidence_texts:
        block_idx = item.get("block_index", -1)
        if block_idx not in texts_by_block:
            texts_by_block[block_idx] = []
        texts_by_block[block_idx].append(item)
    
    # 匹配基金代码 (6位数字)
    code_pattern = r'\b\d{6}\b'
    # 匹配金额 (数字、可能带逗号分隔符、可能带小数点、可能带单位)
    amount_pattern = r'\d+(?:,\d{3})*(?:\.\d+)?(?:元|万|亿)?'
    import re
    
    # 收集所有文本，用于特殊情况处理
    all_texts = [item["text"] for item in high_confidence_texts] if high_confidence_texts else []
    
    # 遍历每个块，按照简单规则提取基金信息
    for block_idx, block_texts in texts_by_block.items():
        if block_idx == -1:  # 跳过全局块
            continue
            
        # 初始化基金条目，使用None作为默认值以便后续处理
        fund_item = {
            "original_text": " ".join([item["text"] for item in block_texts]),
            "fund_name": None,
            "fund_code": None,
            "hold_amount": None,
            "daily_profit": None,
            "hold_profit": None,
            "total_profit": None,
            "hold_proportion": None,
            "block_index": block_idx
        }
        
        # 步骤1：提取基金名称 - 块内最前面的文本作为基金名称
        if block_texts:
            # 取块内第一个文本作为基金名称
            fund_name_text = block_texts[0]["text"].strip()
            
            # 尝试提取基金代码
            code_match = re.search(code_pattern, fund_name_text)
            if code_match:
                fund_item["fund_code"] = code_match.group()
                # 移除代码后的文本作为基金名称
                fund_item["fund_name"] = fund_name_text.replace(fund_item["fund_code"], '').strip()
            else:
                fund_item["fund_name"] = fund_name_text
        
        # 步骤2：提取持仓金额 - 寻找块内所有可能的金额
        # 遍历块内所有文本项寻找金额
        amount_found = False
        for text_item in block_texts[1:]:  # 从第二个文本项开始找金额
            text = text_item["text"]
            amount_match = re.search(amount_pattern, text)
            if amount_match:
                # 清理金额格式
                amount = re.sub(r'[^\d.]', '', amount_match.group())
                
                # 修正金额：如果没有小数点，可能是识别错误，除以100
                if '.' not in amount and amount.isdigit() and len(amount) > 2:
                    try:
                        corrected_amount = float(amount) / 100
                        amount = f"{corrected_amount:.2f}"
                    except ValueError:
                        pass  # 转换失败，保留原始金额
                
                # 确保金额是Python原生类型
                try:
                    # 尝试转换为浮点数，确保不是numpy类型
                    fund_item["hold_amount"] = float(amount) if amount else None
                except (ValueError, TypeError):
                    # 如果转换失败，尝试作为字符串保留
                    fund_item["hold_amount"] = str(amount) if amount else None
                amount_found = True
                break  # 找到一个金额就足够了
        
        # 步骤3：只有同时有基金名称和持仓金额的块才被认为是有效块
        if fund_item["fund_name"] and fund_item["hold_amount"]:
            portfolio_items.append(fund_item)
    
    # 寻找所有收益相关信息
    # 优先处理占比信息的模式
    proportion_pattern = r'(?:占比\s*)?\d+(?:\.\d+)?%'
    # 更严格的收益金额模式，确保能匹配各种格式
    profit_amount_pattern = r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?'
    profit_percentage_pattern = r'[+-]?\d+(?:\.\d+)?%'
    
    # 导入numpy用于类型检查和转换
    import numpy as np
    
    # 收集所有数值信息
    for fund_item in portfolio_items:
        fund_block_idx = fund_item["block_index"]
        block_texts = texts_by_block.get(fund_block_idx, [])
        
        # 先处理占比信息
        for text_item in block_texts[1:]:  # 跳过基金名称文本
            text = text_item["text"]
            # 优先匹配占比信息
            proportion_match = re.search(proportion_pattern, text)
            if proportion_match:
                fund_item["hold_proportion"] = proportion_match.group()
                # 占比信息不应该被当作其他收益字段
                continue
                
        # 再处理其他收益信息（根据用户要求，持有收益应该位于第三列）
        # 假设块内有多个数值文本项，根据位置区分不同类型的收益
        profit_values = []
        for text_item in block_texts[1:]:  # 跳过基金名称文本
            text = text_item["text"]
            # 跳过已经识别为占比的文本
            if re.search(proportion_pattern, text):
                continue
            
            # 寻找收益金额
            profit_match = re.search(profit_amount_pattern, text)
            if profit_match:
                value = profit_match.group()
                try:
                    # 清理金额格式，移除可能的逗号分隔符
                    clean_value = re.sub(r',', '', value)
                    # 尝试将带正负号的金额转换为Python原生浮点数
                    numeric_value = float(clean_value)
                    # 存储为浮点数以便JSON序列化
                    profit_values.append(numeric_value)
                except ValueError:
                    # 如果转换失败，尝试处理不同类型
                    if hasattr(value, 'item'):
                        profit_values.append(value.item())
                    else:
                        # 对于无法转换为数字的，存储为字符串
                        profit_values.append(str(value))
                continue
                
            # 寻找收益百分比
            percentage_match = re.search(profit_percentage_pattern, text)
            if percentage_match:
                profit_values.append(percentage_match.group())
        
        # 根据位置分配收益信息
        # 按照用户要求：持有收益应该位于第三列
        # 这里简化处理：假设第一个数值是当日收益，第三个数值是持有收益
        if len(profit_values) >= 1:
            fund_item["daily_profit"] = profit_values[0]
        if len(profit_values) >= 3:
            fund_item["hold_profit"] = profit_values[2]  # 持有收益在第三列
        elif len(profit_values) >= 2:
            fund_item["hold_profit"] = profit_values[1]  # 如果没有第三列，使用第二列
        if len(profit_values) >= 3:
            fund_item["total_profit"] = profit_values[2]  # 累计收益也可以使用第三列
    
    # 9. 打印每一块提取的基金信息
    for i, item in enumerate(portfolio_items):
        # 确保至少有基金名称或代码
        if item["fund_name"] or item["fund_code"]:
            print(f"=== 块 {item.get('block_index', i+1)} 提取结果 ===")
            print(f"基金名称: {item.get('fund_name', '未识别')}")
            print(f"持仓金额: {item.get('hold_amount', '未识别')}")
            print(f"持有收益: {item.get('hold_profit', '未识别')}")
            print("==================================")
    
    # 10. 去重并清理结果
    unique_items = []
    seen_names = set()
    
    for item in portfolio_items:
        # 确保至少有基金名称或代码
        if item["fund_name"] or item["fund_code"]:
            # 去重
            name_key = item["fund_name"] if item["fund_name"] else item["fund_code"]
            if name_key not in seen_names:
                seen_names.add(name_key)
                # 清理多余的空格和特殊字符
                if item["fund_name"]:
                    item["fund_name"] = ' '.join(item["fund_name"].split())
                unique_items.append(item)
    
    # 10. 确保所有数据都是可JSON序列化的Python原生类型
    print("===== 开始类型转换处理 =====")
    for idx, item in enumerate(unique_items):
        print(f"处理项目 #{idx+1}: {item}")
        # 遍历所有字段进行类型检查和转换
        for key, value in item.items():
            try:
                # 记录原始值和类型
                original_type = type(value)
                # 处理None值，转换为空字符串以便前端显示
                if value is None:
                    print(f"  转换 {key}: None -> ''")
                    item[key] = ""
                # 处理numpy类型
                elif hasattr(value, 'item'):
                    # 特别检测numpy.int32类型
                    if isinstance(value, np.int32):
                        print(f"  发现numpy.int32类型: {key}: {value}")
                    converted_value = value.item()
                    print(f"  转换 {key}: {value} ({original_type}) -> {converted_value} ({type(converted_value)})")
                    item[key] = converted_value
                # 尝试将字符串形式的数字转换为浮点数
                elif isinstance(value, str):
                    # 只针对金额相关字段进行转换尝试
                    if key in ["hold_amount", "daily_profit", "hold_profit", "total_profit"]:
                        try:
                            # 清理可能的逗号分隔符
                            clean_value = re.sub(r',', '', value)
                            # 尝试转换为浮点数
                            converted_value = float(clean_value)
                            print(f"  转换 {key}: '{value}' -> {converted_value} (float)")
                            item[key] = converted_value
                        except (ValueError, TypeError) as e:
                            # 转换失败，保持原样
                            print(f"  转换 {key} 失败 (保持原值): '{value}' -> 错误: {str(e)}")
                            pass
                # 确保其他类型都是Python原生类型
                elif isinstance(value, (np.integer, np.floating, np.ndarray)):
                    converted_value = value.item() if hasattr(value, 'item') else str(value)
                    print(f"  转换 {key}: {value} ({original_type}) -> {converted_value} ({type(converted_value)})")
                    item[key] = converted_value
                # 检查是否为Python原生类型
                elif not isinstance(value, (str, int, float, bool, list, dict)):
                    print(f"  警告: {key} 包含非标准Python类型: {original_type}, 值: {value}")
                    try:
                        # 尝试转换为字符串
                        item[key] = str(value)
                        print(f"    已转换为字符串: {item[key]}")
                    except Exception as e:
                        print(f"    转换失败: {str(e)}")
            except Exception as e:
                print(f"  处理字段 {key} 时发生异常: {str(e)}")
                # 发生异常时，尝试将值转换为字符串以确保安全
                try:
                    item[key] = str(value)
                    print(f"    已恢复为字符串: {item[key]}")
                except:
                    item[key] = "[转换错误]"
                    print("    无法恢复，设置为[转换错误]")
    print("===== 类型转换处理完成 =====")
    
    # 11. 如果还是没有识别到结构化数据，返回优化的原始数据
    if not unique_items:
        for i, line in enumerate(all_texts):
            if line.strip() and len(line.strip()) > 3:
                unique_items.append({
                    "original_text": line.strip(),
                    "line_number": i + 1
                })
    
    return unique_items

@ocr_router.get("/health")
def ocr_health():
    """
    OCR服务健康检查
    """
    return {
        "status": "healthy",
        "service": "OCR Recognition",
        "supported_languages": ["ch_sim", "en"]
    }
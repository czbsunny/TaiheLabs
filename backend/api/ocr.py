from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging

# 导入OCR处理器
from core.ocr_handler import OCRHandler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建OCR路由器
ocr_router = APIRouter()

# 初始化OCR处理器
ocr_handler = OCRHandler(languages=['ch_sim', 'en'], gpu=False)

@ocr_router.post("/recognize-portfolio")
async def recognize_portfolio(
    file: UploadFile = File(...),
    save_blocks: bool = False
):
    """
    识别图片中的持仓组合内容，直接调用OCR处理器的接口
    
    参数:
        file: 包含持仓组合的图片文件
        save_blocks: 是否保存检测到的内容块为临时文件
    
    返回:
        识别出的持仓组合文本内容
    """
    try:
        # 直接调用ocr_handler的recognize_image方法处理识别
        result = await ocr_handler.recognize_image(file, save_blocks=save_blocks)
        print(result)
        # 如果识别成功，返回结果
        if result.get("status") == "success":
            # 对结果进行调整，使其更符合持仓组合识别的需求
            result["message"] = "持仓组合识别成功"
            return JSONResponse(result)
        else:
            # 如果识别失败，抛出异常
            raise HTTPException(status_code=500, detail=result.get("message", "OCR识别失败"))
    except Exception as e:
        logger.error(f"OCR识别失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR识别失败: {str(e)}")

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
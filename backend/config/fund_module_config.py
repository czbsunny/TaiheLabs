from typing import List, Dict, Any, Optional
import os
from datetime import timedelta

class FundModuleConfig:
    """基金信息处理模块配置类"""
    
    # ---------- 数据源配置 ----------
    
    # 可用的数据源列表
    DATA_SOURCES: List[Dict[str, Any]] = [
        {
            "name": "mock_data_source",
            "type": "mock",
            "priority": 1,
            "enabled": True,
            "timeout": 5,  # 超时时间（秒）
            "retry_count": 3,  # 重试次数
            "retry_delay": 2,  # 重试间隔（秒）
            # 其他特定数据源配置
        },
        # 可以添加真实数据源配置
        # {
        #     "name": "real_data_source",
        #     "type": "real",
        #     "priority": 2,
        #     "enabled": False,
        #     "api_key": os.getenv("FUND_DATA_API_KEY", ""),
        #     "base_url": "https://api.example.com/fund",
        #     "timeout": 10,
        #     "retry_count": 3,
        #     "retry_delay": 3,
        # },
    ]
    
    # ---------- 同步配置 ----------
    
    # 同步模式："incremental" (增量) 或 "full" (全量)
    SYNC_MODE: str = "incremental"
    
    # 全量同步间隔（天）
    FULL_SYNC_INTERVAL_DAYS: int = 30
    
    # 增量同步间隔（小时）
    INCREMENTAL_SYNC_INTERVAL_HOURS: int = 24
    
    # 同步任务的超时时间（分钟）
    SYNC_TIMEOUT_MINUTES: int = 60
    
    # 同步时的并发数
    SYNC_CONCURRENCY: int = 5
    
    # 是否启用定时同步
    ENABLE_SCHEDULED_SYNC: bool = True
    
    # 定时同步的Cron表达式（默认为每天凌晨2点）
    SCHEDULED_SYNC_CRON: str = "0 2 * * *"
    
    # ---------- 批量处理配置 ----------
    
    # 批量处理基金的大小
    BATCH_PROCESS_SIZE: int = 100
    
    # 批量处理的并发数
    BATCH_PROCESS_CONCURRENCY: int = 3
    
    # ---------- 缓存配置 ----------
    
    # 是否启用缓存
    ENABLE_CACHE: bool = True
    
    # 缓存过期时间（秒）
    CACHE_TTL_SECONDS: int = 3600  # 1小时
    
    # 缓存最大大小（项数）
    CACHE_MAX_SIZE: int = 1000
    
    # 缓存键前缀
    CACHE_KEY_PREFIX: str = "fund_info_"
    
    # ---------- 日志配置 ----------
    
    # 日志级别：DEBUG, INFO, WARNING, ERROR
    LOG_LEVEL: str = "INFO"
    
    # 日志文件路径
    LOG_FILE: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "fund_module.log")
    
    # 是否在控制台显示日志
    CONSOLE_LOG: bool = True
    
    # 日志文件的最大大小（MB）
    LOG_FILE_MAX_SIZE_MB: int = 10
    
    # 日志文件的备份数量
    LOG_BACKUP_COUNT: int = 5
    
    # ---------- 验证配置 ----------
    
    # 基金代码验证正则表达式
    FUND_CODE_PATTERN: str = r"^[0-9]{6}$"
    
    # 基金名称最大长度
    MAX_FUND_NAME_LENGTH: int = 200
    
    # ---------- 数据清洗配置 ----------
    
    # 是否自动规范化基金名称
    AUTO_NORMALIZE_FUND_NAME: bool = True
    
    # 自动规范化的规则配置
    NAME_NORMALIZATION_RULES: Dict[str, Any] = {
        "remove_spaces": True,
        "normalize_ellipsis": True,
        "normalize_chinese_chars": True,
        "remove_special_chars": False,  # 是否移除特殊字符
        "allowed_special_chars": ["-", "(", ")", ",", "."],  # 允许保留的特殊字符
    }
    
    # ---------- 错误处理配置 ----------
    
    # 是否启用错误邮件通知
    ENABLE_ERROR_EMAIL_NOTIFICATION: bool = False
    
    # 错误通知邮箱列表
    ERROR_NOTIFICATION_EMAILS: List[str] = []
    
    # 错误类型的阈值配置（超过阈值时触发通知）
    ERROR_THRESHOLD_CONFIG: Dict[str, Any] = {
        "sync_error": {
            "count": 10,  # 错误次数阈值
            "time_window": timedelta(hours=1),  # 时间窗口
        },
        "data_validation_error": {
            "count": 20,
            "time_window": timedelta(hours=2),
        },
    }
    
    # ---------- 性能优化配置 ----------
    
    # 数据库查询的批处理大小
    DB_QUERY_BATCH_SIZE: int = 200
    
    # 数据库连接池大小
    DB_POOL_SIZE: int = 20
    
    # 数据库连接池最大溢出数
    DB_MAX_OVERFLOW: int = 10
    
    # ---------- 版本控制配置 ----------
    
    # 是否启用版本控制
    ENABLE_VERSION_CONTROL: bool = True
    
    # 版本保留的最大数量
    MAX_VERSIONS_TO_KEEP: int = 10
    
    # ---------- API配置 ----------
    
    # API请求超时时间（秒）
    API_TIMEOUT_SECONDS: int = 30
    
    # API请求限制配置
    API_RATE_LIMIT_CONFIG: Dict[str, Any] = {
        "enabled": True,
        "default_limit": "100/minute",  # 默认速率限制
        "endpoints": {
            # 特定端点的速率限制
            "/api/fund/sync/all": "10/hour",
            "/api/fund/batch": "50/minute",
        },
    }
    
    # ---------- 调试配置 ----------
    
    # 是否启用调试模式
    DEBUG_MODE: bool = False
    
    # 调试模式下是否记录详细的请求和响应
    DEBUG_LOG_DETAILED_REQUEST_RESPONSE: bool = False
    
    @classmethod
    def get_config(cls, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return getattr(cls, key, default)
    
    @classmethod
    def set_config(cls, key: str, value: Any) -> None:
        """动态设置配置项"""
        if hasattr(cls, key):
            setattr(cls, key, value)
    
    @classmethod
    def load_from_env(cls) -> None:
        """从环境变量加载配置"""
        # 尝试从环境变量加载配置
        for attr_name in dir(cls):
            # 只处理类变量且不是以下划线开头的变量
            if not attr_name.startswith('_') and attr_name.isupper():
                env_var_name = f"FUND_MODULE_{attr_name}"
                env_value = os.getenv(env_var_name)
                if env_value is not None:
                    # 尝试转换类型
                    current_value = getattr(cls, attr_name)
                    if isinstance(current_value, bool):
                        # 布尔值处理
                        setattr(cls, attr_name, env_value.lower() == 'true')
                    elif isinstance(current_value, int):
                        # 整数处理
                        try:
                            setattr(cls, attr_name, int(env_value))
                        except ValueError:
                            pass
                    elif isinstance(current_value, float):
                        # 浮点数处理
                        try:
                            setattr(cls, attr_name, float(env_value))
                        except ValueError:
                            pass
                    elif isinstance(current_value, list) or isinstance(current_value, dict):
                        # 复杂类型处理（JSON格式）
                        try:
                            import json
                            setattr(cls, attr_name, json.loads(env_value))
                        except (json.JSONDecodeError, ImportError):
                            pass
                    else:
                        # 字符串处理
                        setattr(cls, attr_name, env_value)

# 全局配置实例
fund_config = FundModuleConfig()

# 从环境变量加载配置
try:
    fund_config.load_from_env()
except Exception as e:
    print(f"Warning: Failed to load environment variables for fund module config: {e}")

# 确保日志目录存在
if hasattr(fund_config, 'LOG_FILE') and fund_config.LOG_FILE:
    log_dir = os.path.dirname(fund_config.LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except Exception as e:
            print(f"Warning: Failed to create log directory: {e}")
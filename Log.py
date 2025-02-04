'''
日志生成、存储
'''
import logging
import os
from functools import wraps  # 导入 wraps


# 定义日志文件路径
log_file_path = "/tmp/general.log"

# 确保日志目录存在
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# 创建一个logger并设置日志级别
log = logging.getLogger('general_logger')
log.setLevel(logging.DEBUG)

# 创建文件处理器
log_handler = logging.FileHandler(log_file_path)
log_handler.setLevel(logging.DEBUG)

# 创建日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)

# 将处理器添加到logger
log.addHandler(log_handler)

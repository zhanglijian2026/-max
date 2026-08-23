import threading
import logging
import logging.handlers
import os
class Tool:
    _opt_logger = None
    _err_logger = None
    _initialized=False
    _log_lock = threading.Lock()
    #录入数据
    @classmethod
    def _init_loggers(cls):
        with cls._log_lock:
            if cls._initialized:
                return
            log_dir = "./logs"
            try:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                #系统日志
                cls._opt_logger = logging.getLogger('sys_opt')
                cls._opt_logger.setLevel(logging.INFO)
                if not cls._opt_logger.handlers:
                    opt_handler = logging.handlers.RotatingFileHandler(
                        os.path.join(log_dir, '系统操作日志.log'),
                        maxBytes=1024*1024,
                        backupCount=3,
                        encoding='utf-8',
                    )
                    opt_handler.setFormatter(logging.Formatter(
                        '%(asctime)s,%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
                    )
                    cls._opt_logger.addHandler(opt_handler)
                #异常日志
                cls._err_logger = logging.getLogger('sys_err')
                cls._err_logger.setLevel(logging.ERROR)
                if not cls._err_logger.handlers:
                    err_handler = logging.handlers.RotatingFileHandler(
                        os.path.join(log_dir, '系统异常日志.log'),
                        maxBytes=1024*1024,
                        backupCount=3,
                        encoding='utf-8',
                    )
                    err_handler.setFormatter(logging.Formatter(
                        '%(asctime)s,%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
                    )
                    cls._err_logger.addHandler(err_handler)
                cls._initialized = True
            except Exception as e:
                print(f"文件初始化失败：{e}")
    #异步处理系统日志
    @classmethod
    def write_sys_opt_log(cls,oper_mode:str):
        cls._init_loggers()
        t = threading.Thread(target=cls._opt_logger.info, args=(oper_mode,))
        t.daemon = True
        t.start()
    #异步处理错误日志
    @classmethod
    def write_err_log(cls,err_mode:str):
        cls._init_loggers()
        t=threading.Thread(target=cls._err_logger.error, args=(err_mode,))
        t.daemon=True
        t.start()
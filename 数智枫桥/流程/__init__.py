import threading
import logging
import logging.handlers
import os
import queue


class Tool:
    _opt_logger = None
    _err_logger = None
    _initialized = False
    _log_lock = threading.Lock()

    # 队列和消费线程
    _opt_queue = None
    _err_queue = None
    _opt_thread = None
    _err_thread = None

    @classmethod
    def _init_loggers(cls):
        with cls._log_lock:
            if cls._initialized:
                return

            log_dir = "./logs"
            os.makedirs(log_dir, exist_ok=True)

            # ========== 系统操作日志 ==========
            cls._opt_queue = queue.Queue(-1)

            opt_handler = logging.handlers.RotatingFileHandler(
                os.path.join(log_dir, '系统操作日志.log'),
                maxBytes=50 * 1024 * 1024,  # 一次运行1MB，设50MB够跑很久
                backupCount=5,
                encoding='utf-8',
            )
            opt_handler.setFormatter(logging.Formatter(
                '%(asctime)s,%(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))

            # 启动唯一的后台写线程
            cls._opt_thread = threading.Thread(
                target=cls._consume,
                args=(cls._opt_queue, opt_handler),
                daemon=True
            )
            cls._opt_thread.start()

            cls._opt_logger = logging.getLogger('sys_opt')
            cls._opt_logger.setLevel(logging.INFO)
            cls._opt_logger.addHandler(logging.handlers.QueueHandler(cls._opt_queue))

            # ========== 系统异常日志 ==========
            cls._err_queue = queue.Queue(-1)

            err_handler = logging.handlers.RotatingFileHandler(
                os.path.join(log_dir, '系统异常日志.log'),
                maxBytes=50 * 1024 * 1024,
                backupCount=5,
                encoding='utf-8',
            )
            err_handler.setFormatter(logging.Formatter(
                '%(asctime)s,%(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))

            cls._err_thread = threading.Thread(
                target=cls._consume,
                args=(cls._err_queue, err_handler),
                daemon=True
            )
            cls._err_thread.start()

            cls._err_logger = logging.getLogger('sys_err')
            cls._err_logger.setLevel(logging.ERROR)
            cls._err_logger.addHandler(logging.handlers.QueueHandler(cls._err_queue))

            cls._initialized = True

    @staticmethod
    def _consume(q, handler):
        """唯一写文件的线程，从队列取日志统一写入"""
        while True:
            try:
                record = q.get()
                if record is None:
                    break
                handler.handle(record)
            except Exception:
                pass  # 消费线程不能崩

    @classmethod
    def write_sys_opt_log(cls, oper_mode: str):
        cls._init_loggers()
        cls._opt_logger.info(oper_mode)  # 直接写，QueueHandler 自动异步

    @classmethod
    def write_err_log(cls, err_mode: str):
        cls._init_loggers()
        cls._err_logger.error(err_mode)

 noise_list = []
    #np.random.seed(seed)
    for _ in range(i):
        # 生成正态样本，超出边界则重新采样
        while True:
            eps = np.random.normal(loc=config.mu, scale=config.sigma)
            if config.bound_low <= eps <= config.bound_high:
                break
        noise_list.append(eps)
    return noise_list

u_rigid = r1 - c1 - l1 / config.loss_coefficient  # 刚性管控净收益
u_offline = r2 - c2 - l2 / config.loss_coefficient  # 线下服务净收益
if config.R3_:
    u_digital = r3 - c3 - l3 / config.loss_coefficient
else:
    u_digital = (
                            config.alpha * r1 + config.beta * r2) - c3 - l3 / config.loss_coefficient + config.gamma * config.delta_R



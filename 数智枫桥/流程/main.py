import HANLP_
import MATLAB
import pysd
import matplotlib.pyplot as plt
import atexit
import time

def cleanup():
    """程序退出时自动执行"""
    plt.close('all')  # 关闭所有图形
    print("🧹 已清理 matplotlib 资源")

def main():
    start = time.time()
    HANLP_.main()
    end = time.time()
    print(f"{end-start}")
    MATLAB.main()
    end_2 = time.time()
    print(f"{end_2-end}")
    pysd.main()
    end_3 = time.time()
    print(f"{end_3-end_2}")
    print(f"{end_3-start}")

if __name__ == '__main__':
    # 注册清理函数
    atexit.register(cleanup)
    main()




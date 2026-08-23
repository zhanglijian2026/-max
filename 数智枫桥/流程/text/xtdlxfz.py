import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False




#读取数据
def init_data():
    # 读取MATLAB输出的博弈动态时序数据
    game_df = pd.read_csv("game_output_timeseries.csv")
    #t_list = game_df["t"].values 无用，先注释
    p_t = game_df["y_t"].values  # 矛盾激化速率驱动
    s1_t = game_df["x1_t"].values  # 矛盾化解速率驱动
    s2_t = game_df["x2_t"].values
    s3_t = game_df["x3_t"].values
    e_t = game_df["E_t"].values  # 矛盾复发流量驱
    return  p_t, s1_t,s2_t,s3_t, e_t

# 系统动力学存量流量模型定义
# 存量：未化解信访矛盾存量Stock
class ConflictStock:
    def __init__(self):
        self.value = 1000 # 初始矛盾存量1000件
        self.history = [1000]

    def update(self, inflow, outflow, dt=1):
        """存量更新：存量 = 存量 + 新增流入 - 化解流出"""
        self.value = self.value + (inflow - outflow) * dt
        self.history.append(self.value)
        return self.value

#核心计算
def zhe_xin(p_t, s_t,e_t):
    # 24个月长期仿真循环(时间加到常量配置里了)
    conflict_stock = ConflictStock()
    for month in range(25):
        # 匹配当期博弈动态变量
        idx = min(month, len(p_t) - 1)
        p = p_t[idx]  # 矛盾激化概率（新增信访流入速率）
        s = s_t[idx]  # 有效服务化解概率（矛盾流出速率）
        e = e_t[idx]  # 均衡偏离度（矛盾复发回流）

        # 流量方程（博弈变量动态驱动）
        inflow = 8 * (1-p)  # 每月新增矛盾流量，激化概率越高流入越大
        outflow = 12 * s  # 每月化解矛盾流量，有效策略越高化解越多
        reflow = 3 * e  # 矛盾复发回流流量

        net_inflow = inflow + reflow - outflow
        conflict_stock.update(net_inflow, 0)
    return  conflict_stock

# 输出24个月存量变化幅度（论文结论数据）
def print_stock(x_1, x_2, x_3):
    strategies = [
            ("刚性管控策略", x_1),
            ("线下服务策略", x_2),
            ("数智融合策略", x_3)
        ]
    for name, stock in strategies:
        start = stock.history[0]
        end = stock.history[-1]
        change_rate = (end - start) / start * 100
        print(f"{name} 24个月矛盾存量变化幅度：{change_rate:.2f}%")

#出图（待修改）
def show_image_process(x_1, x_2, x_3):
    p = Process(target=plot, args=(x_1, x_2, x_3))
    p.daemon = False
    p.start()
# 仿真结果可视化
def plot(x_1, x_2, x_3):
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(26), x_1.history, 'g-', linewidth=2,label="纯刚性政策曲线")
        plt.plot(range(26), x_2.history, 'r-', linewidth=2,label="纯线下柔性曲线")
        plt.plot(range(26), x_3.history, 'b-', linewidth=2,label="数智刚柔融合曲线")
        plt.xlabel("治理周期（月）")
        plt.ylabel("未化解信访矛盾存量")
        plt.title("基于演化博弈时序驱动的退役军人服务保障系统动力学仿真")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('ss.png', dpi=300)  # 路径和保存的图片分辨率
        print("图片已保存")
    except Exception as e:
        print(e)

#执行
def main():
    p_t, s1_t, s2_t, s3_t, e_t=init_data()
    s1_t=zhe_xin(p_t, s1_t,e_t)
    s2_t=zhe_xin(p_t, s2_t,e_t)
    s3_t=zhe_xin(p_t, s3_t,e_t)
    show_image_process(s1_t,s2_t,s3_t)
    print_stock(s1_t,s2_t,s3_t)

if __name__=="__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import config
from multiprocessing import Process
import decorators
from graphviz import Digraph
from log import Tool
#读取数据
@decorators.validate_and_catch(func_name="pysd模块读取数据")
def init_data():
    # 读取MATLAB输出的博弈动态时序数据
    game_df = pd.read_csv(config.sd_csv)
    #t_list = game_df["t"].values 无用，先注释
    p_t = game_df["y_t"].values  # 矛盾激化速率驱动
    s1_t = game_df["x1_t"].values  # 矛盾化解速率驱动
    s2_t = game_df["x2_t"].values
    s3_t = game_df["x3_t"].values
    e_t = game_df["E_t"].values  # 矛盾复发流量驱动
    Tool.write_sys_opt_log(f"成功读取数据,矛盾激化速率驱动,矛盾化解速率驱动,矛盾复发流量驱动")
    return  p_t, s1_t,s2_t,s3_t, e_t

# 系统动力学存量流量模型定义
# 存量：未化解信访矛盾存量Stock
class ConflictStock:
    def __init__(self):
        self.value = config.init_stock  # 初始矛盾存量1000件
        self.history = [config.init_stock]

    def update(self, inflow, outflow, dt=1):
        """存量更新：存量 = 存量 + 新增流入 - 化解流出"""
        self.value = self.value + (inflow - outflow) * dt
        self.history.append(self.value)
        self.value = max(self.value, 0.0)
        return self.value

#核心计算
@decorators.validate_and_catch(func_name="pysd的核心计算")
def zhe_xin(p_t, s_t,e_t):
    # 24个月长期仿真循环(时间加到常量配置里了)
    conflict_stock = ConflictStock()
    for month in range(config.sim_month):
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
@decorators.validate_and_catch(func_name="pysd的存量变化幅度")
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
        plt.plot(range(config.sim_month + 1), x_1.history, 'g-', linewidth=2,label="纯刚性政策曲线")
        plt.plot(range(config.sim_month + 1), x_2.history, 'r-', linewidth=2,label="纯线下柔性曲线")
        plt.plot(range(config.sim_month + 1), x_3.history, 'b-', linewidth=2,label="数智刚柔融合曲线")
        plt.xlabel("治理周期（月）")
        plt.ylabel("未化解信访矛盾存量")
        plt.title("基于演化博弈时序驱动的退役军人服务保障系统动力学仿真")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('ss.png', dpi=300)  # 路径和保存的图片分辨率
        print("图片已保存")
        Tool.write_sys_opt_log("图片保存成功")
        plt.show()
    except Exception as e:
        print(e)
        Tool.write_sys_opt_log(f"pysd模块的绘图出现{e}错误")

def plot_mean():
    cld = Digraph("信访矛盾_CLD", format="svg")
    cld.attr(rankdir="LR", fontname="Microsoft YaHei")

    # 节点
    cld.node("S", "未化解信访矛盾存量", style="filled", fillcolor="#dcebfa")
    cld.node("p", "矛盾激化概率 p_t", shape="ellipse")
    cld.node("s", "策略化解概率 s_t", shape="ellipse")
    cld.node("e", "均衡偏离度 e_t", shape="ellipse")
    cld.node("Inflow", "新增矛盾流入", shape="ellipse")
    cld.node("Reflow", "矛盾复发回流", shape="ellipse")
    cld.node("Outflow", "矛盾化解流出", shape="ellipse")
    cld.node("Game", "演化博弈输出时序", style="filled", fillcolor="#fff2cc")

    # 因果连线并标注极性
    cld.edge("p", "Inflow", label="-")
    cld.edge("Inflow", "S", label="+")

    cld.edge("e", "Reflow", label="+")
    cld.edge("Reflow", "S", label="+")

    cld.edge("s", "Outflow", label="+")
    cld.edge("Outflow", "S", label="-")

    cld.edge("Game", "p")
    cld.edge("Game", "s")
    cld.edge("Game", "e")
    cld.render("conflict_cld", cleanup=True)
    print("CLD矢量图已生成 conflict_cld.svg")
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




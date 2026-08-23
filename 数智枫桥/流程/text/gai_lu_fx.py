import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#读取数据
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False


def init_parameter():
    # 导入 HanLP 量化参数（从 hanlp_game_params.xlsx 读取）
    param_table = pd.read_excel("dd.xlsx")
    # 定义需要匹配的策略名称（顺序可调，不影响结果）
    values = []
    date= [
        "数智刚柔融合",
        "纯刚性管控",
        "纯线下柔性服务"
    ]
    for name in date:
        # 按名称匹配行
        row = param_table[param_table["名称"] == name]
        if row.empty:
            raise ValueError(f"未找到名称为 '{name}' 的行")
        # 提取 C、R、L
        c = row.iloc[0]["成本C"]
        r = row.iloc[0]["收益R"]
        l = row.iloc[0]["次生损耗L"]
        values.extend([c, r, l])
    # 解包为 9 个变量

    c1, r1, l1, c2, r2, l2, c3, r3, l3= values
    return c1, r1, l1, c2, r2, l2, c3, r3, l3

#动态微分方程
# 将C/R/L参数传入微分
def replicator_ode(_t, z, c1, r1, l1, c2, r2, l2,c3, r3, l3)->list:
    _x1,_x2,_x3, _y = z

    # 2. 退役军人三类策略收益
    v_pos = _x3 * (r3 - l3/10) + _x2 * (r2 - l2/10) + _x1 *(r1-l1/10)       # 合规反馈收益
    v_neg = _x1 * l1/10 + _x2* l2/10 + _x3 * l3/10                            # 消极维权收益
    v_avg = _y * v_pos + (1 - _y) * v_neg                             # 退役军人平均收益
    # 3. 标准复制动态方程
    # 如果某策略收益高于平均，其占比会增加

    dy_dt = _y *(1-_y)* (v_pos - v_avg)
    return [dy_dt]

#计算主逻
def calculator(c1, r1, l1,c2, r2, l2,  c3, r3, l3):
    args = (c1, r1, l1, c2, r2, l2, c3, r3, l3)
    #Solve_ivp是python中的SciPy 中用于求解常微分方程组（ODE）初值问题的函数，等价与MATLAB中的obe45
    i=0
    for x in range(1,100,1):
        i+=1
        sol = solve_ivp(
            lambda _t, z: replicator_ode(_t, z, *args),             #
            [0,100],                                          # 时间范围 [0, 100]
            [0.33,0.33,0.34,i/100],                              # 初始值
            method='RK45',                                          # 等价于 MATLAB 的 ode45（默认就是 RK45），也是python里的scipy的一个算法
            rtol=1e-6,                                              # 等价于 'RelTol',1e-6，相对误差容限（精度控制）
            t_eval=np.linspace(0, 100, 1000)        # 输出采样点
        )

        t=sol.t
        y_t  = np.clip(sol.y[3],0,1.0) # #P(t)：#群众矛盾激化概率时序
        # E(t)：系统均衡偏离度（稳态最优x=0.86）
        # E(t)：策略收益不均衡度（后处理计算，不作为ODE状态变量）
        _plot_sync2(t,y_t,x,i)
def _plot_sync2(t, y_t,x,i):
        try:
            plt.figure(figsize=(10, 5))  # 创建画布
            plt.plot(t, y_t, 'b-', linewidth=2, label=f'退役军人协同诉求表达概率,x={round(x/100,2)}')  # 曲线：退役军人合规反馈概率（蓝色实线）
            plt.plot(t, 1 - y_t, "m-", linewidth=2, label=f'退役军人非协同诉求表达概率,x2={round(1-x/100,2)}')
            plt.xlabel('演化迭代周期 t')  # x轴线
            plt.ylabel('策略选择概率')  # y轴线
            plt.legend()  # 对以上曲线显示图例
            plt.title('退役军人演化博弈轨迹')  # 标题
            plt.grid(True)  # 显示网格线
            plt.tight_layout()  # 自动调整间距，就是美化，有点神奇
            plt.savefig(f'协同诉求表达的概率初始值分析/output{i}', dpi=300)  # 路径和保存的图片分辨率
            # 非阻塞显
        except Exception as e:
            print(e)


if __name__ == "__main__":
    calculator(*init_parameter())
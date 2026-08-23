import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False
def init_parameter():

    param_table = pd.read_excel("dd.xlsx")

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

    c1, r1, l1, c2, r2, l2, c3, r3, l3 = values
    return c1, r1, l1,c2, r2, l2, c3, r3, l3

def replicator_ode(_t, z, c1, r1, l1, c2, r2, l2,c3, r3, l3)->list:
    _x1,_x2,_x3 = z
    u_rigid = r1 - c1 - l1/10       # 刚性管控净收益
    u_offline = r2 - c2 - l2/10     # 线下服务净收益
    u_digital = r3 - c3 - l3/10     # 数智融合净收益
    # 基层平均收益（政府群体）
    # x 比例的人选数智，1-x 比例的人选线下
    u_avg = _x1 * u_rigid+ _x2 * u_offline + _x3 * u_digital

    # 3. 标准复制动态方程
    # 如果某策略收益高于平均，其占比会增加
    dx1_dt = _x1  * (u_rigid - u_avg)
    dx2_dt = _x2 * (u_offline - u_avg)
    dx3_dt = _x3 * (u_digital - u_avg)
    return [dx1_dt, dx2_dt, dx3_dt]


def calculator(c1, r1, l1,c2, r2, l2,  c3, r3, l3):
    args = (c1, r1, l1, c2, r2, l2, c3, r3, l3)
    #Solve_ivp是python中的SciPy 中用于求解常微分方程组（ODE）初值问题的函数，等价与MATLAB中的obe45
    i=0
    for x_1 in range(1,9,1):
        for x_2 in range(1,10-x_1,1):
            i+=1
            x_3=1-x_2/10-x_1/10
            sol = solve_ivp(
                lambda _t, z: replicator_ode(_t, z, *args),             #
                [0,100],                                          # 时间范围 [0, 100]
                [x_1/10,x_2/10,x_3],                              # 初始值
                method='RK45',                                          # 等价于 MATLAB 的 ode45（默认就是 RK45），也是python里的scipy的一个算法
                rtol=1e-6,                                              # 等价于 'RelTol',1e-6，相对误差容限（精度控制）
                t_eval=np.linspace(0, 100, 1000)        # 输出采样点
            )

            t=sol.t
            x1_t = np.clip(sol.y[0],0,1.0)# S(t):
            x2_t = np.clip(sol.y[1],0,1.0)
            x3_t = np.clip(sol.y[2],0,1.0)
            _plot_sync(t, x1_t, x2_t, x3_t,x_1,x_2,x_3,i)


#绘图
def _plot_sync(t, x1_t,x2_t,x3_t,x_1,x_2,x_3,i):
        plt.figure(figsize=(10, 5))    #创建画布
        plt.plot(t, x1_t, 'r-', linewidth=2, label=f'选择纯刚性服务x1(t),x1={round(x_1/10,2)}')#曲线：基层选择数智融合概率（红色实线）
        plt.plot(t, x2_t, 'g-', linewidth=2, label=f'选择纯线下柔性服务 x2(t),x2={round(x_2/10,2)}')
        plt.plot(t,x3_t,'m-',linewidth=2,label=f"选择数智融合占比 x3(t),x3={round(x_3,2)}")
        plt.xlabel('演化迭代周期 t')#x轴线
        plt.ylabel('策略选择概率')#y轴线
        plt.legend()#对以上曲线显示图例
        plt.title('三策略演化博弈轨迹')#标题
        plt.grid(True)#显示网格线
        plt.tight_layout()#自动调整间距，就是美化，有点神奇
        plt.savefig(f'三策略概率初始值分析/output{i}', dpi=300,bbox_inches="tight")#路径和保存的图片分辨率




if __name__ == '__main__':
    calculator(*init_parameter())

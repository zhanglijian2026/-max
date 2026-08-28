import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import asyncio
from log import Tool
import config
from multiprocessing import Process
import decorators
import random


data=[]
T=[]
X_1,X_2,X_3=[],[],[]
Y_1,Y_2,E_t=[],[],[]


#读取数据
@decorators.validate_and_catch(func_name="读取haNLP数据")
def init_parameter():
    # 导入 HanLP 量化参数（从 haNLP_game_params.xlsx 读取）
    param_table = pd.read_excel(config.ff)
    # 定义需要匹配的策略名称（顺序可调，不影响结果）
    values = []
    for name in config.date:
        # 按名称匹配行
        row = param_table[param_table["名称"] == name]
        if row.empty:
            raise ValueError(f"未找到名称为 '{name}' 的行")
        # 提取 C、R、L
        c = row.iloc[0]["成本C"]
        r = row.iloc[0]["收益R"]
        l = row.iloc[0]["次生损耗L"]
        j = row.iloc[0]["合规反馈概率"]
        values.extend([c, r, l,j])
    # 解包为 9 个变量

    c1, r1, l1,j1, c2, r2, l2,j2, c3, r3, l3,j3 = values
    Tool.write_sys_opt_log("导入 haNLP 量化参数成功")
    return c1, r1, l1, j1,c2, r2, l2,j2, c3, r3, l3, j3


def truncated_normal_noise(i,_seed):
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

#动态微分方程
# 将C/R/L参数传入微分函数
@decorators.validate_and_catch(func_name="演化博弈模块动态微分方程")
def replicator_ode(t, z, c1, r1, l1, c2, r2, l2,c3, _r3, l3)->list:
    _x1,_x2,_x3, _y1,_y2 = z

    #以_t为种子，设置随机噪音
    seed = int(t * 1000) % 2 ** 32
    u_rigid = r1 - c1 - l1/config.loss_coefficient       # 刚性管控净收益
    u_offline = r2 - c2 - l2/config.loss_coefficient     # 线下服务净收益
    u_digital = (config.alpha*r1+config.beta*r2) - c3 - l3/config.loss_coefficient +config.gamma*config.delta_R    # 数智融合净收益

    # x 比例的人选数智，1-x 比例的人选线下


    # 2. 退役军人三类策略收益
    v_pos = _x3 * ((config.alpha*r1+config.beta*r2+config.gamma*config.delta_R ) - l3/config.loss_coefficient) + _x2 * (r2 - l2/config.loss_coefficient) + _x1 *(r1-l1/config.loss_coefficient)        # 合规反馈收益
    v_neg = _x1 * l1/config.loss_coefficient + _x2* l2/config.loss_coefficient + _x3 * l3/config.loss_coefficient                            # 消极维权收益


    if  config.perceptual_noise==False or config.noise==False:
        v_avg = _y1 * v_pos +  _y2 * v_neg # 退役军人平均收益
        u_avg = _x1 * u_rigid+ _x2 * u_offline + _x3 * u_digital   # 基层平均收益（政府群体）

        dx1_dt = _x1 * (u_rigid - u_avg)
        dx2_dt = _x2 * (u_offline - u_avg)
        dx3_dt = _x3 * (u_digital - u_avg)
        dy1_dt = _y1 * (v_pos - v_avg)
        dy2_dt = _y2 * (v_neg - v_avg)
        return [dx1_dt, dx2_dt, dx3_dt, dy1_dt,dy2_dt]
    x_x1, x_x2, x_x3 ,y_y1,y_y2=truncated_normal_noise(5,seed)
    pnp_u_rigid=u_rigid*(1+x_x1);pnp_u_offline=u_offline*(1+x_x2);pnp_u_digital=u_digital*(1+x_x3);pnp_v_pos=v_pos*(1+y_y1);pnp_v_neg=v_neg*(1+y_y2)

    v_avg = _y1*pnp_v_pos + _y2*pnp_v_neg
    u_avg = _x1 * pnp_u_rigid + _x2 * pnp_u_offline + _x3 * pnp_u_digital
    # 3. 标准复制动态方程
    # 如果某策略收益高于平均，其占比会增加
    dx1_dt = _x1  * (pnp_u_rigid - u_avg)
    dx2_dt = _x2 * (pnp_u_offline - u_avg)
    dx3_dt = _x3 * (pnp_u_digital - u_avg)
    dy1_dt = _y1 * (pnp_v_pos - v_avg)
    dy2_dt = (1-_y1) * (pnp_v_neg - v_avg)
    return [dx1_dt, dx2_dt, dx3_dt,dy1_dt,dy2_dt]

#计算主逻辑
@decorators.validate_and_catch(func_name="演化博弈模块的计算主逻辑")
def calculator(c1, r1, l1, j1,c2, r2, l2, j2, c3, r3, l3,j3):
    if  config.initial_noise==True and config.noise==True:
        now= random.uniform(1, 100)
        seed = int(now * 1000) % 2 ** 32
        #这个用是否固定随机的噪音
        #np.random.seed(12)
        x_c1, x_c2, x_c3,x_r1,x_r2,x_r3,x_l1,x_l2,x_l3,y_x1,y_x2,y_x3= truncated_normal_noise(12,seed)

        pnp_c1=c1*(1+x_c1);pnp_c2=c2*(1+x_c2);pnp_c3=c3*(1+x_c3)
        pnp_r1=r1*(1+x_r1);pnp_r2=r2*(1+x_r2);pnp_r3=r3*(1+x_r3)
        pnp_l1=l1*(1+x_l1);pnp_l2=l2*(1+x_l2);pnp_l3=l3*(1+x_l3)
        pnp_x1=config.init_cond[0]*(1+y_x1);pnp_x2=config.init_cond[1]*(1+y_x2);pnp_x3=np.clip(1-pnp_x1-pnp_x2,0,1)
    else:
        pnp_c1=c1;pnp_c2=c2;pnp_c3=c3;pnp_r1=r1;pnp_r2=r2;pnp_r3=r3;pnp_l1=l1;pnp_l2=l2;pnp_l3=l3
        pnp_x1=config.init_cond[0];pnp_x2=config.init_cond[1];pnp_x3=config.init_cond[2]

    args = (pnp_c1,pnp_r1,pnp_l1,pnp_c2,pnp_r2,pnp_l2,pnp_c3,pnp_r3,pnp_l3)
    #Solve_ivp是python中的SciPy 中用于求解常微分方程组（ODE）初值问题的函数，等价与MATLAB中的obe45
    sol = solve_ivp(
        lambda _t, z: replicator_ode(_t, z, *args),             #
        config.ts_pan,                                          # 时间范围 [0, 100]
        [pnp_x1,pnp_x2,pnp_x3,(pnp_x1*j1+pnp_x2*j2+pnp_x3*j3),1-(pnp_x1*j1+pnp_x2*j2+pnp_x3*j3)],                              # 初始值
        method='RK45',                                          # 等价于 MATLAB 的 ode45（默认就是 RK45），也是python里的scipy的一个算法
        rtol=1e-6,                                              # 等价于 'RelTol',1e-6，相对误差容限（精度控制）
        t_eval=np.linspace(0, 100, 1000)        # 输出采样点
    )
    if not sol.success:
        raise RuntimeError(f"演化博弈ODE求解失败，信息：{sol.message}")
    #solve_ivp 返回一个 OdeResult 对象，包含4个属性
    #属性	        含义	                    用法
    #sol.t	       时间点数组	                t = sol.t
    #sol.y	  状态变量数组（每行是一个变量）	    x_t = sol.y[0]，y_t = sol.y[1]
    #sol.success	是否求解成功	            可用来判断是否收敛
    #sol.message	求解状态信息	            可用于调试

    Tool.write_sys_opt_log("演化博弈完成")

    #提取博弈输出时序变量（供给SD系统动力学）
    t=sol.t
    x1_t = np.clip(sol.y[0],0,1.0)# S(t):
    x2_t = np.clip(sol.y[1],0,1.0)
    x3_t = np.clip(sol.y[2],0,1.0)
    y1_t  = np.clip(sol.y[3],0,1.0) # #P(t)：#群众矛盾激化概率时序
    y2_t = np.clip(sol.y[4],0,1.0)

    # E(t)：系统均衡偏离度（稳态最优x=0.86）
    # E(t)：策略收益不均衡度（后处理计算，不作为ODE状态变量）
    u_rigid_arr = r1 - c1 - l1 / config.loss_coefficient
    u_offline_arr = r2 - c2 - l2 / config.loss_coefficient
    u_digital_arr = (config.alpha*r1+config.beta*r2) - c3 - l3/config.loss_coefficient +config.gamma*config.delta_R
    u_avg_arr = x1_t * u_rigid_arr + x2_t * u_offline_arr + x3_t * u_digital_arr
    e_t = (x1_t * np.abs(u_rigid_arr - u_avg_arr) +
           x2_t * np.abs(u_offline_arr - u_avg_arr) +
           x3_t * np.abs(u_digital_arr - u_avg_arr))

    return t, x1_t,x2_t,x3_t, y1_t,y2_t,e_t
#噪音后数据分流
def mc_traj(all_data):
    for i in all_data:
        if not i:continue
        t,x1_t,x2_t,x3_t,y1_t,y2_t,e_t = i
        T.append(t)
        X_1.append(x1_t)
        X_2.append(x2_t)
        X_3.append(x3_t)
        Y_1.append(y1_t)
        Y_2.append(y2_t)
        E_t.append(e_t)

#均值曲线
def mean_data():
    arg=[T,X_1,X_2,X_3,Y_1,Y_2,E_t]
    arg_s=[]
    for i in arg:
        df=pd.DataFrame(i)
        # 计算平均曲线：每一列求平均
        arg_s.append(df.mean(axis=0))
    return arg_s

#绘图
def _plot_sync(t, x1_t,x2_t,x3_t):
    try:
        plt.figure(figsize=(10, 5))    #创建画布
        plt.plot(t, x1_t, 'r-', linewidth=2, label='选择纯刚性服务x1(t)')#曲线：基层选择数智融合概率（红色实线）
        plt.plot(t, x2_t, 'g-', linewidth=2, label='选择纯线下柔性服务 x2(t)')
        plt.plot(t,x3_t,'m-',linewidth=2,label="选择数智融合占比 x3(t)")
        plt.xlabel('演化迭代周期 t')#x轴线
        plt.ylabel('策略选择概率')#y轴线
        plt.legend()#对以上曲线显示图例
        plt.title('三策略演化博弈轨迹')#标题
        plt.grid(True)#显示网格线
        plt.tight_layout()#自动调整间距，就是美化，有点神奇
        plt.savefig('结果/game_evolution.png', dpi=300,bbox_inches="tight")#路径和保存的图片分辨率
         # 非阻塞显
        Tool.write_sys_opt_log("演化博弈图绘制完成")
        plt.show()
    except Exception as e:
        print(e)
        Tool.write_err_log(f"绘图出现了{e}问题")

def _plot_sync2(t,  y1_t,y2_t):
    try:
        plt.figure(figsize=(10, 5))    #创建画布
        plt.plot(t, y1_t, 'b-', linewidth=2, label='退役军人协同诉求表达概率 ')#曲线：退役军人合规反馈概率（蓝色实线）
        plt.plot(t,y2_t,"m-",linewidth=2,label='退役军人非协同诉求表达概率')
        plt.xlabel('演化迭代周期 t')#x轴线
        plt.ylabel('策略选择概率')#y轴线
        plt.legend()#对以上曲线显示图例
        plt.title('退役军人演化博弈轨迹')#标题
        plt.grid(True)#显示网格线
        plt.tight_layout()#自动调整间距，就是美化，有点神奇
        plt.savefig('结果/game_evolution2.png', dpi=300)#路径和保存的图片分辨率
         # 非阻塞显
        Tool.write_sys_opt_log("退役军人演化博弈图绘制完成")
        plt.show()
    except Exception as e:
        print(e)
        Tool.write_err_log(f"的绘图出现了{e}问题")


def _plot_sync3(t,_all_data):
    try:
        plt.figure(figsize=(10, 5))
        for i in range(len(_all_data)):
            data_name=_all_data[i]
            if i == 0:
                lab1 = "选择纯刚性服务x1(t)"
                lab2 = "选择纯线下柔性服务 x2(t)"
                lab3 = "选择数智融合占比 x3(t)"
            else:
                lab1 = lab2 = lab3 = ""
            plt.plot(t, data_name[1], 'r-', linewidth=2, label=lab1)  # 曲线：基层选择数智融合概率（红色实线）
            plt.plot(t, data_name[2], 'g-', linewidth=2, label=lab2)
            plt.plot(t, data_name[3], 'm-', linewidth=2, label=lab3)
        plt.xlabel('演化迭代周期 t')  # x轴线
        plt.ylabel('策略选择概率')  # y轴线
        plt.legend()  # 对以上曲线显示图例
        plt.title('三策略演化博弈带噪音轨迹')  # 标题
        plt.grid(True)  # 显示网格线
        plt.tight_layout()  # 自动调整间距，就是美化，有点神奇
        plt.savefig('结果/game_evolution3.png', dpi=300, bbox_inches="tight")  # 路径和保存的图片分辨率
        # 非阻塞显
        Tool.write_sys_opt_log("演化博弈图绘制完成")
        plt.show()
    except Exception as e:
        print(e)
#启动独立分进程来显示图片和保存csv
#异步
async def plt_numer_yi_bu(t, x1_t,x2_t,x3_t ,y1_t,y2_t,e_t,all_data):
    def show_image_process(_t, _x1_t,_x2_t,_x3_t):
        p = Process(target=_plot_sync, args=(_t, _x1_t, _x2_t,_x3_t))
        p.daemon = False
        p.start()
    def show_image_process2(_t,_y1_t,_y2_t):
        p = Process(target=_plot_sync2, args=(_t,_y1_t,_y2_t))
        p.daemon=False
        p.start()
    def show_image_process3(_t,_all_data):
        p = Process(target=_plot_sync3,args=(_t,_all_data))
        p.daemon = False
        p.start()
    def matlab_csv(_t, _x1_t, _x2_t,_x3_t,_y1_t,_e_t):
        # 保存为CSV
        pd.DataFrame({"t": _t,
                      "x1_t": _x1_t,
                      "x2_t": _x2_t,
                      "x3_t": _x3_t,
                      "y_t": _y1_t,
                      "E_t": _e_t
                      }).to_csv(config.sd_csv, index=False,encoding='utf-8-sig')
        Tool.write_sys_opt_log("保存为CSV成功")
        print("保存为CSV成功")
    await asyncio.to_thread(show_image_process, t, x1_t,x2_t, x3_t)
    await asyncio.to_thread(show_image_process2, t, y1_t,y2_t)
    await asyncio.to_thread(show_image_process3, t,all_data)
    await asyncio.to_thread(matlab_csv, t, x1_t,x2_t,x3_t, y1_t,e_t)

# 执行
def main():
    if not config.noise:asyncio.run(plt_numer_yi_bu(*calculator(*init_parameter()),data));return
    for _ in range(config.counts):data.append(calculator(*init_parameter()))
    if data:mc_traj(data);asyncio.run(plt_numer_yi_bu(*mean_data(),data))
    else:raise Exception("数据异常")
if __name__ == '__main__':
    main()
import streamlit as st
import MATLAB as ML
import config
import matplotlib.pyplot as plt

def build_sidebar_panel():
    st.header("🎛️ 参数调节面板")

    st.subheader("📊 核心参数")
    ML.mu = st.slider("中心值", -1.0, 1.0, 0.0)
    ML.sigma = st.slider("标准差", 0.0, 1.0, 1.0)
    ML.bound_low = st.slider("下界", -1.0, 0.0, -0.50)
    ML.bound_high = st.slider("上界", 0.0, 1.0, 0.50)

    st.subheader("🎨 显示选项")
    ML.perceptual_noise = st.checkbox("感知噪声", value=False)
    ML.initial_noise = st.checkbox("初始噪声", value=False)
    ML.noise = st.checkbox("噪声", value=False)

def plt_numer_yi_bu(t, x1_t,x2_t,x3_t ,y1_t,y2_t,_,_all_data):
    # 非阻塞显
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t, x1_t, 'b-', linewidth=2, label='选择纯刚性服务x1(t)')
    ax1.plot(t, x2_t, 'g-',  linewidth=2, label='选择纯线下柔性服务 x2(t)')
    ax1.plot(t, x3_t, 'm-', linewidth=2, label="选择数智融合占比 x3(t)")
    ax1.set_title("三策略演化博弈轨迹")
    ax1.legend()
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

    fig2,ax2=plt.subplots(figsize=(10, 5))  # 创建画布
    ax2.plot(t, y1_t, 'b-', linewidth=2, label='退役军人协同诉求表达概率 ')  # 曲线：退役军人合规反馈概率（蓝色实线）
    ax2.plot(t, y2_t, "m-", linewidth=2, label='退役军人非协同诉求表达概率')
    ax2.set_xlabel('演化迭代周期 t')  # x轴线
    ax2.set_ylabel('策略选择概率')  # y轴线
    ax2.legend()  # 对以上曲线显示图例
    ax2.set_title('退役军人演化博弈轨迹')  # 标题
    ax2.grid(True)  # 显示网格线  # 自动调整间距，就是美化，有点神奇
    st.pyplot(fig2)
    plt.close(fig2)
    # 非阻塞显
    if ML.noise:
        fig3,ax3=plt.subplots(figsize=(10, 5))
        for i in range(len(_all_data)):
            name=_all_data[i]
            if i == 0:
                lab1 = "选择纯刚性服务x1(t)"
                lab2 = "选择纯线下柔性服务 x2(t)"
                lab3 = "选择数智融合占比 x3(t)"
            else:lab1 = lab2 = lab3 = ""
            ax3.plot(t, name[1], 'r-', linewidth=2, label=lab1)  # 曲线：基层选择数智融合概率（红色实线）
            ax3.plot(t, name[2], 'g-', linewidth=2, label=lab2)
            ax3.plot(t, name[3], 'm-', linewidth=2, label=lab3)
        ax3.set_xlabel('演化迭代周期 t')  # x轴线
        ax3.set_ylabel('策略选择概率')  # y轴线
        ax3.legend()  # 对以上曲线显示图例
        ax3.set_title('三策略演化博弈带噪音轨迹')  # 标题
        ax3.grid(True)  # 显示网格线
        # 自动调整间距，就是美化，有点神奇
        st.pyplot(fig3)
        plt.close(fig3)

def main():
    """程序主入口"""
    # 页面配置
    st.set_page_config(
        page_title="动态调参面板",
        page_icon="📐",
        layout="wide"
    )

    # 标题
    st.title("📐 数学模型动态调参面板")
    st.caption("拖动左侧滑块，实时调整参数并查看曲线变化")
    left_col, right_col = st.columns([1, 3])
    with left_col:
        build_sidebar_panel()

    with right_col:
        if not ML.noise:plt_numer_yi_bu(*ML.calculator(*ML.init_parameter()), ML.data)
        else:
            for _ in range(config.counts): ML.data.append(ML.calculator(*ML.init_parameter()))
            if ML.data:ML.mc_traj(ML.data);plt_numer_yi_bu(*ML.mean_data(), ML.data)
            else:raise Exception("数据异常")
    # ---- 1. 构建参数面板 ----


    # ---- 2. 生成数据 ----

    # ---- 4. 绘制图表 ----

    # ---- 5. 显示图表 ----
    # 分成左右两列，比例 1:3（左窄右宽）

    # ---- 6. 显示数据统计（可选） ----

if __name__ == '__main__':
    main()


import streamlit as st
import MATLAB as ML
import config
import matplotlib.pyplot as plt
import HANLP_
import pandas as pd
import pysd
import json
import inspect


def build_sidebar_panel():
    st.header("🎛️ 参数调节面板")
    st.subheader("📊 hanlp相关")
    with st.expander("点击展开", expanded=False):
        with st.expander("权重值", expanded=False):
            DATA["高权重"]= st.slider("高权重值", min_value=DATA["中权重"], max_value=10, value=config.DEFAULT_CONFIG["高权重"])
            DATA["中权重"] = st.slider("中权重值", min_value=DATA["低权重"], max_value=DATA["高权重"], value=config.DEFAULT_CONFIG["中权重"])
            DATA["低权重"] = st.slider("低权重",min_value=0,max_value=DATA["中权重"],value=config.DEFAULT_CONFIG["低权重"])
        st.write("")
        col1,col2=st.columns(2)
        with col1:
            DATA["精细度(必须大于0)"]=st.segmented_control("精细度",options=[1,2,3,4],default=config.DEFAULT_CONFIG["精细度(必须大于0)"],selection_mode="single")
            st.write("当前选中:",DATA["精细度(必须大于0)"])
        with col2:
            config.text=st.checkbox("校准度()",value=True)
        st.write("")
        config.text_hanlp=st.checkbox("标点符号（剔除）",value=True)
        DATA["CPU"]=st.slider("CPU核数(看电脑配置，太高会重启)",1,12,value=config.DEFAULT_CONFIG["CPU"])
        st.write("")

    st.subheader("演化博弈")
    with st.expander("点击展开", expanded=False):
        with st.expander("演化博弈初始值", expanded=False):
                DATA["微分方程组初始值"]["政府初始选择刚性管控策略的概率"]=st.slider("刚性管控政策初始概率",0,1,config.DEFAULT_CONFIG["微分方程组初始值"]["政府初始选择刚性管控策略的概率"])
                DATA["微分方程组初始值"]["政府初始选择纯线下服务策略的概率"] = st.slider("纯线下服务策略", 0, 1,config.DEFAULT_CONFIG["微分方程组初始值"]["政府初始选择纯线下服务策略的概率"])
                DATA ["微分方程组初始值"]["政府初始选择数智融合概率"]= st.slider("数智刚柔融合概率", 0, 1,config.DEFAULT_CONFIG["微分方程组初始值"]["政府初始选择数智融合概率"] )
                total = DATA["微分方程组初始值"]["政府初始选择刚性管控策略的概率"] + DATA["微分方程组初始值"]["政府初始选择纯线下服务策略的概率"] + DATA["微分方程组初始值"]["政府初始选择数智融合概率"]
                if abs(total - 1.0) > 0.001:st.warning(f"概率之和为 {total:.2f}，必须等于 1")
                else:st.success("概率校验通过")
                st.write("")
        with st.subheader("", expanded=False):
            DATA["演化博弈系数"]["损耗系数"]=st.slider("损耗系数",0,20,config.DEFAULT_CONFIG["演化博弈系数"]["损耗系数"])
        st.write("")
        with st.expander("公式参数", expanded=False):
            DATA["演化博弈公式参数"]["刚性收益比例"]=st.slider("刚性收益比例",0,1.5,config.DEFAULT_CONFIG["演化博弈公式参数"]["刚性收益比例"])
            DATA["演化博弈公式参数"]["线下收益比例"]=st.slider("线下收益比例",0,1.5,config.DEFAULT_CONFIG["演化博弈公式参数"]["线下收益比例"])
            DATA["演化博弈公式参数"]["刚柔融合额外收益"]=st.slider("刚柔融合额外收益系数",0,1.5,config.DEFAULT_CONFIG["演化博弈公式参数"]["刚柔融合额外收益"])
            DATA["演化博弈公式参数"]["单位比例收益值"]=st.slider("单位比例收益值",0,1.5,config.DEFAULT_CONFIG["演化博弈公式参数"]["单位比例收益值"])
            st.write("")

        config.noise = st.checkbox("噪声", value=False)
        if config.noise:
            with st.expander("噪音设置", expanded=False):
                DATA["噪音博弈次数"]=st.slider("博弈仿真次数",0,100,config.DEFAULT_CONFIG["噪音博弈次数"])
                st.write("")
                config.perceptual_noise = st.checkbox("感知噪声", value=False)
                config.initial_noise = st.checkbox("初始噪声", value=False)


                if config.noise and (config.perceptual_noise or config.initial_noise):
                    with st.expander("噪音程度值", expanded=False):
                        DATA["噪音程度设置"]["中心值"] = st.slider("中心值", -1.0, 1.0, 0.0)
                        DATA["噪音程度设置"]["方差"] = st.slider("标准差", 0.0, 1.0, 1.0)
                        DATA["噪音程度设置"]["取的最大负面值"] = st.slider("下界", -1.0, 0.0, -0.50)
                        DATA["噪音程度设置"]["取的正面最大值"] = st.slider("上界", 0.0, 1.0, 0.50)

# 用 json 模块读写
def copy_json_file(source_path, target_path):
    with open(source_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 加载为字典
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return data.copy()# 写入新文件

# 使用
def copy_json_file2():
    with open(config.CK_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(DATA, f, indent=4, ensure_ascii=False)

#出图
def plt_numer_yi_bu(t, x1_t,x2_t,x3_t ,y1_t,y2_t,_,_all_data):
    # 非阻塞显
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t, x1_t, 'b-', linewidth=2, label='选择纯刚性服务x1(t)')
    ax1.plot(t, x2_t, 'g-',  linewidth=2, label='选择纯线下柔性服务 x2(t)')
    ax1.plot(t, x3_t, 'm-', linewidth=2, label="选择数智融合占比 x3(t)")
    ax1.set_title("三策略演化博弈轨迹")
    ax1.legend()
    ax1.set_xlabel("策略占比")
    ax1.set_ylabel("博弈单位时间")
    ax1.grid(True, alpha=0.3)



    fig2,ax2=plt.subplots(figsize=(10, 5))  # 创建画布
    ax2.plot(t, y1_t, 'b-', linewidth=2, label='退役军人协同诉求表达概率 ')  # 曲线：退役军人合规反馈概率（蓝色实线）
    ax2.plot(t, y2_t, "m-", linewidth=2, label='退役军人非协同诉求表达概率')
    ax2.set_xlabel('演化迭代周期 t')  # x轴线
    ax2.set_ylabel('策略选择概率')  # y轴线
    ax2.legend()  # 对以上曲线显示图例
    ax2.set_title('退役军人演化博弈轨迹')  # 标题
    ax2.grid(True)  # 显示网格线  # 自动调整间距，就是美化，有点神奇

    # 非阻塞显
    with st.expander("演化博弈图", expanded=False):
        st.pyplot(fig1)
        st.write("")
        plt.close(fig1)
    with st.expander("退役军人", expanded=False):
        st.pyplot(fig2)
        st.write("")
        plt.close(fig2)
    if config.noise:
        fig3,ax3=plt.subplots(figsize=(10, 5))
        for i in range(len(_all_data)):
            _name=_all_data[i]
            if i == 0:
                lab1 = "选择纯刚性服务x1(t)"
                lab2 = "选择纯线下柔性服务 x2(t)"
                lab3 = "选择数智融合占比 x3(t)"
            else:lab1 = lab2 = lab3 = ""
            ax3.plot(t, _name[1], 'r-', linewidth=2, label=lab1)  # 曲线：基层选择数智融合概率（红色实线）
            ax3.plot(t, _name[2], 'g-', linewidth=2, label=lab2)
            ax3.plot(t, _name[3], 'm-', linewidth=2, label=lab3)
        ax3.set_xlabel('演化迭代周期 t')  # x轴线
        ax3.set_ylabel('策略选择概率')  # y轴线
        ax3.legend()  # 对以上曲线显示图例
        ax3.set_title('三策略演化博弈带噪音轨迹')  # 标题
        ax3.grid(True)  # 显示网格线
        # 自动调整间距，就是美化，有点神奇
        with st.expander("噪音图", expanded=False):
            st.pyplot(fig3)
            st.write("")
            plt.close(fig3)

#给值
def who_called():
    stack = inspect.stack()
    caller = stack[1]
    return caller

def run():

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
        copy_json_file2()
        HANLP_.main(name)


    with right_col:
        if config.ff is not None:
            # 读取 Excel
            df = pd.read_excel(config.ff)
            # 显示表格
            with st.expander("hanlp结果", expanded=False):
                st.dataframe(df)
        st.subheader(f"{config.C}X{config.R}x{config.L}")

        if not config.noise:plt_numer_yi_bu(*ML.calculator(*ML.init_parameter()), ML.data)
        else:
            for _ in range(config.counts): ML.data.append(ML.calculator(*ML.init_parameter()))
            if ML.data:ML.mc_traj(ML.data);plt_numer_yi_bu(*ML.mean_data(), ML.data)
            else:raise Exception("数据异常")
        pysd.main()
        pysd.plot_mean()

    # ---- 1. 构建参数面板 ----


    # ---- 2. 生成数据 ----

    # ---- 4. 绘制图表 ----

    # ---- 5. 显示图表 ----
    # 分成左右两列，比例 1:3（左窄右宽）

    # ---- 6. 显示数据统计（可选） ----

if __name__ == '__main__':
    name = who_called()
    DATA = copy_json_file(config.CONFIG_PATH, config.CK_CONFIG_PATH)
    run()
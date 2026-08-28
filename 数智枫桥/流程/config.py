import json
import os
import matplotlib.pyplot as plt
import warnings
from log import Tool
import shutil
import decorators
import streamlit as st

# 检索配置文件
CK_CONFIG_PATH=os.path.join(os.path.dirname(__file__), "ck_config.json")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
CUN_DANG = os.path.join(os.path.dirname(__file__), "存档.json")

@decorators.validate_and_catch("加载配置")
def config_main():
    #消除警告
    warnings.filterwarnings('ignore')

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

#配置文件备份
DEFAULT_CONFIGS={
    "保留的策略名称（三级编码）":[
        "纯刚性管控",
        "纯线下柔性服务",
        "数智刚柔融合"],
    "成本类词汇(数值越高=成本越大)": {
        "cost_high": [
            "人力消耗大", "人力", "运维", "基层", "初期", "超支", "昂贵", "开发", "治理", "硬件",
            "授权", "培训", "耗材", "差旅", "会议", "印刷", "档案", "建设", "维护", "安全", "引进",
            "研发", "测试", "部署", "运营", "行政", "协调", "沟通", "时间", "试错", "返工", "闲置",
            "重复", "迁移", "集成", "接口", "人力", "值班", "外包", "咨询", "法务", "审计", "评估",
            "差旅", "场地", "水电", "折旧", "耗材", "备件", "物流", "保险", "成本", "人力", "救急", "救难"

        ],
        "cost_mid": [
            "档案查阅方便", "适中", "可控", "正常", "合理", "简单", "中等", "一般", "稳定",
            "足够", "一般", "普通", "均衡", "可调", "有限", "适度", "轻量", "基础", "简易",
            "标准", "常规", "适当", "合理", "足够", "可接受", "正常", "平均", "常规", "稳定",
            "一般", "适中", "可控", "标准", "合格", "足够", "可行", "合理", "常规", "均衡",
            "可调", "适当", "足够", "合理", "标准", "常规", "平均", "适中", "可接", "一般", "合格", "稳定"
        ],
        "cost_low": [
            "会议效率高", "印刷质量好", "档案查阅快", "快速", "极低", "轻微", "很少", "小", "低廉",
            "短", "低", "少", "低", "少", "低", "易", "小", "低", "少", "低", "小", "短", "低", "低",
            "低", "低", "少", "低", "极少", "高", "少", "低", "低", "快", "足", "少", "低", "低", "低",
            "低", "低", "短", "少", "低", "少", "低", "少", "少", "低", "低", "高", "好", "快"
        ]
    },
    "收益类词汇（数值越高=收益越高）": {
        "gain_high": [

            "协同治理", "制度优势", "服务提质", "闭环", "留痕", "响应", "赋能", "精准", "闭环",
            "通办", "响应", "充沛", "找人", "零跑", "增效", "治理", "获得", "幸福", "荣誉", "归属",
            "尊崇", "化解", "终结", "驱动", "治理", "画像", "服务", "智治", "即办", "预办", "响应",
            "周期", "监管", "熔断", "治理", "共治", "责任", "政务", "施策", "保障", "平安", "工程",
            "经验", "增值", "共创", "共治", "共享", "提升", "发展", "体系", "治理", "优势", "提质"
        ],
        "gain_mid": [

            "线上预约", "上门探望", "温和", "激化", "到位", "引领", "顺畅", "疏导",
            "得力", "重建", "提升", "知晓", "意识", "提升", "创业", "赋能", "招聘",
            "援助", "慰问", "帮扶", "落实", "优待", "光荣", "喜报", "欢迎", "一站",
            "安置", "发放", "覆盖", "推介", "贷款", "联盟", "辅导", "滴灌", "通道",
            "咨询", "接续", "转接", "登记", "查询", "服务", "驿站", "联动", "就医",
            "救援", "帮扶", "疏导", "联系", "帮服", "发放", "预约", "探望"
        ],
        "gain_low": [

            "态度软化", "配合小提", "满意", "改善", "见效", "化解", "成效", "缓解",
            "好转", "提升", "改善", "维持", "下降", "可控", "恶化", "稳定", "好转",
            "向好", "达成", "夯实", "守牢", "可控", "缓和", "平缓", "稳定", "下降",
            "未增", "略短", "简化", "小升", "略减", "略提", "小扩", "慢升", "渐增",
            "稳固", "增加", "保持", "渐建", "渐成", "恢复", "保底", "提升", "到位",
            "增强", "落实", "建立", "回升", "稳定", "平复", "软化", "提高"

        ]
    },
    "损耗风险词汇（数值越高=次生损耗越大）": {
        "loss_high": [

            "重复信访", "抵触", "重复", "越级", "激化", "舆情", "冲突", "对抗", "积案", "堵塞",
            "反复", "压力", "困难", "鸿沟", "破裂", "对立", "极端", "隐患", "遗留", "偏差",
            "维权", "争议", "不佳", "失败", "破产", "加剧", "紧张", "恶化", "缺失", "沉重",
            "无着", "不公", "排斥", "标签", "污名", "否定", "失败", "冲突", "分化", "激化",
            "涣散", "不力", "缺位", "不足", "形式", "无效", "失信", "困难", "下降", "受损",
            "损害", "越级投诉", "矛盾激化", "舆情风险"
        ],
        "loss_mid": [

            "部门协作不畅", "信息传递失真", "响应速度变慢", "波动", "积累", "繁琐", "较长",
            "增大", "不足", "困难", "普及", "受损", "萌芽", "冲突", "遗留", "偏差", "未解",
            "偏低", "滞后", "不佳", "率低", "紧张", "疏远", "欠佳", "下降", "困难", "不足",
            "不均", "偏见", "较低", "不顺", "困难", "低效", "不足", "薄弱", "缓慢", "反复",
            "微降", "变样", "打折", "微损", "动摇", "增加", "较低", "升高", "加大", "拖延",
            "争议", "不一", "混乱", "不畅", "失真", "变慢"
        ],
        "loss_low": [

            "响应速度正常", "稳定", "轻微", "平稳", "少量", "略降", "稍增", "较小", "不便",
            "较弱", "可修", "摩擦", "个别", "微偏", "待解", "可调", "略慢", "一般", "观察",
            "偶有", "较少", "波动", "持平", "够用", "有保", "均等", "存在", "正常", "平缓",
            "可控", "正常", "够用", "到位", "推进", "较少", "微调", "稳定", "兑现", "良好",
            "信任", "可控", "增加", "顺畅", "到位", "及时", "统一", "清晰", "可用", "顺畅",
            "准确", "正常", "全程留痕、及时响应", "深入"
        ],
    },
    "导入的excel名称(在同一文件夹中)":"nvivo_coding_output2(1).xlsx",
    "导出的excel名字":"dd.xlsx",
    "校准模式(tok/fine(细校准)或tok/coarse(粗校准)":"tok/fine",
    "成本类权重":3,
    "收益类权重":2,
    "损耗类权重":1,
    "精细度(必须大于0)":3,
    "微分方程组初始值":{
        "政府初始选择刚性管控策略的概率":0.2,
        "政府初始选择纯线下服务策略的概率":0.3,
        "政府初始选择数智融合概率":0.5,
    },
    "博弈迭代周期":{
        "初始":0,
        "结束":100
    },
    "双主体演化博弈策略演化轨迹图":"game_evolution.png",
    "提取博弈输出时序变量CSV":"game_output_timeseries.csv",
    "存量":100,
    "sd_仿真月数":24,
    "CPU":8
}

results=[]

#读取配置
class ConfigData:
    _instance = None
    _initialized=False
    config={}
    def __new__(cls):
        if  cls._instance is None:
            cls._instance = super(ConfigData, cls).__new__(cls)
        return cls._instance
    def __init__(self):
        if ConfigData._initialized :
            return
        ConfigData.config=self.load_config(CONFIG_PATH)
        ConfigData._initialized=True

    @staticmethod
    @st.cache_resource
    def load_config(data) -> dict:
        if not os.path.exists(data):
            with open(data, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIGS, f, ensure_ascii=False, indent=4)
            return DEFAULT_CONFIGS.copy()
        try:
            with open(data, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            print("加载")
            return ConfigData.merge(user_config, DEFAULT_CONFIGS)
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            print("配置文件损坏或读取失败，已重新加载上一次配置")
            with open(data, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIGS, f, ensure_ascii=False, indent=4)
            return DEFAULT_CONFIGS.copy()

#递归和并
    @staticmethod
    def merge(default: dict, user: dict) -> dict:
        """递归合并配置"""
        result = default.copy()
        for k, v in result.items():
            if k in user and isinstance(user[k], dict) and isinstance(v, dict):
                user[k] = ConfigData.merge(v,user[k])
            else:
                user[k] = v
        Tool.write_sys_opt_log("加载配置成功")
        return user

#存档
def main():
    while True:
        try:
            print("1. 存档 (备份当前配置)")
            print("2. 读档 (恢复备份配置)")
            print("3. 查看配置内容")
            print("4. 查看备份内容")
            print("5. 退出")
            opt=int(input("请选择操作"))
            if opt == 1:
                # 存档
                if os.path.exists(CONFIG_PATH):

                    # 备份
                    shutil.copy2(CONFIG_PATH, CUN_DANG)
                    print(f" 最新备份已更新")
                    Tool.write_sys_opt_log("存档成功")
                else:
                    print(" 配置文件不存在，请先运行重新获取默认配置")
                    Tool.write_sys_opt_log("存档失败")

            elif opt == 2:
                # 读档
                if os.path.exists(CUN_DANG)and os.path.exists(CONFIG_PATH):
                    # 从备份恢复
                    shutil.copy2(CUN_DANG, CONFIG_PATH)
                    print(f" 配置已从备份恢复: {CUN_DANG}")
                    Tool.write_sys_opt_log("读档成功")
                else:
                    print(" 备份文件或配置文件不存在，请先存档或创建默认配置")
                    Tool.write_sys_opt_log("读档失败")

            elif opt == 3:
                # 查看当前配置
                if os.path.exists(CONFIG_PATH):
                    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    print(" 当前配置:")
                    print(json.dumps(config, ensure_ascii=False, indent=2))
                else:
                    print("配置文件不存在")

            elif opt == 4:
                # 查看备份配置
                if os.path.exists(CUN_DANG):
                    with open(CUN_DANG, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    print(" 备份配置:")
                    print(json.dumps(config, ensure_ascii=False, indent=2))
                else:
                    print("备份文件不存在")

            elif opt == 5:
                print("👋 退出程序")
                break

            else:
                print(" 无效选项，请输入 1-5")
                Tool.write_err_log("你输入了无效值")

        except ValueError:
            print(" 请输入正确的数值")
            Tool.write_err_log("你在运行config文件时输入了非数值")

if __name__ =="__main__":
    main()
#配置

config_main()
obj_config=ConfigData()
DEFAULT_CONFIG=obj_config.config

#成本类词汇
cost_high =set(DEFAULT_CONFIG["成本类词汇(数值越高=成本越大)"]["cost_high"])
cost_mid = set(DEFAULT_CONFIG["成本类词汇(数值越高=成本越大)"]["cost_mid"])
cost_low =set(DEFAULT_CONFIG["成本类词汇(数值越高=成本越大)"]["cost_low"])

# 收益类词汇（数值越高=收益越高）
gain_high =set(DEFAULT_CONFIG["收益类词汇（数值越高=收益越高）"]["gain_high"])
gain_mid =set(DEFAULT_CONFIG["收益类词汇（数值越高=收益越高）"]["gain_mid"])
gain_low = set(DEFAULT_CONFIG["收益类词汇（数值越高=收益越高）"]["gain_low"])

# 损耗风险词汇（数值越高=次生损耗越大）
loss_high = set(DEFAULT_CONFIG["损耗风险词汇（数值越高=次生损耗越大）"]["loss_high"])
loss_mid = set(DEFAULT_CONFIG["损耗风险词汇（数值越高=次生损耗越大）"]["loss_mid"])
loss_low = set(DEFAULT_CONFIG["损耗风险词汇（数值越高=次生损耗越大）"]["loss_low"])

#名称
date=DEFAULT_CONFIG["保留的策略名称（三级编码）"]

#关于各部分的衔接文件
excel_name=DEFAULT_CONFIG["导入的excel名称(在同一文件夹中)"]
ff=DEFAULT_CONFIG["导出的excel名字"]
tu_p=DEFAULT_CONFIG["双主体演化博弈策略演化轨迹图"]
sd_csv=DEFAULT_CONFIG["提取博弈输出时序变量CSV"]

#在hanlp中的词性分析和模型
key=DEFAULT_CONFIG["校准模式(tok/fine(细校准)或tok/coarse(粗校准)"]

#权重
C=DEFAULT_CONFIG["高权重"]
R=DEFAULT_CONFIG["中权重"]
L=DEFAULT_CONFIG["低权重"]

#hanlp算出来的小数点位数
round_data=DEFAULT_CONFIG["精细度(必须大于0)"]

"""仿真基础设置"""

#博弈迭代周期100期
ts_pan = [DEFAULT_CONFIG["博弈迭代周期"]["初始"], DEFAULT_CONFIG["博弈迭代周期"]["结束"]]

#微分方程组初始值
init_cond = [DEFAULT_CONFIG["微分方程组初始值"]["政府初始选择刚性管控策略的概率"], DEFAULT_CONFIG["微分方程组初始值"]["政府初始选择纯线下服务策略的概率"],DEFAULT_CONFIG["微分方程组初始值"]["政府初始选择数智融合概率"]]


""""""

#未化解信访矛盾存量
init_stock=DEFAULT_CONFIG["存量"]

#长期仿真循环时间
sim_month=DEFAULT_CONFIG["sd_仿真月数"]

#hanlp系统分词的CPU核数
CPU=DEFAULT_CONFIG["CPU"]

#演化博弈系数
loss_coefficient=DEFAULT_CONFIG["演化博弈系数"]["损耗系数"]

text=True
text_hanlp=True

#演化博弈公式参数
alpha=DEFAULT_CONFIG["演化博弈公式参数"]["刚性收益比例"]
beta=DEFAULT_CONFIG["演化博弈公式参数"]["线下收益比例"]
gamma=DEFAULT_CONFIG["演化博弈公式参数"]["刚柔融合额外收益"]
delta_R=DEFAULT_CONFIG["演化博弈公式参数"]["单位比例收益值"]


#相对噪音的随机值
mu = DEFAULT_CONFIG["噪音程度设置"]["中心值"]
sigma = DEFAULT_CONFIG["噪音程度设置"]["方差"]
bound_low =  DEFAULT_CONFIG["噪音程度设置"]["取的最大负面值"]
bound_high =  DEFAULT_CONFIG["噪音程度设置"]["取的正面最大值"]

#博弈次数
counts=DEFAULT_CONFIG["噪音博弈次数"]





#噪音
perceptual_noise=True
initial_noise=True
noise=False










import json
import os
import matplotlib.pyplot as plt
import warnings
from log import Tool
import shutil
import decorators

# 检索配置文件
CK_CONFIG_PATH=os.path.join(os.path.dirname(__file__), "ck_config.json")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
CUN_DANG = os.path.join(os.path.dirname(__file__), "存档.json")
COST_PATH = os.path.join(os.path.dirname(__file__), "领域词典.json")

@decorators.validate_and_catch("加载配置")
def config_main():
    #消除警告
    warnings.filterwarnings('ignore')

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

#配置文件备份
COST_DATA={
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
}

DEFAULT_CONFIGS={
    "保留的策略名称（三级编码）":[
        "纯刚性管控",
        "纯线下柔性服务",
        "数智刚柔融合"],
    "锚点":{
        "成本":"需要投入大量人力物力财力，行政成本非常高",
        "收益":"能够快速闭环解决诉求，大幅提升服务效能",
        "损耗":"容易引发重复信访越级上访，造成严重次生矛盾"
    },

    "高权重":3,
    "中权重":2,
    "低权重":1,
    "精细度(必须大于0)":3.0,
    "微分方程组初始值":{
        "政府初始选择刚性管控策略的概率":0.2,
        "政府初始选择纯线下服务策略的概率":0.3,
        "政府初始选择数智融合概率":0.5,
    },
    "博弈迭代周期":{
        "初始":0,
        "结束":100
    },
    "演化博弈系数":{
        "损耗系数":10
    },
    "演化博弈公式参数":{
        "刚性收益比例":0.4,
        "线下收益比例":0.6,
        "刚柔融合额外收益":0.0,
        "单位比例收益值":0.1,
    },
    "噪音程度设置": {
        "中心值": 0.0,
        "方差": 1.0,
        "取的最大负面值": 0.5,
        "取的正面最大值": 0.5
    },
    "噪音博弈次数":100,
    "模型值设置":{
        "开始值":0,
        "时间值":100,
        "曲线点数":1000
    },
    "存量":1000,
    "sd_仿真月数":24,
    "CPU":8,
    "拥挤效应阈值":0.5,
    "导入的excel名称(在同一文件夹中)":r"素材\nvivo_coding_output2_split.xlsx",
    "导出的excel名字":"dd.xlsx",
    "双主体演化博弈策略演化轨迹图": "game_evolution.png",
    "提取博弈输出时序变量CSV": "game_output_timeseries.csv",

}

results = []

# 读取配置
class ConfigData:
    _instance = None
    _initialized=False
    config={}
    cost={}
    def __new__(cls):
        if  cls._instance is None:
            cls._instance = super(ConfigData, cls).__new__(cls)
        return cls._instance
    def __init__(self):
        if ConfigData._initialized :
            return
        ConfigData.config=self.load_config(CONFIG_PATH)
        ConfigData.cost=self.load_config(COST_PATH)
        ConfigData._initialized=True

    @staticmethod
    def load_config(data) -> dict:
        if not os.path.exists(data):
            with open(data, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIGS, f, ensure_ascii=False, indent=4)
            return DEFAULT_CONFIGS.copy()
        try:
            with open(data, "r", encoding="utf-8") as f:
                user_config = json.load(f)
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

#配置

config_main()
obj_config=ConfigData()
DEFAULT_CONFIG=obj_config.config
COST_DATAS=obj_config.cost

#成本类词汇
cost_high =set(COST_DATA["成本类词汇(数值越高=成本越大)"]["cost_high"])
cost_mid = set(COST_DATA["成本类词汇(数值越高=成本越大)"]["cost_mid"])
cost_low =set(COST_DATA["成本类词汇(数值越高=成本越大)"]["cost_low"])

# 收益类词汇（数值越高=收益越高）
gain_high =set(COST_DATA["收益类词汇（数值越高=收益越高）"]["gain_high"])
gain_mid =set(COST_DATA["收益类词汇（数值越高=收益越高）"]["gain_mid"])
gain_low = set(COST_DATA["收益类词汇（数值越高=收益越高）"]["gain_low"])

# 损耗风险词汇（数值越高=次生损耗越大）
loss_high = set(COST_DATA["损耗风险词汇（数值越高=次生损耗越大）"]["loss_high"])
loss_mid = set(COST_DATA["损耗风险词汇（数值越高=次生损耗越大）"]["loss_mid"])
loss_low = set(COST_DATA["损耗风险词汇（数值越高=次生损耗越大）"]["loss_low"])

#锚点
anchor_C=DEFAULT_CONFIG["锚点"]["成本"]
anchor_R=DEFAULT_CONFIG["锚点"]["收益"]
anchor_L=DEFAULT_CONFIG["锚点"]["损耗"]

# ==================== 成本维度 ====================
ANCHOR_COST_HIGH = [
    "措施需要投入大量基层人力逐户摸排走访，行政成本极高",
    "导致财政经费严重超支，平台建设和维护费用难以承受",
    "需要协调多个部门反复开会沟通，时间成本和协调成本巨大",
    "需要配备专职人员全天候值守，人力消耗远超现有编制承受能力",
    "需要购置大量硬件设备并持续投入运维经费，资金缺口极为突出",
    "需要反复填报多套系统台账，基层工作人员大量时间被行政事务占用",
    "要求工作人员频繁下乡入户，差旅和人力成本长期居高不下",
    "需要外聘第三方机构提供专业支持，服务采购费用持续增加",
    "涉及跨层级、跨部门反复审批，流程冗长导致行政效率损耗严重"
]

ANCHOR_COST_LOW = [
    "措施几乎不需要新增经费投入，完全依靠现有人员即可运转",
    "通过流程优化实现，没有任何额外的行政开支和人力成本",
    "不需要购置新设备或建设新平台，实施成本极低",
    "由现有基层工作人员兼职完成，不产生额外费用",
    "主要依靠线上自助办理，大幅节约了人力和场地成本",
    "利用已有系统和数据即可完成，无需额外投入开发资源",
    "通过整合现有服务窗口实现，不增加任何新的财政负担",
    "完全依托村居现有场地和人员，没有产生新的运维支出",
    "通过简化材料、减少环节实现，不增加基层和群众负担"
]

# ==================== 收益维度 ====================
ANCHOR_GAIN_HIGH = [
    "能够快速闭环解决退役军人合理诉求，服务效能显著提升",
    "大幅提高了退役军人满意度，有效增强了政民互信和群众获得感",
    "形成了全流程留痕和及时响应的机制，矛盾化解效率极高",
    "成功化解了长期遗留的信访积案，治理成效突出",
    "在很多个方面取到了非常大的作用"
    "推动退役军人服务从被动应对转向主动预防，源头治理效果明显",
    "实现了让数据多跑路、群众少跑腿，办事体验明显改善",
    "通过精准识别和主动推送，大幅提高了服务对象的获得感",
    "整合了多部门资源，实现一站式办理，有效减少了群众往返",
    "显著降低了重复信访和越级上访的发生率，治理秩序明显好转"
]

ANCHOR_GAIN_LOW = [
    "对解决退役军人实际诉求几乎没有帮助，服务效能低下",
    "流于形式，未能改善退役军人的服务体验和满意度",
    "响应迟缓且缺乏闭环管理，实际问题长期得不到解决",
    "未能有效化解矛盾，反而增加了退役军人的不满情绪",
    "执行效果甚微，退役军人的获得感和信任度没有明显提升",
    "停留在表面留痕，对实际办事效率提升没有实质作用",
    "未能覆盖高频需求，群众仍需反复跑腿才能办成事",
    "缺乏后续跟踪机制，问题解决效果难以持续巩固",
    "增加了基层工作量，却未能带来相应的服务改善"
]

# ==================== 损耗维度 ====================
ANCHOR_LOSS_HIGH = [
    "容易引发退役军人重复信访和越级上访，造成严重的次生矛盾",
    "执行不当范围的退役军人不满和集体投诉，社会负面影响剧烈",
    "造成了新的历史遗留问题，矛盾积压日益严重",
    "导致退役军人对基层治理的信任度急剧下降，干群关系紧张",
    "执行过程中出现推诿扯皮，群众合理诉求长期得不到回应",
    "导致不同群体之间待遇攀比，引发新的不公平感和心理落差",
    "因操作门槛过高，将老年退役军人排斥在数字化服务之外",
    "未能有效化解矛盾，反而使重复信访和越级访数量持续上升"
]

ANCHOR_LOSS_LOW = [
    "基本不会引发新的矛盾，退役军人群体反应平稳",
    "实施后没有出现重复信访或舆情风险，社会面稳定",
    "运行平稳，未产生任何负面社会影响或次生问题",
    "得到了退役军人的普遍理解和支持，不存在激化矛盾的风险",
    "执行过程中未出现投诉和争议，治理环境保持和谐稳定",
    "充分考虑老年群体需求，未出现因操作困难引发的抵触情绪",
    "实施后群众满意度稳中有升，未出现新的信访或舆情事件",
    "有效化解了潜在矛盾，没有产生任何次生风险或遗留问题",
    "推行后基层反映良好，未出现服务对象投诉或不满情绪"
]



#名称
date=DEFAULT_CONFIG["保留的策略名称（三级编码）"]

#关于各部分的衔接文件
excel_name=DEFAULT_CONFIG["导入的excel名称(在同一文件夹中)"]
ff=DEFAULT_CONFIG["导出的excel名字"]
tu_p=DEFAULT_CONFIG["双主体演化博弈策略演化轨迹图"]
sd_csv=DEFAULT_CONFIG["提取博弈输出时序变量CSV"]

#权重
C=DEFAULT_CONFIG["高权重"]
R=DEFAULT_CONFIG["中权重"]
L=DEFAULT_CONFIG["低权重"]

#hanlp算出来的小数点位数
round_data=DEFAULT_CONFIG["精细度(必须大于0)"]

#博弈迭代周期100期
ts_pan = [DEFAULT_CONFIG["博弈迭代周期"]["初始"], DEFAULT_CONFIG["博弈迭代周期"]["结束"]]

#微分方程组初始值
init_cond = [DEFAULT_CONFIG["微分方程组初始值"]["政府初始选择刚性管控策略的概率"], DEFAULT_CONFIG["微分方程组初始值"]["政府初始选择纯线下服务策略的概率"],DEFAULT_CONFIG["微分方程组初始值"]["政府初始选择数智融合概率"]]

#未化解信访矛盾存量
init_stock=DEFAULT_CONFIG["存量"]

#长期仿真循环时间
sim_month=DEFAULT_CONFIG["sd_仿真月数"]

#hanlp系统分词的CPU核数
CPU=DEFAULT_CONFIG["CPU"]

#演化博弈系数
loss_coefficient=DEFAULT_CONFIG["演化博弈系数"]["损耗系数"]

#粗分还是细分
text=False

#剔除标点符号
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

#噪音开关
perceptual_noise=False
initial_noise=False
noise=False

#模型值设置
start=DEFAULT_CONFIG["模型值设置"]["开始值"]
stop=DEFAULT_CONFIG["模型值设置"]["时间值"]
num=DEFAULT_CONFIG["模型值设置"]["曲线点数"]

#数智融合的久公式
R3_=True

#拥挤效应的阈值
ovr=DEFAULT_CONFIG["拥挤效应阈值"]


if __name__ =="__main__":
    main()








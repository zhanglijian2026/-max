import pandas as pd
import config
from multiprocessing import Pool
from log import Tool
import decorators
import shutil
import re
import hanlp

# 加载模型
sts_model = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)
# 扩展否定词表，覆盖常见否定表达
_NEGATION_WORDS = {'不', '没', '无', '非', '未', '别', '莫', '毫无', '并非', '不是', '没有', '缺乏', '缺少', '无需', '免于'}

# 分句工具函数（借鉴论文层次化解构思想
def split_sentences(text: str) -> list:
    """
    按中文标点将长文本拆分为独立子句
    借鉴论文中"单词→语句→文本"的层次化解构思想
    """
    if not text or len(text.strip()) < 2:
        return []
    # 按句号、感叹号、问号、分号、换行符拆分
    raw_sent = re.split(r'[。！？；\n]', text)
    # 过滤掉空句和过短的句子（小于3个字符）
    return [s.strip() for s in raw_sent if len(s.strip()) >= 3]



def has_negation(text: str) -> bool:
    """检测文本中是否包含否定词"""
    for word in _NEGATION_WORDS:
        if word in text:
            return True
    return False


# =========================== 4. 核心：计算单个子句与锚点的相似度 ===========================
def calc_sentence_similarity(sentence: str, anchor_text: str) -> float:
    """
    使用 HanLP STS 模型计算单个子句与锚点句子的语义相似度
    返回 0~1 之间的相似度值
    """
    if not sentence or len(sentence.strip()) < 2:
        return 0.0
    # HanLP STS 输入：列表的列表，每个子列表是一对文本
    # 返回值：列表，每个元素是对应文本对的相似度
    result = sts_model([(sentence, anchor_text)])
    # result 格式为 [0.85] 或 [0.92]
    sim = result[0] if result else 0.0
    return max(0.0, min(1.0, sim))  # 截断到 [0, 1]


def calc_sentence_score(sentence: str, anchor_text: str, anchor_reverse_text: str = None) -> float:
    """
    计算单个子句的得分，支持双向锚点参照
    - 如果提供了反向锚点，采用相对竞争分数
    - 如果未提供，直接映射相似度 × 10
    """

    if not sentence or len(sentence.strip()) < 2:
        return 0.0

    # 正向相似度
    sim_high = calc_sentence_similarity(sentence, anchor_text)

    # 判断是否存在否定词（影响分数）
    neg_factor = 0.6 if has_negation(sentence) else 1.0

    # 如果提供了反向锚点，使用双向参照
    if anchor_reverse_text:
        sim_low = calc_sentence_similarity(sentence, anchor_reverse_text)
        # 相对竞争分数：当文本偏向正向时，分数接近10；偏向反向时，分数接近0
        total = sim_high + sim_low
        if total == 0:
            base_score = 5.0  # 都不像时，取中间值
        else:
            base_score = (sim_high / total) * 10
    else:
        # 单锚点直接映射，乘以否定词折扣
        base_score = sim_high * 10 * neg_factor

    return round(max(0.0, min(10.0, base_score)), 2)


# 完整文本打分：分句聚合 + 双向锚点
def calc_text_score(text: str, anchor_high: str, anchor_low: str = None) -> float:

    # 分句
    sentences = split_sentences(text)
    if not sentences:
        return 0.0

    # 每个子句打分，取最高分（代表最极端的倾向）
    scores = [calc_sentence_score(s, anchor_high, anchor_low) for s in sentences]
    #scores= [calc_sentence_score(text, anchor_high, anchor_low)]
    return max(scores) if scores else 0.0


# 读取锚点
# 三个维度的正向锚点（极端高值）
ANCHOR_COST_HIGH = getattr(config, 'anchor_C', '需要投入大量人力物力财力，行政成本极高，财政负担无法承受')
ANCHOR_GAIN_HIGH = getattr(config, 'anchor_R', '能够快速闭环解决退役军人诉求，大幅提升服务效能和群众信任度')
ANCHOR_LOSS_HIGH = getattr(config, 'anchor_L', '容易引发重复信访越级上访和严重舆情风波，造成剧烈的次生矛盾')

# 三个维度的反向锚点（极端低值）——用于双向参照
ANCHOR_COST_LOW = getattr(config, 'anchor_C_low', '几乎不需要任何经费投入，完全依靠现有资源运转')
ANCHOR_GAIN_LOW = getattr(config, 'anchor_R_low', '对退役军人诉求解决没有明显效果，服务效能基本没有提升')
ANCHOR_LOSS_LOW = getattr(config, 'anchor_L_low', '几乎不会引发任何新的社会矛盾，社会面反应极其平稳')


# 对外接口函数
def calc_cost(text: str) -> float:
    return calc_text_score(text, ANCHOR_COST_HIGH, ANCHOR_COST_LOW)

def calc_gain(text: str) -> float:
    return calc_text_score(text, ANCHOR_GAIN_HIGH, ANCHOR_GAIN_LOW)

def calc_loss(text: str) -> float:
    return calc_text_score(text, ANCHOR_LOSS_HIGH, ANCHOR_LOSS_LOW)

def calc_non_prob(text: str) -> float:
    """
    基于损耗锚点计算非协同概率
    与损耗锚点相似度越高，非协同概率越高
    """
    if not text or len(text.strip()) < 2:
        return 0.0

    # 直接用正反向锚点做参照
    sim_high = calc_sentence_similarity(text, ANCHOR_LOSS_HIGH)
    sim_low = calc_sentence_similarity(text, ANCHOR_LOSS_LOW)
    total = sim_high + sim_low
    if total == 0:
        prob = 0.5
    else:
        prob = sim_high / total
    return round(max(0.0, min(1.0, prob)), 4)


# 数据清洗函数
@decorators.validate_and_catch(func_name="数据清洗")
def filter_strategy():
    df = pd.read_excel(config.excel_name)
    print("Excel 数据读取完成：")
    Tool.write_sys_opt_log("成功导入数据")

    data = df.copy()
    if "名称" in df.columns and "编码文本" in df.columns:
        name_col = "名称"
        text_col = "编码文本"
        Tool.write_sys_opt_log("成功识别 NVivo 列名")
    else:
        raise ValueError("Excel 中缺少'名称'或'编码文本'列")

    data = data.dropna(subset=[name_col, text_col])
    data[name_col] = data[name_col].astype(str).str.strip()
    data[text_col] = data[text_col].astype(str).str.strip()

    keep_lower = [n.lower() for n in config.date]
    data = data[data[name_col].str.lower().isin(keep_lower)]
    data = data[~data[name_col].str.lower().isin(["nan", "none", "null", ""])]
    data = data[data[text_col].str.len() >= 5]
    data = data.drop_duplicates(subset=[name_col, text_col])
    data = data.reset_index(drop=True)

    print(f"保留策略: {data[name_col].unique().tolist()}")
    print(f"共 {len(data)} 条文本")
    Tool.write_sys_opt_log("数据清洗完成")

    return data, name_col, text_col


@decorators.validate_and_catch(func_name="获取原始文本")
def load_texts(df, name_col, text_col):
    """直接读取原始文本，不分词"""
    results = []
    for _, row in df.iterrows():
        results.append({
            "策略": row[name_col],
            "文本": row[text_col]  # 原始字符串
        })
    return results


# 核心计算逻辑
def count_with_labels(batch):
    batch_results = []
    for item in batch:
        text = item["文本"]

        # 三个维度的得分
        c = calc_cost(text)
        r = calc_gain(text)
        l = calc_loss(text)

        # 非协同概率（基于损耗锚点）
        prob_neg = calc_non_prob(text)
        prob_pos = 1 - prob_neg

        batch_results.append({
            "名称": item["策略"],
            "成本C": c,
            "收益R": r,
            "次生损耗L": l,
            "协同诉求表达概率": round(prob_pos, 4),
            "非协同诉求表达概率": round(prob_neg, 4)
        })
    return batch_results


@decorators.validate_and_catch(func_name="分割列表")
def chunkify(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


# 主程序
def main(name=None):
    if name:
        shutil.copy2(config.CK_CONFIG_PATH, config.CONFIG_PATH)

    data, name_col, text_col = filter_strategy()
    text_list = load_texts(data, name_col, text_col)
    chunks = chunkify(text_list, config.CPU)

    with Pool(processes=config.CPU) as pool:
        results = pool.map(count_with_labels, chunks)

    flat_results = []
    for batch in results:
        flat_results.extend(batch)

    result_df = pd.DataFrame(flat_results)
    summary = result_df.groupby("名称").mean().round(config.round_data)
    print("\n量化结果汇总：")
    print(summary)
    summary.to_excel(config.ff, index=True)
    Tool.write_sys_opt_log("成功导出 Excel")


if __name__ == "__main__":
    main()
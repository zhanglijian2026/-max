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

# 否定表达
_NEGATION_WORDS = { '不是', '没有', '不会', '不能', '无法', '未能','缺乏', '缺少', '不足', '尚未', '并非', '毫无', '无需' ,'毫无', '并非',   '免于'}

# 分句工具函数，借鉴论文层次化解构思想
def split_sentences(text: str) -> list:
    raw_sent = re.split(r'[。！？；\n]', text)
    return [s.strip() for s in raw_sent if len(s.strip()) >= 2]

#判断是否存在否定次
def has_negation(text: str) -> bool:
    for word in _NEGATION_WORDS:
        if word in text:return True
    return False

#  核心：计算单个子句与锚点的相似度
def calc_sentence_similarity(sentence: str, anchor_text: list) -> float:
    scores = [sts_model([(sentence, word)])[0] for word in anchor_text]
    sim = max(scores) if scores else 0.0
    return max(0.0, min(1.0, sim))
    # 截断到 [0, 1]

#极端聚合
def calc_sentence_score(sentence: str, anchor_text: list, anchor_reverse_text: list = None) :
    # 正向相似度
    sim_high = calc_sentence_similarity(sentence, anchor_text)

    # 判断是否存在否定词（影响分数）
    neg_factor = 0.6 if has_negation(sentence) else 1.0

    # 如果提供了反向锚点，使用双向参照
    sim_low = calc_sentence_similarity(sentence, anchor_reverse_text)
    # 相对竞争分数：当文本偏向正向时，分数接近10；偏向反向时，分数接近0
    total = sim_high + sim_low
    if total == 0:
        return None
    else:
        base_score = (sim_high / total) * 10


    return round(max(0.0, min(10.0, base_score*neg_factor)), 2)

# 完整文本打分：分句聚合 + 双向锚点
def calc_text_score(text: str, anchor_high: list, anchor_low: list = None) :

    # 分句
    sentences = split_sentences(text)
    if not sentences:
        return 0.0

    # 每个子句打分，取最高分（代表最极端的倾向）
    scores = [calc_sentence_score(s, anchor_high, anchor_low) for s in sentences]
    #scores= [calc_sentence_score(text, anchor_high, anchor_low)]
    return max(scores) if scores else 0.0

# 对外接口函数
#c
def calc_cost(text: str) -> float:
    return calc_text_score(text, config.ANCHOR_COST_HIGH, config.ANCHOR_COST_LOW)
#r
def calc_gain(text: str) -> float:
    return calc_text_score(text, config.ANCHOR_GAIN_HIGH, config.ANCHOR_GAIN_LOW)
#l
def calc_loss(text: str) -> float:
    return calc_text_score(text, config.ANCHOR_LOSS_HIGH, config.ANCHOR_LOSS_LOW)

#概率
def calc_non_prob(text: str) -> float:
    if not text or len(text.strip()) < 2:
        return 0.0

    # 直接用正反向锚点做参照
    sim_high = calc_sentence_similarity(text, config.ANCHOR_LOSS_HIGH)
    sim_low = calc_sentence_similarity(text, config.ANCHOR_LOSS_LOW)
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
    data = data[data[text_col].str.len() >= 2]
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
    #x=count_with_labels( text_list)
    chunks = chunkify(text_list, config.CPU)

    with Pool(processes=config.CPU) as pool:results = pool.map(count_with_labels, chunks)

    flat_results = []
    for batch in results:
        flat_results.extend(batch)

    result_df = pd.DataFrame(flat_results)
    print(result_df)
    result_df.to_excel("./hh.xlsx", index=True)
    summary = result_df.groupby("名称").mean().round(config.round_data)
    print("\n量化结果汇总：")
    print(summary)
    summary.to_excel(config.ff, index=True)
    Tool.write_sys_opt_log("成功导出 Excel")


if __name__ == "__main__":
    main()
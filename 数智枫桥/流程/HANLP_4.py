import pandas as pd
import numpy as np
import re
import config
from log import Tool
import decorators
import shutil
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

_ANCHOR_COST_HIGH_VEC = _model.encode(config.ANCHOR_COST_HIGH)
_ANCHOR_COST_LOW_VEC  = _model.encode(config.ANCHOR_COST_LOW)
_ANCHOR_GAIN_HIGH_VEC = _model.encode(config.ANCHOR_GAIN_HIGH)
_ANCHOR_GAIN_LOW_VEC  = _model.encode(config.ANCHOR_GAIN_LOW)
_ANCHOR_LOSS_HIGH_VEC = _model.encode(config.ANCHOR_LOSS_HIGH)
_ANCHOR_LOSS_LOW_VEC  = _model.encode(config.ANCHOR_LOSS_LOW)



def has_negation(text: str) -> bool:
    return False

def split_sentences(text: str) -> list:
    if not text or len(text.strip()) < 2:
        return []
    raw = re.split(r'[。！？；\n]', text)
    return [s.strip() for s in raw if len(s.strip()) >= 2]

# 相似度计算
def calc_similarity_with_anchors(sent_vec, anchor_vecs):
    """计算一个子句向量与一组锚点向量的最大余弦相似度"""
    sims = cosine_similarity(sent_vec.reshape(1, -1), anchor_vecs)[0]
    return float(np.max(sims)) if len(sims) > 0 else 0.0

def calc_sentence_score(sent_vec, anchor_high_vec, anchor_low_vec, neg_flag):
    """计算单个子句在某维度上的得分，返回 0~10 或 None"""
    sim_high = calc_similarity_with_anchors(sent_vec, anchor_high_vec)
    sim_low  = calc_similarity_with_anchors(sent_vec, anchor_low_vec)

    total = sim_high + sim_low
    if total == 0:
        return None
    base_score = (sim_high / total) * 10

    # 否定词折扣
    if neg_flag:
        base_score *= 0.6

    return round(max(0.0, min(10.0, base_score)), 4)

def calc_text_score(sent_vecs, anchor_high_vec, anchor_low_vec, neg_flags):
    """整段文本得分：子句间取最大有效分"""
    scores = []
    for vec, neg in zip(sent_vecs, neg_flags):
        s = calc_sentence_score(vec, anchor_high_vec, anchor_low_vec, neg)
        if s is not None:
            scores.append(s)
    if not scores:
        return np.nan
    return max(scores)

def calc_non_prob(sent_vecs, neg_flags):
    """基于损耗维度正反向竞争计算非协同概率"""
    probs = []
    for vec, neg in zip(sent_vecs, neg_flags):
        sim_high = calc_similarity_with_anchors(vec, _ANCHOR_LOSS_HIGH_VEC)
        sim_low  = calc_similarity_with_anchors(vec, _ANCHOR_LOSS_LOW_VEC)
        total = sim_high + sim_low
        if total > 0:
            probs.append(sim_high / total)
    if not probs:
        return 0.5
    return round(max(0.0, min(1.0, max(probs))), 4)

# 数据清洗与主流程
@decorators.validate_and_catch(func_name="数据清洗")
def filter_strategy():
    df = pd.read_excel(config.excel_name)
    print("Excel 数据读取完成")
    Tool.write_sys_opt_log("成功导入数据")

    data = df.copy()
    if "名称" in df.columns and "编码文本" in df.columns:
        name_col, text_col = "名称", "编码文本"
    else:
        raise ValueError("Excel 中缺少'名称'或'编码文本'列")

    data = data.dropna(subset=[name_col, text_col])
    data[name_col] = data[name_col].astype(str).str.strip()
    data[text_col] = data[text_col].astype(str).str.strip()

    keep = [n.lower() for n in config.date]
    data = data[data[name_col].str.lower().isin(keep)]
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
    results = []
    for _, row in df.iterrows():
        results.append({"策略": row[name_col], "文本": row[text_col]})
    return results

# 主程序
def main(name=None):
    if name:shutil.copy2(config.CK_CONFIG_PATH, config.CONFIG_PATH)

    data, name_col, text_col = filter_strategy()
    text_list = load_texts(data, name_col, text_col)

    # 收集所有子句，批量编码
    all_sentences = []
    for item in text_list:
        sent = split_sentences(item["文本"])
        all_sentences.extend(sent)

    print(f"共 {len(all_sentences)} 个子句，开始批量编码...")
    all_vecs = _model.encode(all_sentences, batch_size=32, show_progress_bar=True)

    # 将子句向量按文本分组
    vec_idx = 0
    results = []
    for item in text_list:
        sent = split_sentences(item["文本"])
        n = len(sent)
        sent_vecs = all_vecs[vec_idx: vec_idx + n]
        neg_flags = [has_negation(s) for s in sent]
        vec_idx += n

        c = calc_text_score(sent_vecs, _ANCHOR_COST_HIGH_VEC, _ANCHOR_COST_LOW_VEC, neg_flags)
        r = calc_text_score(sent_vecs, _ANCHOR_GAIN_HIGH_VEC, _ANCHOR_GAIN_LOW_VEC, neg_flags)
        l = calc_text_score(sent_vecs, _ANCHOR_LOSS_HIGH_VEC, _ANCHOR_LOSS_LOW_VEC, neg_flags)

        prob_neg = calc_non_prob(sent_vecs, neg_flags)
        prob_pos = 1 - prob_neg

        results.append({
            "名称": item["策略"],
            "成本C": c,
            "收益R": r,
            "次生损耗L": l,
            "协同诉求表达概率": round(prob_pos, 4),
            "非协同诉求表达概率": round(prob_neg, 4)
        })

    result_df = pd.DataFrame(results)
    print(result_df)
    result_df.to_excel("./h3.xlsx", index=False)

    # 汇总时自动跳过 NaN
    summary = result_df.groupby("名称").mean().round(config.round_data)
    print("\n量化结果汇总：")
    print(summary)
    summary.to_excel(config.ff, index=True)
    Tool.write_sys_opt_log("成功导出 Excel")

if __name__ == "__main__":
    main()
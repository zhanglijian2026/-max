import pandas as pd
import numpy as np
import config
import hanlp
from multiprocessing import Pool
from log import Tool
import decorators
import re
import shutil
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import inspect

#将词列表编码为向量矩阵
def _encode_word_list(words):

    if not words:
        return np.array([])
    return _semantic_model.encode(list(words))


_semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 预编码所有词库（只做一次，提升速度）
_COST_HIGH_VEC = _encode_word_list(config.cost_high)
_COST_MID_VEC = _encode_word_list(config.cost_mid)
_COST_LOW_VEC = _encode_word_list(config.cost_low)

_GAIN_HIGH_VEC = _encode_word_list(config.gain_high)
_GAIN_MID_VEC = _encode_word_list(config.gain_mid)
_GAIN_LOW_VEC = _encode_word_list(config.gain_low)

_LOSS_HIGH_VEC = _encode_word_list(config.loss_high)
_LOSS_MID_VEC = _encode_word_list(config.loss_mid)
_LOSS_LOW_VEC = _encode_word_list(config.loss_low)


#语义相识度
def _max_similarity(text_vec, anchor_vec, threshold=0.3):
    """
    计算文本向量与锚点向量库的最大余弦相似度
    Args:
        text_vec: 文本向量（1 x dim）
        anchor_vec: 锚点向量矩阵（n x dim）
        threshold: 相似度阈值，低于此值视为无响应
    Returns:
        float: 0~1 之间的最大相似度
    """
    if len(anchor_vec) == 0:
        return 0.0
    sims = cosine_similarity(text_vec, anchor_vec)[0]
    max_sim = np.max(sims)
    return max_sim if max_sim > threshold else 0.0


def _semantic_score(text_list, high_vec, mid_vec, low_vec, threshold=0.3):
    """
    语义相似度聚合评分（内部函数）
    Args:
        text_list: 分词后的词列表
        high_vec: 高层级锚点向量
        mid_vec: 中层级锚点向量
        low_vec: 低层级锚点向量
        threshold: 相似度阈值
    Returns:
        float: 0~10 量化分值
    """
    if not text_list:
        return 0.0

    # 拼接为连续文本（语义模型需要完整句子）
    text_str = ' '.join(text_list)
    if not text_str.strip():
        return 0.0

    # 文本向量化
    text_vec = _semantic_model.encode([text_str])

    # 计算三个层级的相似度
    sim_high = _max_similarity(text_vec, high_vec, threshold)
    sim_mid = _max_similarity(text_vec, mid_vec, threshold)
    sim_low = _max_similarity(text_vec, low_vec, threshold)

    # 加权聚合
    raw_score = sim_high * config.C + sim_mid * config.R + sim_low * config.L

    # 归一化到 0~10
    max_possible = config.C + config.R + config.L
    if max_possible == 0:
        return 0.0
    norm_score = np.clip(raw_score / max_possible * 10, 0, 10)
    return round(norm_score, 2)


# =========================== 4. 替换原来的 calc_text_score ===========================
@decorators.validate_and_catch(func_name="语义相似度打分（替代原calc_text_score）")
def calc_text_score(text: list, high_words, mid_words, low_words, threshold=0.3) -> float:
    """
    语义相似度量化函数（完全兼容原接口）
    输入：单段编码文本分词列表、高/中/低特征词库（已转为向量）
    输出：0~10量化分值
    """
    # 根据传入的词库类型，选择对应的预编码向量
    # 注意：这里为了保持接口兼容，high_words等参数仍然接收，但实际使用的是预编码向量
    # 通过 inspect 调用栈判断当前是哪个维度，或者直接使用全局预编码向量
    # 由于原代码调用方式为 calc_text_score(text, config.cost_high, config.cost_mid, config.cost_low)
    # 我们无法在函数内区分当前是成本/收益/损耗，因此使用调用者的帧信息来判断


    frame = inspect.currentframe()
    caller_frame = frame.f_back
    caller_locals = caller_frame.f_locals

    # 判断调用上下文：检查调用时传入的 high_words 是哪个 config 属性
    # 通过 id() 比较对象引用
    if id(high_words) == id(config.cost_high):
        return _semantic_score(text, _COST_HIGH_VEC, _COST_MID_VEC, _COST_LOW_VEC, threshold)
    elif id(high_words) == id(config.gain_high):
        return _semantic_score(text, _GAIN_HIGH_VEC, _GAIN_MID_VEC, _GAIN_LOW_VEC, threshold)
    elif id(high_words) == id(config.loss_high):
        return _semantic_score(text, _LOSS_HIGH_VEC, _LOSS_MID_VEC, _LOSS_LOW_VEC, threshold)
    else:
        # 兜底：直接使用传入词库（兼容旧逻辑）
        # 如果传入的是普通列表，临时编码
        high_vec = _encode_word_list(high_words) if not isinstance(high_words, np.ndarray) else high_words
        mid_vec = _encode_word_list(mid_words) if not isinstance(mid_words, np.ndarray) else mid_words
        low_vec = _encode_word_list(low_words) if not isinstance(low_words, np.ndarray) else low_words
        return _semantic_score(text, high_vec, mid_vec, low_vec, threshold)



@decorators.validate_and_catch(func_name="在hanlp上导入并清洗数据")
def filter_strategy():
    # 导入excel表格
    df = pd.read_excel(config.excel_name)
    print("excel数据读取完成：")
    Tool.write_sys_opt_log("成功导入数据")

    # 创建一个副本去除图像化影响
    date = df.copy()
    # 判断excel列名
    if "名称" in df.columns and "编码文本" in df.columns:
        name = "名称"
        text = "编码文本"
        Tool.write_sys_opt_log("成功识别nvivo的列名")
    else:
        raise ValueError("Excel 中缺少'名称'或'编码文本'列")

    date = date.dropna(subset=[name, text])
    date[name] = date[name].astype(str).str.strip()
    date[text] = date[text].astype(str).str.strip()

    # 只保留指定名称（不区分大小写）
    keep_lower = [name.lower() for name in config.date]
    date = date[date[name].str.lower().isin(keep_lower)]
    Tool.write_sys_opt_log("成功清洗无关编码名称")

    # 删除无效值
    date = date[~date[name].str.lower().isin(["nan", "none", "null", ""])]
    date = date[date[text].str.len() >= 5]
    date = date.drop_duplicates(subset=[name, text])
    Tool.write_sys_opt_log("删除了无效值和重复编码")

    date = date.reset_index(drop=True)
    print(f"保留名称: {date[name].unique().tolist()}")
    print(f"共 {len(date)} 条数据")
    Tool.write_sys_opt_log("清洗数据成功")

    return date, name, text


@decorators.validate_and_catch(func_name="正则清洗2")
def ti_cu2(texts):
    j = r"[^\u4e00-\u9fa5']"
    text = [re.sub(j, "", str(i)) for i in texts]
    texts = [b for b in text if b]
    return texts


@decorators.validate_and_catch(func_name="正则清洗")
def ti_cu(items):
    for text in items:
        ors_text = text["词列表"]
        clean = ti_cu2(ors_text)
        text["词列表"] = clean
    return items


@decorators.validate_and_catch(func_name="批量分词")
def segment_with_labels(df, name, text):
    model = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_SMALL)
    texts = df[text].tolist()
    results = model(texts, coarse=config.text)

    labeled_results = []
    for idx, doc in enumerate(results):
        labeled_results.append({
            "策略": df.iloc[idx][name],
            "词列表": doc,
        })
    return labeled_results


def count_with_labels(batch):
    if config.text_hanlp:
        batch = ti_cu(batch)
    batch_results = []
    for word in batch:
        # 使用语义相似度计算 C/R/L
        c = calc_text_score(word["词列表"], config.cost_high, config.cost_mid, config.cost_low)
        r = calc_text_score(word["词列表"], config.gain_high, config.gain_mid, config.gain_low)
        l = calc_text_score(word["词列表"], config.loss_high, config.loss_mid, config.loss_low)

        # 计算协同/非协同概率（此部分逻辑不变）
        neg_word_count = sum([1 for w in word["词列表"] if w in (config.loss_high | config.loss_low | config.loss_mid)])
        prob_neg = np.clip(neg_word_count / max(len(word["词列表"]) / 100, 1), 0, 1)
        prob_pos = 1 - prob_neg

        batch_results.append({
            "名称": word["策略"],
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


def main(name=None):
    if name:
        shutil.copy2(config.CK_CONFIG_PATH, config.CONFIG_PATH)

    with Pool(processes=config.CPU) as pool:
        results = pool.map(count_with_labels, chunkify(segment_with_labels(*filter_strategy()), config.CPU))

    flat_results = []
    for batch in results:
        flat_results.extend(batch)

    result_df = pd.DataFrame(flat_results)
    summary = result_df.groupby("名称").mean().round(config.round_data)
    print(summary)
    summary.to_excel(config.ff, index=True)
    Tool.write_sys_opt_log("成功导入excel")


if __name__ == "__main__":
    main()
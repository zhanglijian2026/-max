import pandas as pd
import numpy as np
import config
import hanlp
from multiprocessing import Pool
from log import Tool
import decorators
import re
import shutil

#导入数据并数据清洗功能
@decorators.validate_and_catch(func_name="在hanlp上导入并清洗数据")
def filter_strategy():

        # 导入excel表格
        df = pd.read_excel(config.excel_name)
        print("excel数据读取完成：")

        Tool.write_sys_opt_log("成功导入数据")

        #创建一个副本去除图像化影响
        date=df.copy()
        #判断excel列名
        if "名称" in df.columns and "编码文本" in df.columns:name = "名称";text = "编码文本";Tool.write_sys_opt_log("成功识别nvivo的列名")
        else:raise ValueError("Excel 中缺少'名称'或'编码文本'列")

        date= date.dropna(subset=[name, text])

        # 清洗文本
        date[name] = date[name].astype(str).str.strip()#
        date[text] = date[text].astype(str).str.strip()
        # 只保留指定名称（不区分大小写）
        keep_lower = [name.lower() for name in config.date]
        date = date[date[name].str.lower().isin(keep_lower)]
        Tool.write_sys_opt_log("成功清洗无关编码名称")
        # 删除无效值
        date = date[~date[name].str.lower().isin(["nan", "none", "null", ""])]
        date= date[date[text].str.len() >= 5]
        # 删除重复
        date = date.drop_duplicates(subset=[name, text])
        Tool.write_sys_opt_log("删除了无效值和重复编码")
        # 重置索引
        date = date.reset_index(drop=True)
        print(f"保留名称: {date[name].unique().tolist()}")
        print(f"共 {len(date)} 条数据")

        Tool.write_sys_opt_log("清洗数据成功")
        return date,name,text
@decorators.validate_and_catch(func_name="正则清洗2")
def ti_cu2(texts):
    j=r"[^\u4e00-\u9fa5']"
    text=[re.sub(j,"",str(i)) for i in texts]
    texts = [b for b in text if  b]
    return texts
@decorators.validate_and_catch(func_name="正则清洗")
def ti_cu(items):
    for text in items:
        ors_text=text["词列表"]
        clean=ti_cu2(ors_text)
        text["词列表"]=clean
    return items
#分词
@decorators.validate_and_catch(func_name="批量分词")
def segment_with_labels(df,name,text):
    # CTB9_TOK_ELECTRA_SMALL
    model = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_SMALL)

    texts = df[text].tolist()
    results = model(texts,coarse=config.text) # 批量分词
    #results = results[config.key]
    # 组装结果，保留策略名称
    labeled_results = []
    for idx, doc in enumerate(results):
        labeled_results.append({
            "策略": df.iloc[idx][name],  # 保留对应策略名称
            "词列表": doc, #
        })
    return labeled_results

#核心计算逻辑
def count_with_labels(batch):
    if config.text_hanlp:batch=ti_cu(batch)
    batch_results = []
    for word in batch:
        #计算
        c = calc_text_score(word["词列表"], config.cost_high, config.cost_mid, config.cost_low)
        r = calc_text_score(word["词列表"], config.gain_high, config.gain_mid, config.gain_low)
        l = calc_text_score(word["词列表"], config.loss_high, config.loss_mid, config.loss_low)

        # 计算初始群众行为概率（负面词越多，消极维权概率越高）
        neg_word_count = sum([1 for w in word["词列表"] if w in (config.loss_high | config.loss_low | config.loss_mid)])
        prob_neg = np.clip(neg_word_count / max(len(word["词列表"]), 1), 0, 1)
        prob_pos = 1 - prob_neg
            # 存入列表
        batch_results.append({
            "名称": word["策略"],
            "成本C": c,
            "收益R": r,
            "次生损耗L": l,
            "合规反馈概率": prob_pos,
            "消极越级维权概率": prob_neg
        })
    return batch_results

#打分和计算功能
@decorators.validate_and_catch(func_name="在hanlp上打分和计算功能")
def calc_text_score(text:list, high_words, mid_words, low_words)->float:
    """
    输入：单段编码文本、高/中/低特征词库
    输出：0~10量化分值
    """
    #对提炼分词结果
    high_cnt = mid_cnt = low_cnt = 0

    for w in text:
        if w in high_words:high_cnt += 1
        elif w in mid_words:mid_cnt += 1
        elif w in low_words:low_cnt += 1

    # 加权计算原始分数，映射到0~10的区间
    raw_score = high_cnt * config.C + mid_cnt * config.R + low_cnt * config.L
    max_possible = max(len(text) , 1)
    norm_score = np.clip(raw_score / max_possible *10, 0, 10)
    return round(norm_score, 2)

#分割列表
@decorators.validate_and_catch(func_name="分割列表")
def chunkify(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

#执行
def main(name=None):
    if name:
        shutil.copy2(config.CK_CONFIG_PATH,config.CONFIG_PATH )
    with Pool(processes=config.CPU,) as pool:
        results = pool.map(count_with_labels,chunkify(segment_with_labels(*filter_strategy()),config.CPU))
    flat_results = []
    for batch in results:flat_results.extend(batch)
    #转图
    result_df = pd.DataFrame(flat_results)
    # 按策略分组汇总
    summary = result_df.groupby("名称").mean().round(config.round_data)
    print(summary)
    # 导出excel
    summary.to_excel(config.ff, index=True)
    Tool.write_sys_opt_log("成功导入excel")

if __name__ == "__main__":
    main()
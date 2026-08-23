import hanlp

# 模型注册表：名称 → 预训练常量
MODEL_REGISTRY = {
    # ====== 多任务模型（精度高，速度慢） ======
    "mtl_small": hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH,
    "mtl_base": hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH,

    # ====== 单任务分词模型（精度高，速度快） ======
    "tok_small": hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH,  # 粗分，快
    "tok_base": hanlp.pretrained.tok.CTB9_TOK_ELECTRA_BASE,  # 高精度
    "tok_crf": hanlp.pretrained.tok.CTB9_TOK_ELECTRA_BASE_CRF,  # CRF 增强
}


def load_model(model_name):
    """根据名称加载模型"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}，可用选项: {list(MODEL_REGISTRY.keys())}")
    return hanlp.load(MODEL_REGISTRY[model_name])
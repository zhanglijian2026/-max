from log import Tool

#自制异常捕获与数值校验装饰器
def validate_and_catch(func_name=None, expected_type=None, min_val=None, max_val=None,
                       validate_vars=None):
    """
    异常捕获 + 数值校验装饰器
    分类捕获 20+ 种常见异常，并支持校验函数内部变量
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            local_vars = {}
            try:
                result = func(*args, **kwargs)
            # 文件/IO 相关异常
            except FileNotFoundError as e:
                name = func_name if func_name else func.__name__
                print(f"[文件错误] 函数 '{name}' 找不到文件")
                print(f"  路径: {e.filename if hasattr(e, 'filename') else '未知'}")
                print(f"  错误: {e}")
                Tool.write_err_log(f"{name}出现{e.filename}")
                return None
            except PermissionError as e:
                name = func_name if func_name else func.__name__
                print(f"[权限错误] 函数 '{name}' 没有访问权限")
                print(f"  路径: {e.filename if hasattr(e, 'filename') else '未知'}")
                print(f"  错误: {e}")
                Tool.write_err_log(f"{name}出现{e.filename}问题")
                return None
            # 网络/连接相关异常
            except ConnectionError as e:
                name = func_name if func_name else func.__name__
                print(f"[连接错误] 函数 '{name}' 网络连接失败")
                print(f"  错误: {e}")
                Tool.write_err_log(f"{name}出现连接错误问题，{e}")
                return None
            except TimeoutError as e:
                name = func_name if func_name else func.__name__
                print(f"[超时错误] 函数 '{name}' 操作超时")
                print(f"  错误: {e}")
                Tool.write_err_log(f"{name},出现出现连接超时错误，{e}")
                return None
            except IsADirectoryError as e:
                name = func_name if func_name else func.__name__
                print(f"[目录错误] 函数 '{name}' 期望文件但得到目录")
                print(f"  路径: {e.filename if hasattr(e, 'filename') else '未知'}")
                print(f"  错误: {e}")
                Tool.write_err_log(f"{name}出现目录问题，{e}")
                return None
            except NotADirectoryError as e:
                name = func_name if func_name else func.__name__
                print(f"[目录错误] 函数 '{name}' 期望目录但得到文件")
                print(f"  路径: {e.filename if hasattr(e, 'filename') else '未知'}")
                print(f"  错误: {e}")
                Tool.write_err_log(f"目录错误] 函数 '{name}' 期望目录但得到文件:{e}")
                return None
            except OSError as e:
                name = func_name if func_name else func.__name__
                print(f"[系统错误] 函数 '{name}' 发生操作系统错误")
                print(f"  错误码: {e.errno if hasattr(e, 'errno') else '未知'}")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[系统错误] 函数 '{name}' 发生操作系统错误:{e}")
                return None
            # 数值/类型相关异常
            except ValueError as e:
                name = func_name if func_name else func.__name__
                print(f"[值错误] 函数 '{name}' 参数值不合法")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[值错误] 函数 '{name}' 参数值不合法")
                return None
            except TypeError as e:
                name = func_name if func_name else func.__name__
                print(f"[类型错误] 函数 '{name}' 参数类型不匹配")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[类型错误] 函数 '{name}' 参数类型不匹配")
                return None
            except ZeroDivisionError as e:
                name = func_name if func_name else func.__name__
                print(f"[除零错误] 函数 '{name}' 除以了零")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[除零错误] 函数 '{name}' 除以了零")
                return None
            except OverflowError as e:
                name = func_name if func_name else func.__name__
                print(f"[溢出错误] 函数 '{name}' 数值溢出")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[溢出错误] 函数 '{name}' 数值溢出")
                return None
            except ArithmeticError as e:
                name = func_name if func_name else func.__name__
                print(f"[算术错误] 函数 '{name}' 发生算术运算错误")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[算术错误] 函数 '{name}' 发生算术运算错误")
                return None
            #容器/序列相关异常
            except KeyError as e:
                name = func_name if func_name else func.__name__
                print(f"[键错误] 函数 '{name}' 访问了不存在的键")
                print(f"  缺失键: {e}")
                Tool.write_err_log(f"[键错误] 函数 '{name}' 访问了不存在的键")
                return None
            except IndexError as e:
                name = func_name if func_name else func.__name__
                print(f"[索引错误] 函数 '{name}' 访问了不存在的索引")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[索引错误] 函数 '{name}' 访问了不存在的索引")
                return None
            except StopIteration as e:
                name = func_name if func_name else func.__name__
                print(f"[迭代错误] 函数 '{name}' 迭代器已结束")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[迭代错误] 函数 '{name}' 迭代器已结束")
                return None
            except AttributeError as e:
                name = func_name if func_name else func.__name__
                print(f"[属性错误] 函数 '{name}' 访问了不存在的属性")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[属性错误] 函数 '{name}' 访问了不存在的属性")
                return None
            #导入/模块相关异常
            except ModuleNotFoundError as e:
                name = func_name if func_name else func.__name__
                print(f"[模块错误] 函数 '{name}' 找不到模块")
                print(f"  模块: {e.name if hasattr(e, 'name') else '未知'}")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[模块错误] 函数 '{name}' 找不到模块")
                return None
            except ImportError as e:
                name = func_name if func_name else func.__name__
                print(f"[导入错误] 函数 '{name}' 缺少依赖模块")
                print(f"  模块: {e.name if hasattr(e, 'name') else '未知'}")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[导入错误] 函数 '{name}' 缺少依赖模块")
                return None
            #编码/解码相关异常
            except UnicodeDecodeError as e:
                name = func_name if func_name else func.__name__
                print(f"[解码错误] 函数 '{name}' Unicode 解码失败")
                print(f"  位置: {e.start}-{e.end} 字节")
                print(f"  编码: {e.encoding}")
                print(f"  原因: {e.reason}")
                Tool.write_err_log(f"[解码错误] 函数 '{name}' Unicode 解码失败")
                return None
            except UnicodeEncodeError as e:
                name = func_name if func_name else func.__name__
                print(f"[编码错误] 函数 '{name}' Unicode 编码失败")
                print(f"  位置: {e.start}-{e.end} 字符")
                print(f"  编码: {e.encoding}")
                print(f"  原因: {e.reason}")
                Tool.write_err_log(f"[编码错误] 函数 '{name}' Unicode 编码失败")
                return None
            #内存/资源相关异常
            except MemoryError as e:
                name = func_name if func_name else func.__name__
                print(f"[内存错误] 函数 '{name}' 内存不足")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[内存错误] 函数 '{name}' 内存不足")
                return None
            except RecursionError as e:
                name = func_name if func_name else func.__name__
                print(f"[递归错误] 函数 '{name}' 递归超过最大深度")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[递归错误] 函数 '{name}' 递归超过最大深度")
                return None
            # 其他/兜底异常
            except KeyboardInterrupt:
                name = func_name if func_name else func.__name__
                print(f"[用户中断] 函数 '{name}' 被用户手动中断")
                Tool.write_err_log(f"[用户中断] 函数 '{name}' 被用户手动中断")
                return None
            except Exception as e:
                name = func_name if func_name else func.__name__
                print(f"[未知错误] 函数 '{name}' 发生未分类异常")
                print(f"  异常类型: {type(e).__name__}")
                print(f"  错误: {e}")
                Tool.write_err_log(f"[未知错误] 函数 '{name}' 发生未分类异常,{type(e).__name__}")
                return None
            #  校验
            name = func_name if func_name else func.__name__
            # 类型校验
            if expected_type is not None:
                if not isinstance(result, expected_type):print(f"[校验失败] 函数 '{name}' 返回值类型错误");print(f"  期望: {expected_type.__name__}, 实际: {type(result).__name__};");Tool.write_err_log(f"{name}数值类型错误");return None
            # 数值范围校验
            if isinstance(result, (int, float)):
                if min_val is not None and result < min_val:print(f"[校验失败] 函数 '{name}' 返回值小于最小值");print(f"  期望 >= {min_val}, 实际: {result}");Tool.write_err_log(f"数值小于最小值{min_val}，实际: {result}");return None
                if max_val is not None and result > max_val:print(f"[校验失败] 函数 '{name}' 返回值大于最大值");print(f"  期望 <= {max_val}, 实际: {result}");Tool.write_err_log(f"数值大于最小值{min_val}，实际: {result}");return None
            # 内部变量校验
            if validate_vars:
                for var_name, (var_type, var_min, var_max) in validate_vars.items():
                    var_value = local_vars.get(var_name)
                    if var_value is None:print(f"[警告] 变量 '{var_name}' 未在校验前赋值，跳过校验");Tool.write_err_log(f"[警告] 变量 '{var_name}' 未在校验前赋值，跳过校验");continue
                    if not isinstance(var_value, var_type):print(f"[校验失败] 变量 '{var_name}' 类型错误");print(f"  期望: {var_type.__name__}, 实际: {type(var_value).__name__}");Tool.write_err_log(f"[校验失败] 变量 '{var_name}' 类型错误");print(f"  期望: {var_type.__name__}, 实际: {type(var_value).__name__}");return None
                    if isinstance(var_value, (int, float)):
                        if var_min is not None and var_value < var_min:print(f"[校验失败] 变量 '{var_name}' 小于最小值");print(f"  期望 >= {var_min}, 实际: {var_value}");Tool.write_err_log(f"[校验失败] 变量 '{var_name}' 小于最小值");print(f"  期望 >= {var_min}, 实际: {var_value}");return None
                        if var_max is not None and var_value > var_max:print(f"[校验失败] 变量 '{var_name}' 大于最大值");print(f"  期望 <= {var_max}, 实际: {var_value}");Tool.write_err_log(f"[校验失败] 变量 '{var_name}' 大于最大值");print(f"  期望 <= {var_max}, 实际: {var_value}");return None
            return result
        return wrapper
    return decorator
def set_validate_var(local_vars, name, value):
    """将需要校验的变量存入 local_vars 字典"""
    local_vars[name] = value
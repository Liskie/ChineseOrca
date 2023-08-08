import os
import re


def dir_check(directory: str) -> None:
    # Check if the output directory exists
    if not os.path.exists(os.path.dirname(directory)):
        os.makedirs(os.path.dirname(directory))


def retry_condition(response) -> bool:
    if response.startswith('<error> <429>') or response.startswith('<error> <server_error>'):
        return True


def contains_only_zh_en_num_punct(string) -> bool:
    # 定义匹配中文、英文、数字和标点符号的正则表达式
    pattern = re.compile(r'^[\u4e00-\u9fa5a-zA-Z0-9\s，。、％“”‘’/ⅡⅢ！？《》°（）&@#$%^*￥…—－\-_+=·,.:：；「」{}\[\]\\|;\'"!?<>()-]*$')
    # 使用正则表达式匹配字符串
    match = pattern.match(string)
    # 如果匹配成功，则字符串只包含中文、英文、数字和标点符号
    # 否则，字符串包含其他字符
    return match is not None

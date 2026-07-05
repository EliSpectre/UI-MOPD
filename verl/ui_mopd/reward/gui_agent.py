# Copyright (c) Xiaoai Platforms, Inc. and affiliates.
# All rights reserved.

import json
import re
import traceback


def parse_tool_call(text: str) -> dict | None:
    """从文本中提取 <tool_call>...</tool_call> 并解析为展平的 dict。

    返回格式: {"name": "mobile_use"/"computer_use", "action": "click", "coordinate": [...], ...}
    解析失败返回 None。
    """
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not match:
        return None
    try:
        tool_call = json.loads(match.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return None

    name = tool_call.get("name", "")
    arguments = tool_call.get("arguments", {})
    if not isinstance(arguments, dict) or "action" not in arguments:
        return None

    result = {"name": name}
    result.update(arguments)
    return result


def position_in_bbox(coordinate, bbox) -> bool:
    """判断 coordinate [x, y] 是否在 bbox [[x1,y1],[x2,y2]] 内。"""
    if not coordinate or not bbox:
        return False
    if len(coordinate) != 2 or len(bbox) != 2:
        return False

    x, y = coordinate[0], coordinate[1]
    (x1, y1), (x2, y2) = bbox[0], bbox[1]

    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)

    return min_x <= x <= max_x and min_y <= y <= max_y


# ==================== Mobile Reward Functions (mobile_use) ====================


def reward_click(predict, label, extra_info) -> float:
    """click: action 匹配 + coordinate 在 bbox 内 → 2 维"""
    score = 0
    if predict.get("action") == label.get("action"):
        score += 1

    bbox = extra_info.get("bbox")
    if bbox and "coordinate" in predict:
        if position_in_bbox(predict["coordinate"], bbox):
            score += 1
    elif "coordinate" in predict and "coordinate" in label:
        if predict["coordinate"] == label["coordinate"]:
            score += 1

    return score / 2.0


def reward_long_press(predict, label, extra_info) -> float:
    """long_press: action 匹配 + coordinate 在 bbox 内 → 2 维"""
    score = 0
    if predict.get("action") == label.get("action"):
        score += 1

    bbox = extra_info.get("bbox")
    if bbox and "coordinate" in predict:
        if position_in_bbox(predict["coordinate"], bbox):
            score += 1
    elif "coordinate" in predict and "coordinate" in label:
        if predict["coordinate"] == label["coordinate"]:
            score += 1

    return score / 2.0


def reward_swipe(predict, label, extra_info) -> float:
    """swipe: action 匹配 + start 在 bbox 内 + end 在 bbox2 内 → 3 维"""
    score = 0
    if predict.get("action") == label.get("action"):
        score += 1

    bbox = extra_info.get("bbox")
    if bbox and "coordinate" in predict:
        if position_in_bbox(predict["coordinate"], bbox):
            score += 1
    elif "coordinate" in predict and "coordinate" in label:
        if predict["coordinate"] == label["coordinate"]:
            score += 1

    bbox2 = extra_info.get("bbox2")
    if bbox2 and "coordinate2" in predict:
        if position_in_bbox(predict["coordinate2"], bbox2):
            score += 1
    elif "coordinate2" in predict and "coordinate2" in label:
        if predict["coordinate2"] == label["coordinate2"]:
            score += 1

    return score / 3.0


def reward_type_mobile(predict, label, extra_info) -> float:
    """type/answer: action 匹配 + text 大小写不敏感匹配 → 2 维"""
    score = 0
    if predict.get("action") == label.get("action"):
        score += 1

    pred_text = str(predict.get("text", "")).strip().lower()
    label_text = str(label.get("text", "")).strip().lower()
    if pred_text == label_text:
        score += 1

    return score / 2.0


def reward_system_button(predict, label, extra_info) -> float:
    """system_button: action 匹配 + button 匹配 → 2 维"""
    score = 0
    if predict.get("action") == label.get("action"):
        score += 1

    if predict.get("button") == label.get("button"):
        score += 1

    return score / 2.0


def reward_wait(predict, label, extra_info) -> float:
    """wait: 仅 action 匹配 → 1 维"""
    if predict.get("action") == label.get("action"):
        return 1.0
    return 0.0


def reward_terminate(predict, label, extra_info) -> float:
    """terminate: action 匹配 + status 匹配 → 2 维"""
    score = 0
    if predict.get("action") == label.get("action"):
        score += 1

    if predict.get("status") == label.get("status"):
        score += 1

    return score / 2.0


# ==================== Desktop Reward Functions (computer_use) ====================


def reward_mouse_click(predict, label, extra_info) -> float:
    """鼠标点击类: action 匹配 + coordinate 在 bbox 内 → 2 维
    覆盖: left_click, right_click, middle_click, double_click, triple_click, mouse_move
    """
    score = 0
    if predict.get("action") == label.get("action"):
        score += 1

    bbox = extra_info.get("bbox")
    if bbox and "coordinate" in predict:
        if position_in_bbox(predict["coordinate"], bbox):
            score += 1
    elif "coordinate" in predict and "coordinate" in label:
        if predict["coordinate"] == label["coordinate"]:
            score += 1

    return score / 2.0


def reward_click_drag(predict, label, extra_info) -> float:
    """left_click_drag: action 匹配 + 终点 coordinate 在 bbox 内 → 2 维"""
    score = 0
    if predict.get("action") == label.get("action"):
        score += 1

    bbox = extra_info.get("bbox")
    if bbox and "coordinate" in predict:
        if position_in_bbox(predict["coordinate"], bbox):
            score += 1
    elif "coordinate" in predict and "coordinate" in label:
        if predict["coordinate"] == label["coordinate"]:
            score += 1

    return score / 2.0


def reward_type_desktop(predict, label, extra_info) -> float:
    """type: action 匹配 + text 大小写不敏感 → 2 维"""
    score = 0
    if predict.get("action") == label.get("action"):
        score += 1

    pred_text = str(predict.get("text", "")).strip().lower()
    label_text = str(label.get("text", "")).strip().lower()
    if pred_text == label_text:
        score += 1

    return score / 2.0


def reward_key(predict, label, extra_info) -> float:
    """key: action 匹配 + keys 集合相等 → 2 维"""
    score = 0
    if predict.get("action") == label.get("action"):
        score += 1

    pred_keys = predict.get("keys", [])
    label_keys = label.get("keys", [])
    if isinstance(pred_keys, list) and isinstance(label_keys, list):
        if set(k.lower() for k in pred_keys) == set(k.lower() for k in label_keys):
            score += 1

    return score / 2.0


def reward_scroll(predict, label, extra_info) -> float:
    """scroll: action 匹配 + coordinate 在 bbox 内 + 滚动方向匹配 → 3 维"""
    score = 0
    if predict.get("action") == label.get("action"):
        score += 1

    bbox = extra_info.get("bbox")
    if bbox and "coordinate" in predict:
        if position_in_bbox(predict["coordinate"], bbox):
            score += 1
    elif "coordinate" in predict and "coordinate" in label:
        if predict["coordinate"] == label["coordinate"]:
            score += 1

    pred_pixels = predict.get("pixels", 0)
    label_pixels = label.get("pixels", 0)
    try:
        pred_sign = (pred_pixels > 0) - (pred_pixels < 0)
        label_sign = (label_pixels > 0) - (label_pixels < 0)
        if pred_sign == label_sign:
            score += 1
    except (TypeError, ValueError):
        pass

    return score / 3.0


# ==================== Dispatch Tables ====================


MOBILE_REWARD_FUNCS = {
    "click": reward_click,
    "long_press": reward_long_press,
    "swipe": reward_swipe,
    "type": reward_type_mobile,
    "answer": reward_type_mobile,
    "system_button": reward_system_button,
    "wait": reward_wait,
    "terminate": reward_terminate,
}

DESKTOP_REWARD_FUNCS = {
    "left_click": reward_mouse_click,
    "right_click": reward_mouse_click,
    "middle_click": reward_mouse_click,
    "double_click": reward_mouse_click,
    "triple_click": reward_mouse_click,
    "mouse_move": reward_mouse_click,
    "left_click_drag": reward_click_drag,
    "type": reward_type_desktop,
    "key": reward_key,
    "scroll": reward_scroll,
    "hscroll": reward_scroll,
    "wait": reward_wait,
    "terminate": reward_terminate,
}


# ==================== Main Entry Point ====================


def _compute_single(predict_str: str, ground_truth: str, extra_info: dict) -> float:
    """对单条样本计算 reward（不含 Plan B 逻辑）。
    返回: 1.0 / -0.5 / -1.0
    """
    label = parse_tool_call(ground_truth)
    if label is None:
        print(f"[reward] Failed to parse ground_truth: {ground_truth[:200]}")
        return -1.0

    predict = parse_tool_call(predict_str)
    if predict is None:
        return -1.0

    name = label.get("name", "")
    action = label.get("action", "")

    if name == "mobile_use":
        reward_fn = MOBILE_REWARD_FUNCS.get(action)
    elif name == "computer_use":
        reward_fn = DESKTOP_REWARD_FUNCS.get(action)
    else:
        print(f"[reward] Unknown tool name: {name}")
        return -1.0

    if reward_fn is None:
        print(f"[reward] Unknown action '{action}' for tool '{name}'")
        return -1.0

    raw_reward = reward_fn(predict, label, extra_info)
    return 1.0 if raw_reward == 1.0 else -0.5


async def compute_score(predict_str: str, ground_truth: str, extra_info: dict, data_source: str = "") -> dict:
    """主入口函数。兼容 DAPORewardManager 的调用约定。

    Args:
        predict_str: 模型生成的 response 文本（已解码）
        ground_truth: 来自 reward_model["ground_truth"] 的标签
        extra_info: 包含 bbox, bbox2, pixel, mode 等信息
        data_source: 数据源标识

    Returns:
        dict with keys: score, acc, thought_length
    """
    think_len = 0
    try:
        think_match = re.search(r"<think(?:ing)?>(.*?)</think(?:ing)?>", predict_str, re.DOTALL)
        if not think_match:
            think_match = re.search(r"^(.*?)</think(?:ing)?>", predict_str, re.DOTALL)
        if think_match:
            tokenizer = extra_info.get("tokenizer")
            think_text = think_match.group(1).strip()
            if tokenizer:
                think_len = len(tokenizer.encode(think_text))
            else:
                think_len = len(think_text)
    except Exception:
        pass

    try:
        reward = _compute_single(predict_str, ground_truth, extra_info)
    except Exception as e:
        print(f"[reward] Exception in compute_score: {traceback.format_exc()}")
        reward = -1.0

    # Plan B: 如果 Plan A 不满分，尝试备选答案
    if reward != 1.0:
        other_label = extra_info.get("other_label")
        if other_label:
            try:
                other_extra_info = dict(extra_info)
                if "other_bbox" in extra_info:
                    other_extra_info["bbox"] = extra_info["other_bbox"]
                if "other_bbox2" in extra_info:
                    other_extra_info["bbox2"] = extra_info["other_bbox2"]
                plan_b_reward = _compute_single(predict_str, other_label, other_extra_info)
                if plan_b_reward == 1.0:
                    reward = 1.0
            except Exception:
                pass

    return {
        "score": reward,
        "acc": 1.0 if reward == 1.0 else 0.0,
        "thought_length": think_len,
    }

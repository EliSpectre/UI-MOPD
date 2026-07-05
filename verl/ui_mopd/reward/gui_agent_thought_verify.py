import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

import torch
from openai import OpenAI

THOUGHT_MATCH_PROMPT = """# Role
你是一位极其严谨的 GUI Agent 动作一致性评估专家。你的核心能力是精准识破文本描述与实际函数调用之间的逻辑矛盾。

# Task
阅读模型生成的 `thought`（思考过程）、`action`（动作简述）和 `func`（实际调用的函数），判断这三者在**最终意图、具体操作对象与函数类型**上是否**完全一致**。

# Evaluation Criteria (严格判定标准)

**第一关：Thought 与 Action 的意图一致性（逻辑与语义核对）**
1. **提取最终决策**：忽略 `thought` 中冗长的推理、幻觉或页面描述，**只看它最终得出的决定**。这个决定必须与 `action` 的意图完全吻合。
2. **严抓逻辑矛盾（致命错误）**：
   - 若 `thought` 认为“需求已满足/已找到”，但 `action` 是“点击返回/继续操作” -> **不一致**。
   - 若 `thought` 认为“需要重新搜索/继续操作”，但 `action` 是“任务完成/失败” -> **不一致**。
   - 若 `thought` 仅仅客观描述了当前页面（例如“当前是搜索页面，有搜索框”），**并没有明确表明下一步要做什么**，但 `action` 却凭空给出了动作（如“点击搜索”） -> **不一致**。
3. **允许合理的 UI 语义泛化与省略（宽容条件）**：
   - 允许 UI 元素的合理同义替换（如 `thought` 说“耳机图标/菜单”，`action` 说“频道/更多功能”），视为**一致**。
   - 允许 `action` 对 `thought` 的合理简写（如 `thought` 说“点击顶部右侧的少儿按钮”，`action` 说“点击少儿”），视为**一致**。

**第二关：Action 与 Func 的功能匹配度（边界核对）**
1. **UI操作 vs 系统操作**：
   - 如果 `action` 描述的是点击页面上的某个具体元素（如“点击返回箭头”、“点击某个按钮”），`func` 必须是 UI 点击类（如 `Tap`）。
   - 如果 `action` 描述的是调用系统级指令（如“按系统返回键”、“返回上一页面”），`func` 才对应系统级返回（如 `Back`）。发生错位 -> **不一致**。
2. **特殊状态匹配**：
   - 如果 `action` 表达的是“需要用户手动输入密码/身份验证/任务无法完成”，对应的 `func` 必须是状态反馈类（如 `Fail` 且带有对应类型），视为**一致**。

# Input Data
<thought>
{thought}
</thought>
<action>
{action}
</action>
<func>
{func}
</func>

# Output Format
请仔细进行上述两关的对比，直接输出最终判定结果。
<result>是/否</result>"""


def extract_label(response_str):
    """Parse v4 format: <think>..</think><action>..</action><tool_call>{"name":..,}</tool_call>"""
    try:
        thought = response_str.split("<think>")[1].split("</think>")[0]
    except IndexError:
        thought = ""

    action_match = re.search(r"<action>(.*?)</action>", response_str, re.DOTALL)
    action = action_match.group(1).strip() if action_match else ""

    func = ""
    tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", response_str, re.DOTALL)
    if tool_call_match:
        try:
            tool_call = json.loads(tool_call_match.group(1).strip())
            func = tool_call.get("name", "")
        except Exception:
            pass

    return thought, action, func


# 定义单个请求的处理函数
def process_single_request(client, request_data, temperature=0.01, top_p=0.9, top_k=1):
    index, uid, prompt, thought_part, action, func = request_data
    if not thought_part or not action or not func:  # 不符合格式要求，认为不匹配
        return (index, uid, False)
    response = ""
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="base_model",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=temperature,
                top_p=top_p,
                extra_body={
                    "top_k": top_k, 
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response = response.choices[0].message.content
            if response != "":
                break
        except:
            continue
    
    if not response:
        print("vllm服务请求失败！！！请检查vllm服务状态")
        return (index, uid, True)  # vllm请求失败时默认通过
    elif "<result>否</result>" in response:
        print(f"thought不匹配, Thought: {thought_part}, Action: {action}, Func: {func}")
        return (index, uid, False)
    else:
        return (index, uid, True)


def check_thought_action_consistency(batch, base_url):
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    # 检查client是否正常
    try:
        response = client.models.list()
        models = [model.id for model in response.data]
        assert "base_model" in models
    except:
        print("vllm服务不可用！！！不使用模型判断thought一致性")
        return batch

    thoughts_to_check = []
    for i in range(len(batch)):
        if batch.non_tensor_batch['score'][i] == 1.0:
            thoughts_to_check.append((i, batch[i].non_tensor_batch['response_str'], batch[i].non_tensor_batch['uid']))

    if thoughts_to_check:
        print(f"开始判断thought一致性，共{len(thoughts_to_check)}条数据")
        start_time = time()
        match_results = []
        batch_requests = []
        
        # 准备批量请求
        for index, response_str, uid in thoughts_to_check:
            thought, action, func = extract_label(response_str)
            PROMPT = THOUGHT_MATCH_PROMPT.format(thought=thought, action=action, func=func)
            batch_requests.append((index, uid, PROMPT, thought, action, func))
        
        # 使用线程池并发执行所有请求
        with ThreadPoolExecutor(max_workers=min(len(batch_requests), 256)) as executor:
            # 提交所有任务
            future_to_request = {executor.submit(process_single_request, client, request): request for request in batch_requests}
            
            # 收集所有结果
            for future in as_completed(future_to_request):
                try:
                    result = future.result()
                    match_results.append(result)
                except Exception as e:
                    print(f"处理请求时发生错误: {e}")
                    # 获取对应的请求数据以确定index
                    request = future_to_request[future]
                    index = request[0]
                    uid = request[1]
                    match_results.append((index, uid, True))  # 错误时默认通过
        
        end_time = time()
        print(f"判断thought一致性完成，共{len(match_results)}条数据，耗时{end_time - start_time}秒")

        # 后处理：若一个uid的所有数据均不一致，则不对该uid的数据进行惩罚（避免group全为负例）
        uid2results = defaultdict(list)
        for index, uid, is_match in match_results:
            uid2results[uid].append(is_match)
        uid_exempt_list = []
        for uid, results_list in uid2results.items():
            if all(not is_match for is_match in results_list):
                uid_exempt_list.append(uid)
        for i in range(len(match_results)):
            index = match_results[i][0]
            uid = match_results[i][1]
            if uid in uid_exempt_list:
                match_results[i] = (index, uid, True)

        # 对thought不一致的数据进行惩罚
        for index, uid, is_match in match_results:
            if not is_match:
                batch.non_tensor_batch['score'][index] = -0.5
                batch.non_tensor_batch['acc'][index] = 0.0
                nonzero_indices = torch.nonzero(batch[index].batch['token_level_scores'], as_tuple=True)
                batch[index].batch['token_level_scores'][nonzero_indices] = -0.5
                # batch[index].batch['token_level_rewards'][nonzero_indices] = -0.5

    return batch

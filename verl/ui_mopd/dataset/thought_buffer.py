"""
按照uid存储采样结果，存储的内容包括response_thoughts, func
"""
from collections import defaultdict
from dataclasses import dataclass
import json
from hashlib import md5

@dataclass
class ThoughtBufferItem:
    response_thoughts: list  # response_thoughts是个长度为thought_buffer_size的队列
    response_thoughts_set: set  # 用于快速判断response_thought是否已存在
    response_actions: list  # response_thoughts是个长度为thought_buffer_size的队列
    response_actions_set: set  # 用于快速判断response_thought是否已存在
    func: str



class ThoughtBuffer:
    def __init__(self, thought_buffer_size: int=1):
        self.buffer = {}
        self.thought_buffer_size = thought_buffer_size

    def add(self, uid, response_thought, response_action, func):
        if uid not in self.buffer:
            self.buffer[uid] = ThoughtBufferItem(
                [response_thought], 
                {response_thought},  # 初始化set
                [response_action],
                {response_action},
                func
            )
        else:
            buffer_item = self.buffer[uid]
            # 快速判断是否已存在
            if response_thought in buffer_item.response_thoughts_set and response_action in buffer_item.response_actions_set:
                return # 已存在，直接返回
            
            if response_thought not in buffer_item.response_thoughts_set:
            # 如果队列未满，直接添加
                if len(buffer_item.response_thoughts) < self.thought_buffer_size:
                    buffer_item.response_thoughts.append(response_thought)
                    buffer_item.response_thoughts_set.add(response_thought)
                else:
                    # 队列已满，移除最旧的元素
                    old_thought = buffer_item.response_thoughts.pop(0)
                    buffer_item.response_thoughts_set.discard(old_thought)
                    # 添加新元素
                    buffer_item.response_thoughts.append(response_thought)
                    buffer_item.response_thoughts_set.add(response_thought)
            
            if response_action not in buffer_item.response_actions_set:
                if len(buffer_item.response_actions) < self.thought_buffer_size:
                    buffer_item.response_actions.append(response_action)
                    buffer_item.response_actions_set.add(response_action)
                else:
                    # 队列已满，移除最旧的元素
                    old_action = buffer_item.response_actions.pop(0)
                    buffer_item.response_actions_set.discard(old_action)
                    # 添加新元素
                    buffer_item.response_actions.append(response_action)
                    buffer_item.response_actions_set.add(response_action)
    

    def add_batch(self, batch):
        # 记录每个数据输出的thoughts, 需要保证记录的thought都是唯一的，一个batch内，同一个uid只增加thought_buffer_size次
        uid2count = {}
        for batch_item in batch:
            extra_info = batch_item.non_tensor_batch.get("extra_info", "{}")
            extra_info = json.loads(extra_info)
            uid = batch_item.non_tensor_batch.get("uid", None)
            score = batch_item.non_tensor_batch.get("score", None)
            if score != 1.0: continue
            response_str = batch_item.non_tensor_batch.get("response_str", "{}")
            think = response_str.split("<think>")[1].split("</think>")[0]
            try:
                predict = json.loads(response_str.split("<answer>")[1].split("</answer>")[0])
                # response_js = json.loads(response_str)
                func = predict.get("func", None)
                action = predict.get("action", None)

                if score != 1.0 \
                        or not uid \
                        or not think \
                        or not func \
                        or not action \
                        or uid2count.get(uid, 0) >= self.thought_buffer_size:
                    continue

                if not self.contains(uid, think, action):
                    self.add(uid, think, action, func)
                    if uid not in uid2count:
                        uid2count[uid] = 0
                    uid2count[uid] += 1
            except Exception as e:
                print(f"dynamic_history add_batch error: {e}, response_str: {response_str}")
                continue
        del uid2count
        print(f"add_batch: {len(batch)} thoughts added to buffer, buffer size: {len(self.buffer)}")

    def get(self, uid) -> ThoughtBufferItem:
        return self.buffer.get(uid, None)
    
    def contains(self, uid: str, thought: str, action: str) -> bool:
        return uid in self.buffer and thought in self.buffer[uid].response_thoughts_set and action in self.buffer[uid].response_actions_set
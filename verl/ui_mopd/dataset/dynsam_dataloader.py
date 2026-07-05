import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict
import traceback


# TODO: support multiprocessing (num_workers > 0)
class DynamicSamplingDataloader:

    def __init__(
        self,
        dataset,
        total_step=5000,
        collate_fn=None,
        passrate1_strategy='replay',
        shuffle=True,
        replay_decay_factor=0.2,
        replay_min_ratio=0.04,
        reset_queue=False,
    ):
        # 参数
        self.dataset = dataset
        self.total_step = total_step
        self.collate_fn = collate_fn
        # options: replay, filter, none
        self.passrate1_strategy = passrate1_strategy
        assert self.passrate1_strategy in ('replay', 'filter', 'none')
        self.shuffle = shuffle
        self.replay_decay_factor = replay_decay_factor
        self.replay_min_ratio = replay_min_ratio
        self.reset_queue = reset_queue
        # 测试
        self.verbose = True

        # 状态变量
        self.index2passrate = defaultdict(list)  # 记录所有数据的采样passrate历史
        self.queue = list()  # 采样队列
        self.step = 0

        # 采样变量
        self.passrate1_set = defaultdict(int)  # 最近一次采样全对的数据集合
        self.passrate0_set = defaultdict(int)  # 最近一次采样全错的数据集合
        self.index2replay_ratio = [1.0] * len(self.dataset) # 记录每条数据的replay_ratio，初始时为1.0
        self.sample_step = 0

        # state to be managed
        self.states = [
            'index2passrate',
            'queue',
            'step',
            'passrate1_set',
            'passrate0_set',
            'index2replay_ratio',
            'sample_step',
            'dataset.thought_buffer.buffer',
        ]

    def __len__(self):
        return self.total_step

    def add_step(self):
        # step前调用，计数器加一
        self.step += 1
        self._verbose_print(f"Step={self.step}")

    def is_end(self):
        # while loop结束条件
        return self.step >= self.total_step

    def _verbose_print(self, *args, **kwargs):
        if self.verbose:
            print("[DynamicSamplingDataloader]:" ,*args, **kwargs)
    
    def _build_sample_filter(self):
        # passrate1_strategy = "filter"时使用，将最近2~3次采样中通过率过高的数据过滤掉
        # 注意：这是一种比较激进的课程学习策略，一旦一条数据被过滤掉，之后的rl训练中永远都不会采样该数据
        filter_set = set()
        for index, passrate_list in self.index2passrate.items():
            if len(passrate_list) >= 2:
                if len(passrate_list) >= 3:
                    passrate_list = passrate_list[-3:]
                avg_passrate = sum(passrate_list) / len(passrate_list)
                if avg_passrate > 46.9999/48:
                    filter_set.add(index)

        return filter_set

    def sample(self, batch_size: int):
        # 采样batch_size个数据，返回取样的数据
        
        assert isinstance(batch_size, int), "batch_size must be an integer"
        assert batch_size > 0, "batch_size must be greater than 0"

        self._verbose_print(f"Start sample {batch_size} data")
        self._verbose_print(f"len(dataset)={len(self.dataset)}, len(passrate1_set)={len(self.passrate1_set)}, len(passrate0_set)={len(self.passrate0_set)}, len(queue)={len(self.queue)}")

        data_index = list()  # 采样数据下标列表

        if self.passrate1_strategy in ('filter', 'none'):
            filter_set = self._build_sample_filter() if self.passrate1_strategy == 'filter' else set() 
            self._verbose_print(f"Compute filter index set, size={len(filter_set)}")

            while len(data_index) < batch_size:

                if len(self.queue) == 0:
                    self._verbose_print(f"New Queue")
                    new_queue = list(range(len(self.dataset)))
                    if self.shuffle:
                        random.shuffle(new_queue)
                    self.queue = new_queue

                index = self.queue.pop()
                if index not in filter_set:
                    data_index.append(index)

        elif self.passrate1_strategy == 'replay':
            sample_size = batch_size
            data_index = []
            passrate1_sample_size, other_sample_size = 0, 0
            while len(data_index) < sample_size:
                # 确保队列不为空，避免pop(0)时队列为空
                if len(self.queue) == 0:
                    self._verbose_print(f"New Queue")
                    new_queue = list(range(len(self.dataset)))
                    if self.shuffle:
                        random.shuffle(new_queue)
                    self.queue.extend(new_queue)
                
                index = self.queue.pop(0)
                if index in self.passrate1_set:
                    if random.random() < self.index2replay_ratio[index]:
                        data_index.append(index)
                        passrate1_sample_size += 1
                else:
                    data_index.append(index)
                    other_sample_size += 1
            self._verbose_print(f"passrate1_sample_size={passrate1_sample_size}, other_sample_size={other_sample_size}")

        # 使用并发执行来获取数据集中的数据
        with ThreadPoolExecutor(max_workers=min(len(data_index), 32)) as executor:
            batch_data = list(executor.map(lambda i: self.dataset[i], data_index))
        collated_batch_data = self.collate_fn(batch_data)

        return collated_batch_data

    def feedback(self, passrate_list):
        # 接受passrate反馈
        for index, passrate in passrate_list:
            if index is not None and passrate is not None:
                self.index2passrate[index].append(passrate)
                if passrate == 1:
                    self.passrate1_set[index] += 1
                    # 如果全做对，采样概率降低
                    try:
                        self.index2replay_ratio[index] = max(self.replay_min_ratio, self.index2replay_ratio[index] * self.replay_decay_factor)
                    except:
                        print(f"index={index}, passrate={passrate}, len(index2replay_ratio)={len(self.index2replay_ratio)}, len(dataset)={len(self.dataset)}")
                        print(traceback.format_exc())
                        raise Exception("index2replay_ratio error")

                else:
                    self.passrate1_set.pop(index, 0)
                    self.index2replay_ratio[index] = 1.0  # 如果没有全做对，下次正常采样
                if passrate == 0:
                    self.passrate0_set[index] += 1
                else:
                    self.passrate0_set.pop(index, 0)
        
        # 可选：重置采样队列，下次从全体数据中进行采样
        if self.reset_queue:
            self.queue.clear()
    
    def is_valid_passrate0(self, index, threshold=3):
        passrate0_count = self.passrate0_set.get(index, 0)
        if passrate0_count >= threshold:
            self.passrate0_set[index] = 0
            return True
        else:
            return False

    def accept(self, passrate_list):
        self.feedback(passrate_list)

    def get_dataset_len(self):
        return len(self.dataset)

    def clear_state(self):
        for k in self.states:
            ori_value = getattr(self, k)
            if isinstance(ori_value, list):
                setattr(self, k, ori_value[:0])
            elif isinstance(ori_value, set):
                setattr(self, k, set())
            elif isinstance(ori_value, int):
                setattr(self, k, 0)
            else:
                raise NotImplementedError(f"clear_state not implemented for {ori_value}(type {type(ori_value)})")

    def _get_nested_attr(self, obj, attr_path: str):
        # 支持通过点号分隔的路径访问嵌套属性
        # 例如: "dataset.thought_buffer.buffer" -> obj.dataset.thought_buffer.buffer
        parts = attr_path.split('.')
        current = obj
        for part in parts:
            if current is None:
                return None
            current = getattr(current, part, None)
        return current

    def _set_nested_attr(self, obj, attr_path: str, value):
        # 支持通过点号分隔的路径设置嵌套属性
        # 例如: "dataset.thought_buffer.buffer" -> obj.dataset.thought_buffer.buffer = value
        parts = attr_path.split('.')
        # 获取到最后一个属性之前的对象
        current = obj
        for part in parts[:-1]:
            if current is None:
                # 如果中间属性为None，给出警告并跳过设置
                # 例如：如果 thought_buffer 为 None，则无法设置 buffer
                print(f"Warning: Cannot set attribute '{attr_path}': intermediate attribute '{part}' is None. Skipping.")
                return
            current = getattr(current, part, None)
            if current is None:
                # 如果中间属性为None，给出警告并跳过设置
                print(f"Warning: Cannot set attribute '{attr_path}': intermediate attribute '{part}' is None. Skipping.")
                return
        # 设置最后一个属性
        setattr(current, parts[-1], value)

    def state_dict(self):
        states = {}
        for k in self.states:
            if '.' in k:
                # 嵌套属性，使用辅助函数获取
                states[k] = self._get_nested_attr(self, k)
            else:
                # 直接属性
                states[k] = getattr(self, k)
        return states


    # def state_dict(self):
    #     states = {}
    #     for k in self.states:
    #         states[k] = getattr(self, k)
    #     return states

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # first check keys
        if not set(state_dict.keys()) == set(self.states):
            print(f"state_dict keys {state_dict.keys()} not match states {self.states}")
        for k in state_dict.keys():
            if k in self.states:
                # 对于 index2replay_ratio，需要确保长度与当前数据集长度一致
                if k == 'index2replay_ratio':
                    saved_ratio = state_dict[k]
                    current_dataset_len = len(self.dataset)
                    if len(saved_ratio) != current_dataset_len:
                        print(f"Error: index2replay_ratio length mismatch. Saved: {len(saved_ratio)}, Current dataset: {current_dataset_len}. Reinitializing.")
                        raise Exception("index2replay_ratio length mismatch")
                    else:
                        setattr(self, k, saved_ratio)
                else:
                    # 支持嵌套属性的设置
                    if '.' in k:
                        self._set_nested_attr(self, k, state_dict[k])
                    else:
                        setattr(self, k, state_dict[k])


# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: data_loader.py
@time: 2020/12/04 22:58:42

这一行开始写关于本文件的说明与解释


'''
from typing import List, Dict
import tensorflow as tf

from vocab import Vocab


class DataLoader:

    def __init__(self, instances: List, vocab: Vocab, batch_size: int,
                 drop_last: bool = False, shuffle: bool = True, shuffle_buffer_size: int = None,
                 reshuffle_each_iteration: bool = True, num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
                 prefretch_buffer_size: int = 3):
        self.instances = instances
        self.vocab = vocab

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size or len(instances)
        self.reshuffle_each_iteration = reshuffle_each_iteration
        self.num_paraller_calls = num_parallel_calls
        self.prefretch_buffer_size = prefretch_buffer_size

        self.dataset = self._gen_dataset()

    def _data_generator(self):
        for instance in self.instances:
            tokens = self.vocab.encode(instance["tokens"], name="tokens")
            label = self.vocab.encode(instance["label"], name="label")

            yield tokens, label

    @staticmethod
    def map_func(*args):
        return args

    def _gen_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(
            self._data_generator, output_types=(tf.int32, tf.int32))
        dataset = dataset.padded_batch(
            batch_size=self.batch_size, drop_remainder=self.drop_last, padded_shapes=(32, 1))
        if self.shuffle:
            dataset = dataset.shuffle(
                self.shuffle_buffer_size, self.reshuffle_each_iteration)
        dataset = dataset.map(
            self.map_func, num_parallel_calls=self.num_paraller_calls, deterministic=False)
        # dataset = dataset.prefetch(self.prefretch_buffer_size)
        return dataset

    def __iter__(self):
        for x, y in self.dataset:
            yield x, y

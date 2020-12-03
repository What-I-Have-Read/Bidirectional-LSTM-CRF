# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: model.py
@time: 2020/12/03 00:21:09

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.text.crf import crf_log_likelihood


class BiLstmCrf(Model):
    def __init__(self, vocab_size, embedding_size, hidden_size, tag_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            hidden_size, return_sequences=True), merge_mode="concat")
        self.dense = tf.keras.layers.Dense(tag_size)
        self.crf = CRF(tag_size)

    @tf.function()
    def call(self, inputs, training=None, mask=None):
        inputs = self.embedding(inputs)
        inputs = self.bi_lstm(inputs)
        inputs = self.dense(inputs)
        # crf layer 会返回四个参数 decoded_sequences, potentials, sequence_lengths, chain_kernel
        return self.crf(inputs)

    def get_config(self):
        return super().get_config()


class CrfLoss(tf.keras.losses.Loss):
    def __init__(self, name="crf_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        label_sequences = tf.convert_to_tensor(y_true)
        decoded_sequences, potentials, sequence_lengths, chain_kernel = y_pred
        log_likelihood, _ = crf_log_likelihood(
            inputs=potentials, tag_indices=label_sequences, sequence_lengths=sequence_lengths,
            transition_params=chain_kernel)
        loss = - tf.reduce_mean(log_likelihood)
        return loss


class CrfAcc(tf.keras.metrics.Metric):
    def __init__(self, name="crf_acc", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)

        self.init_shape = []

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros", dtype=self.dtype
            )

        self.positives = _zero_wt_init("positive")
        self.count = _zero_wt_init("count")

    def update_state(self, y_true, y_pred):
        decoded_sequences, potentials, sequence_lengths, chain_kernel = y_pred
        correct_prediction = tf.equal(
            tf.convert_to_tensor(decoded_sequences, dtype=tf.int32),
            tf.convert_to_tensor(y_true, dtype=tf.int32)
        )
        correct_prediction = tf.cast(correct_prediction, dtype=tf.int32)
        # 使用tf.int32 经过reduce_mean后，只要序列中含有0，则结果为0
        correct_prediction = tf.reduce_mean(correct_prediction, axis=-1)
        self.positives.assign_add(
            tf.cast(tf.reduce_sum(correct_prediction), dtype=tf.float32))
        self.count.assign_add(tf.constant(
            len(sequence_lengths), dtype=tf.float32))

    def result(self):
        return self.positives / self.count

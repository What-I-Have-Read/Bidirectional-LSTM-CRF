# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: trainer.py
@time: 2020/12/03 00:35:17

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf


class Trainer:
    def __init__(self,
                 model: tf.keras.models.Model,
                 training_dataset: tf.data.Dataset,
                 validation_dataset: tf.data.Dataset,
                 loss_object: tf.keras.losses.Loss,
                 acc_metric: tf.keras.metrics.Metric,
                 optimizer: tf.keras.optimizers.Optimizer,
                 epochs: int,
                 output_dir: str
                 ):
        self.model = model
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.loss_object = loss_object()

        self.acc_metric = acc_metric(name="acc")
        self.val_acc_metric = acc_metric(name="val_acc")

        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.val_loss_metric = tf.keras.metrics.Mean(name="val_loss")

        self.optimizer = optimizer
        self.epochs = epochs

        self.output_dir = output_dir

    @tf.function()
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_object(y, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.loss_metric.update_state(loss)
        self.acc_metric.update_state(y, y_pred)

    @tf.function()
    def valid_step(self, x, y):
        y_pred = self.model(x, training=False)
        loss = self.loss_object(y, y_pred)

        self.val_loss_metric.update_state(loss)
        self.val_acc_metric.update_state(y, y_pred)

    def train(self):
        best_acc = 0.0
        for epoch in tf.range(self.epochs):
            tf.print(f"Epoch {epoch+1}/{self.epochs}:")

            self.acc_metric.reset_states()
            self.loss_metric.reset_states()

            self.val_acc_metric.reset_states()
            self.val_loss_metric.reset_states()

            bar = tf.keras.utils.Progbar(len(self.training_dataset), unit_name="sample",
                                         stateful_metrics={"loss", "acc"})
            log_values = []
            for x, y in self.training_dataset:
                self.train_step(x, y)
                log_values.append(("loss", self.loss_metric.result().numpy()))
                log_values.append(("acc", self.acc_metric.result().numpy()))
                bar.add(1, log_values)

            for x, y in self.validation_dataset:
                self.valid_step(x, y)
            tf.print(
                f"val_loss: {self.val_loss_metric.result().numpy()} - val_acc: {self.val_acc_metric.result().numpy()}")
            if self.val_acc_metric.result().numpy() > best_acc:
                tf.print("val acc improved, save model")
                best_acc = self.val_acc_metric.result().numpy()
                self.model.save(self.output_dir, save_format="tf")
            else:
                tf.print("val acc not improved")

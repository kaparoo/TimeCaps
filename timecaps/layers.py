# -*- coding: utf-8 -*-

from timecaps.param import TimeCapsParam

import tensorflow as tf
from typing import List

class Squash(tf.keras.layers.Layer):
    def __init__(self, eps: float = 1e-7, layer_name: str = "squash") -> None:
        super(Squash, self).__init__(name=layer_name)
        self.eps = eps

    def call(self, input_vector: tf.Tensor) -> tf.Tensor:
        norm = tf.norm(input_vector, axis=-1, keepdims=True)
        coef = norm**2 / (norm**2 + 1)
        unit = input_vector / (norm + self.eps)
        return coef * unit


class PrimaryTimeCapsA(tf.keras.layers.Layer):
    def __init__(self, param: TimeCapsParam) -> None:
        super(PrimaryTimeCapsA, self).__init__(name="primary_timecaps_a")
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.conv = tf.keras.layers.Conv1D(
            input_shape=input_shape[1:],
            filters=self.param.num_primary_a * self.param.dim_primary_a,
            kernel_size=self.param.kernel_primary_a,
            padding='same',
            activation=tf.keras.activations.relu)
        self.reshape = tf.keras.layers.Reshape(target_shape=[
            -1, self.param.num_primary_a, self.param.dim_primary_a
        ])
        self.squash = Squash()
        self.built = True

    def call(self, feature_maps: tf.Tensor) -> tf.Tensor:
        return self.squash(self.reshape(self.conv(feature_maps)))


class SecondaryTimeCapsA(tf.keras.layers.Layer):
    def __init__(self, param: TimeCapsParam) -> None:
        super(SecondaryTimeCapsA, self).__init__(name="secondary_timecaps_a")
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.reshape1 = tf.keras.layers.Reshape(
            input_shape=input_shape[1:],
            target_shape=[
                -1, self.param.num_primary_a * self.param.dim_primary_a, 1
            ])
        self.conv = tf.keras.layers.Conv2D(
            filters=self.param.num_timecaps_a * self.param.dim_timecaps,
            kernel_size=[
                self.param.kernel_timecaps_a, self.param.dim_primary_a
            ],
            strides=[1, self.param.dim_primary_a],
            padding='same',
            activation=tf.keras.activations.relu)
        self.reshape2 = tf.keras.layers.Reshape(target_shape=[
            -1, self.param.num_primary_a, self.param.num_timecaps_a,
            self.param.dim_timecaps
        ])
        self.permute = tf.keras.layers.Permute([1, 3, 2, 4])
        self.B = self.add_weight(shape=[
            input_shape[1], self.param.num_timecaps_a, 1,
            self.param.num_primary_a
        ],
                                 dtype=tf.float32,
                                 trainable=True)
        self.squash = Squash()
        self.built = True

    def call(self, primary_timecaps: tf.Tensor) -> tf.Tensor:
        U_hat = self.permute(
            self.reshape2(self.conv(self.reshape1(primary_timecaps))))
        A = tf.matmul(U_hat, U_hat, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.param.dim_primary_a, dtype=tf.float32))
        C = tf.nn.softmax(tf.reduce_sum(A, axis=-2, keepdims=True), axis=-1)
        return self.squash(tf.squeeze(tf.matmul(C + self.B, U_hat), axis=-2))


class PrimaryTimeCapsB(tf.keras.layers.Layer):
    def __init__(self, param: TimeCapsParam) -> None:
        super(PrimaryTimeCapsB, self).__init__(name="primary_timecaps_b")
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.pointwise_conv = tf.keras.layers.Conv1D(
            input_shape=input_shape[1:],
            filters=self.param.tot_primary_b,
            kernel_size=1,
            padding='same',
            activation=tf.keras.activations.relu)
        self.conv = tf.keras.layers.Conv1D(
            filters=self.param.tot_primary_b,
            kernel_size=self.param.kernel_primary_b,
            padding='same',
            activation=tf.keras.activations.relu)
        self.segmenting = tf.keras.layers.Reshape(target_shape=[
            -1, self.param.num_segments_b, self.param.tot_primary_b
        ])
        self.squash = Squash()
        self.built = True

    def call(self, feature_maps: tf.Tensor) -> tf.Tensor:
        feature_maps = self.pointwise_conv(feature_maps)
        return self.squash(self.segmenting(self.conv(feature_maps)))


class SecondaryTimeCapsB(tf.keras.layers.Layer):
    def __init__(self, param: TimeCapsParam) -> None:
        super(SecondaryTimeCapsB, self).__init__(name="secondary_timecaps_b")
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.reshape1 = tf.keras.layers.Reshape(
            input_shape=input_shape[1:],
            target_shape=[
                -1, self.param.num_segments_b * self.param.tot_primary_b, 1
            ])
        self.conv = tf.keras.layers.Conv2D(
            filters=self.param.num_timecaps_b * self.param.dim_timecaps,
            kernel_size=[
                self.param.kernel_timecaps_b, self.param.tot_primary_b
            ],
            strides=[1, self.param.tot_primary_b],
            padding='same',
            activation=tf.keras.activations.relu)
        self.reshape2 = tf.keras.layers.Reshape(target_shape=[
            -1, self.param.num_segments_b, self.param.num_timecaps_b,
            self.param.dim_timecaps
        ])
        self.permute = tf.keras.layers.Permute([1, 3, 2, 4])
        self.B = self.add_weight(shape=[
            input_shape[1], self.param.num_timecaps_b, 1,
            self.param.num_segments_b
        ],
                                 dtype=tf.float32,
                                 trainable=True)
        self.squash = Squash()
        self.built = True

    def call(self, primary_timecaps: tf.Tensor) -> tf.Tensor:
        U_hat = self.permute(
            self.reshape2(self.conv(self.reshape1(primary_timecaps))))
        A = tf.matmul(U_hat, U_hat, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.param.dim_primary_a, dtype=tf.float32))
        C = tf.nn.softmax(tf.reduce_sum(A, axis=-2, keepdims=True), axis=-1)
        return self.squash(tf.squeeze(tf.matmul(C + self.B, U_hat), axis=-2))


class ConcatTimeCaps(tf.keras.layers.Layer):
    def __init__(self, param: TimeCapsParam) -> None:
        super(ConcatTimeCaps, self).__init__(name="concatenated_timecaps")
        self.param = param

    def build(self, _) -> None:
        self.alpha = self.add_weight(name="alpha",
                                     shape=[1],
                                     dtype=tf.float32,
                                     trainable=True)
        self.beta = self.add_weight(name="beta",
                                    shape=[1],
                                    dtype=tf.float32,
                                    trainable=True)
        self.reshape = tf.keras.layers.Reshape(
            target_shape=[-1, self.param.dim_timecaps])
        self.built = True

    def call(self, timecaps: List[tf.Tensor]) -> tf.Tensor:
        timecaps_a, timecaps_b = timecaps
        timecaps_a = self.reshape(timecaps_a)
        timecaps_b = self.reshape(timecaps_b)
        return tf.keras.layers.Concatenate(axis=-2)(
            [self.alpha * timecaps_a, self.beta * timecaps_b])


class Classifier(tf.keras.layers.Layer):
    def __init__(self, param: TimeCapsParam) -> None:
        super(Classifier, self).__init__(name="timecaps_classifier")
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.reshape1 = tf.keras.layers.Reshape(input_shape=input_shape[1:],
                                                target_shape=[-1, 1])
        self.conv = tf.keras.layers.Conv1D(
            filters=self.param.num_classes * self.param.dim_classes,
            kernel_size=self.param.dim_timecaps,
            strides=self.param.dim_timecaps,
            padding='same',
            activation=tf.keras.activations.relu)
        self.reshape2 = tf.keras.layers.Reshape(
            target_shape=[-1, self.param.num_classes, self.param.dim_classes])
        self.permute = tf.keras.layers.Permute([2, 1, 3])
        self.B = self.add_weight(
            shape=[self.param.num_classes, 1, input_shape[1]],
            dtype=tf.float32,
            trainable=True)
        self.squash = Squash()
        self.built = True

    def call(self, timecaps: tf.Tensor) -> tf.Tensor:
        U_hat = self.permute(self.reshape2(self.conv(self.reshape1(timecaps))))
        A = tf.matmul(U_hat, U_hat, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.param.dim_primary_a, dtype=tf.float32))
        C = tf.nn.softmax(tf.reduce_sum(A, axis=-2, keepdims=True), axis=-1)
        S = tf.squeeze(tf.matmul(C + self.B, U_hat), axis=-2)
        return tf.norm(self.squash(S), axis=-1, keepdims=True)

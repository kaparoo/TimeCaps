# -*- coding: utf-8 -*-

import tensorflow as tf

from timecaps import layers
from timecaps import param


# class TimeCaps(tf.keras.models.Model):
#     def __init__(self,
#                  param: param.TimeCapsParam,
#                  model_name: str = "TimeCaps") -> None:
#         super(TimeCaps, self).__init__(name=model_name)
#         self.param = param

#     def build(self, input_shape: tf.TensorShape) -> None:
#         self.conv = tf.keras.layers.Conv1D(
#             name="convolutional_layer",
#             input_shape=input_shape[1:],
#             filters=self.param.filter_conv,
#             kernel_size=self.param.kernel_conv,
#             padding='same',
#             activation=tf.keras.activations.relu)
#         self.primary_a = layers.PrimaryTimeCapsA(self.param)
#         self.primary_b = layers.PrimaryTimeCapsB(self.param)
#         self.timecaps_a = layers.SecondaryTimeCapsA(self.param)
#         self.timecaps_b = layers.SecondaryTimeCapsB(self.param)
#         self.concat = layers.ConcatTimeCaps(self.param)
#         self.classifier = layers.Classifier(self.param)
#         self.built = True

#     def call(self, input_signals: tf.Tensor) -> tf.Tensor:
#         feature_maps = self.conv(input_signals)
#         primary_timecaps_a = self.primary_a(feature_maps)
#         primary_timecaps_b = self.primary_b(feature_maps)
#         timecaps_a = self.timecaps_a(primary_timecaps_a)
#         timecaps_b = self.timecaps_b(primary_timecaps_b)
#         timecaps = self.concat([timecaps_a, timecaps_b])
#         class_probs = self.classifier(timecaps)
#         return class_probs


def make_model(param: param.TimeCapsParam,
               model_name: str = "TimeCaps") -> tf.keras.Model:
    input_signal = tf.keras.layers.Input(name="input_signal",
                                         shape=[param.signal_length, 1])
    feature_maps = tf.keras.layers.Conv1D(
        name="convolutional_layer",
        filters=param.filter_conv,
        kernel_size=param.kernel_conv,
        padding='same',
        activation=tf.keras.activations.relu)(input_signal)
    primary_timecaps_a = layers.PrimaryTimeCapsA(param)(feature_maps)
    primary_timecaps_b = layers.PrimaryTimeCapsB(param)(feature_maps)
    timecaps_a = layers.SecondaryTimeCapsA(param)(primary_timecaps_a)
    timecaps_b = layers.SecondaryTimeCapsB(param)(primary_timecaps_b)
    timecaps = layers.ConcatTimeCaps(param)([timecaps_a, timecaps_b])
    class_probs = layers.Classifier(param)(timecaps)
    return tf.keras.Model(name="TimeCaps",
                          inputs=input_signal,
                          outputs=class_probs)

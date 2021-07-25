# -*- coding: utf-8 -*-


class TimeCapsParam(object):

    __slots__ = [
        "dim_classes", "dim_primary_a", "dim_timecaps", "filter_conv",
        "kernel_conv", "kernel_primary_a", "kernel_primary_b",
        "kernel_timecaps_a", "kernel_timecaps_b", "num_classes",
        "num_primary_a", "num_segments_b", "num_timecaps_a", "num_timecaps_b",
        "signal_length", "tot_primary_b"
    ]

    def __init__(self,
                 signal_length: int = 360,
                 filter_conv: int = 64,
                 kernel_conv: int = 7,
                 kernel_primary_a: int = 5,
                 num_primary_a: int = 8,
                 dim_primary_a: int = 8,
                 kernel_timecaps_a: int = 3,
                 num_timecaps_a: int = 8,
                 kernel_primary_b: int = 5,
                 num_segments_b: int = 10,
                 tot_primary_b: int = 32,
                 kernel_timecaps_b: int = 3,
                 num_timecaps_b: int = 10,
                 dim_timecaps: int = 8,
                 num_classes: int = 13,
                 dim_classes: int = 16) -> None:

        self.signal_length = signal_length

        # Convolutional Layer
        self.filter_conv = filter_conv
        self.kernel_conv = kernel_conv

        # TimeCaps Cell A
        self.kernel_primary_a = kernel_primary_a
        self.num_primary_a = num_primary_a
        self.dim_primary_a = dim_primary_a
        self.kernel_timecaps_a = kernel_timecaps_a
        self.num_timecaps_a = num_timecaps_a

        # TimeCaps Cell B
        self.kernel_primary_b = kernel_primary_b
        self.num_segments_b = num_segments_b
        self.tot_primary_b = tot_primary_b
        self.kernel_timecaps_b = kernel_timecaps_b
        self.num_timecaps_b = num_timecaps_b

        # TimeCaps (concatenated)
        self.dim_timecaps = dim_timecaps

        # Classification Capsules
        self.num_classes = num_classes
        self.dim_classes = dim_classes


def make_param(signal_length: int = 360,
               filter_conv: int = 64,
               kernel_conv: int = 7,
               kernel_primary_a: int = 5,
               num_primary_a: int = 8,
               dim_primary_a: int = 8,
               kernel_timecaps_a: int = 3,
               num_timecaps_a: int = 8,
               kernel_primary_b: int = 5,
               num_segments_b: int = 10,
               tot_primary_b: int = 32,
               kernel_timecaps_b: int = 3,
               num_timecaps_b: int = 10,
               dim_timecaps: int = 8,
               num_classes: int = 13,
               dim_classes: int = 16) -> TimeCapsParam:
    return TimeCapsParam(dim_classes, dim_primary_a, dim_timecaps, filter_conv,
                         kernel_conv, kernel_primary_a, kernel_primary_b,
                         kernel_timecaps_a, kernel_timecaps_b, num_classes,
                         num_primary_a, num_segments_b, num_timecaps_a,
                         num_timecaps_b, signal_length, tot_primary_b)

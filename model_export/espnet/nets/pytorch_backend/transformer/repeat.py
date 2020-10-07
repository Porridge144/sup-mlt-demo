#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Repeat the same layer definition."""

import torch


class MultiSequentialEncoder(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, x, mask):
        """Repeat."""
        for m in self:
            x, mask = m(x, mask)
        return x, mask


def repeat_encoder(N, fn):
    """Repeat module N times.

    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequentialEncoder
    """
    return MultiSequentialEncoder(*[fn() for _ in range(N)])

class MultiSequentialDecoder(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Repeat."""
        for m in self:
            tgt, tgt_mask, memory, memory_mask = m(tgt, tgt_mask, memory, memory_mask)
        return tgt, tgt_mask, memory, memory_mask


def repeat_decoder(N, fn):
    """Repeat module N times.

    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequentialDecoder
    """
    return MultiSequentialDecoder(*[fn() for _ in range(N)])
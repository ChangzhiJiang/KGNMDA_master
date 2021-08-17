# -*- coding: utf-8 -*-

from .aggregator import SumAggregator, ConcatAggregator, NeighAggregator, Sum_concat_Aggregator
from .mapping import DiseaseMicrobeScore

Aggregator = {
    'sum': SumAggregator,
    'concat': ConcatAggregator,
    'neigh': NeighAggregator,
    'sum_concat':Sum_concat_Aggregator
}

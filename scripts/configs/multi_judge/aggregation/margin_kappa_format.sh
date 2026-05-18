#!/bin/bash
# Multi-judge aggregation: margin + kappa + format
# Reward = margin_weight * avg(margin) + format_weight * format_valid + kappa_weight * clip(fleiss_kappa, 0, 1)
# Weights are configurable via MULTI_JUDGE_{MARGIN,FORMAT,KAPPA}_WEIGHT (defaults: 0.5, 0.3, 0.2).

MULTI_JUDGE_AGGREGATION="margin_kappa_format"

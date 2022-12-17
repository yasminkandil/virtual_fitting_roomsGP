# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .data.datasets import builtin  # just to register data
from .converters import builtin as builtin_converters  # register converters
from .config import (
    add_densepose_config,
    add_densepose_head_config,
    add_hrnet_config,
    add_dataset_category_config,
    add_bootstrap_config,
    load_bootstrap_config,
)
from .evaluator import DensePoseCOCOEvaluator
from .modeling.roi_heads import DensePoseROIHeads
from .data.structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData
from .modeling.test_time_augmentation import (
    DensePoseGeneralizedRCNNWithTTA,
    DensePoseDatasetMapperTTA,
)
from .utils.transform import load_from_cfg
from .modeling.hrfpn import build_hrfpn_backbone

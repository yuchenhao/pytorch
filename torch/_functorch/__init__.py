# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import sys
from .._config_utils import make_config_dataclass
from . import config

FunctorchConfig = type(config)
__all__ = ['config', 'FunctorchConfig']

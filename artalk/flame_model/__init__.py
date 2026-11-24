#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

from .FLAME import FLAMEModel

# RenderMesh 依赖 pytorch3d，只用于自带的网格渲染；只推理系数的环境（如 aarch64 的 spark）不装 pytorch3d，
# 这里改为可选导入，缺包时 RenderMesh 为 None，用到时再报错。
try:
    from .renderer_utils import RenderMesh
except ImportError:
    RenderMesh = None

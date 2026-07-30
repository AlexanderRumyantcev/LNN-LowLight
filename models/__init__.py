# pipeline.py (RetinexLNNPipeline) удалён в "Pivot cleanup" (f61bc74) вместе со старым
# видео-пайплайном — этот __init__.py оставался stale и ломал любой import models.*.
# Текущие компоненты не требуют top-level реэкспорта, импортируются из подмодулей:
#   from models.temporal.cfc_probe_module import CfCProbeModule
#   from models.probe_interpolation import ProbeInterpolator
#   from models.baselines import NRDStyleBaseline, NRCStyleBaseline

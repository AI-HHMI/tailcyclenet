"""The box detector: network, data, training and checkpoints. Deployment lives in `tailcyclenet.infer`."""
from .checkpoint import LoadedDetector, load_detector, resolve_detector_checkpoint, tiled_input_wh

__all__ = ['LoadedDetector', 'load_detector', 'resolve_detector_checkpoint', 'tiled_input_wh']

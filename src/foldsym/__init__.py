"""foldsym: rule-based fold symmetry detection in 4D-STEM nanodiffraction patterns."""

from .classify import classify_pattern
from .config import CONFIGS, LABELS, V1, V2, ClassifierConfig, DetectorConfig, PipelineConfig, get_config
from .detect import detect_circular_speckles
from .pipeline import PatternResult, process_image, process_stack, summarize

__version__ = "1.0.0"

__all__ = [
    "CONFIGS",
    "LABELS",
    "V1",
    "V2",
    "ClassifierConfig",
    "DetectorConfig",
    "PatternResult",
    "PipelineConfig",
    "classify_pattern",
    "detect_circular_speckles",
    "get_config",
    "process_image",
    "process_stack",
    "summarize",
]

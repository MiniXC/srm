"""srm_eval — distribution-level speech representation evaluation."""

from srm_eval.data.blizzard import BlizzardData, load_blizzard, parse_filepath
from srm_eval.distance import frechet_distance, sliced_wasserstein, wasserstein_2_perdim
from srm_eval.correlate import correlate

__all__ = [
    "BlizzardData",
    "correlate",
    "frechet_distance",
    "load_blizzard",
    "parse_filepath",
    "sliced_wasserstein",
    "wasserstein_2_perdim",
]

"""CoreModelContract module."""

from dataclasses import dataclass

from redkg.visualization.config.parameters.defaults import Defaults


@dataclass
class CoreModelContract:
    """Core model contract base class."""

    nums: int | list
    forces: dict
    centers: list
    damping_factor: float = Defaults.damping_factor

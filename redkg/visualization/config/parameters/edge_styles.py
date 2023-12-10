"""Edge Styles parameters module."""

from redkg.visualization.utils.frozen_dataclass import reference
from redkg.visualization.utils.reference_base import ReferenceBase


@reference
class EdgeStyles(ReferenceBase):
    """Main EdgeStyles parameters class."""

    line: str = "line"
    circle: str = "circle"

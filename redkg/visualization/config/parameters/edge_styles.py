from redkg.visualization.utils.frozen_dataclass import reference
from redkg.visualization.utils.reference_base import ReferenceBase


@reference
class EdgeStyles(ReferenceBase):
    line: str = "line"
    circle: str = "circle"

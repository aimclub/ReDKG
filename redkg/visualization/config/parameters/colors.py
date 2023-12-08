"""Colors parameters module."""

from redkg.visualization.utils.frozen_dataclass import reference
from redkg.visualization.utils.reference_base import ReferenceBase


@reference
class Colors(ReferenceBase):
    """Main Colors parameters class."""
    red: str = "r"
    green: str = "g"
    gray: str = "gray"
    whitesmoke: str = "whitesmoke"

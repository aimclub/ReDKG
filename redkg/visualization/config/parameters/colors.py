from redkg.visualization.utils.frozen_dataclass import reference
from redkg.visualization.utils.reference_base import ReferenceBase


@reference
class Colors(ReferenceBase):
    red: str = "r"
    green: str = "g"
    gray: str = "gray"
    whitesmoke: str = "whitesmoke"

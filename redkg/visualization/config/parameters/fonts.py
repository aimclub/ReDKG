from redkg.visualization.utils.frozen_dataclass import reference
from redkg.visualization.utils.reference_base import ReferenceBase


@reference
class Fonts(ReferenceBase):
    sans_serif: str = "sans-serif"

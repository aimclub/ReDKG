"""Generator methods parameters module."""


from redkg.visualization.utils.frozen_dataclass import reference
from redkg.visualization.utils.reference_base import ReferenceBase


@reference
class GeneratorMethods(ReferenceBase):
    """Main GeneratorMethods parameter class."""

    custom: str = "custom"
    uniform: str = "uniform"
    low_order_first: str = "low_order_first"
    high_order_first: str = "high_order_first"

"""Fill sizes function."""

from typing import Any, Optional, Union

from redkg.visualization.exceptions.exceptions_classes import ParamsValidationException


def fill_sizes(custom_scales: Optional[Union[float, list]], default_value: Any, length: int) -> list:
    """Fill sizes function."""
    if custom_scales is None:
        return [default_value] * length

    elif isinstance(custom_scales, list):
        if len(custom_scales) != length:
            raise ParamsValidationException("The specified value list has the wrong length.")
        return [default_value * scale for scale in custom_scales]

    elif isinstance(custom_scales, float):
        return [default_value * custom_scales] * length

    elif isinstance(custom_scales, int):
        return [default_value * float(custom_scales)] * length

    else:
        raise ParamsValidationException("The specified value is not a valid type.")

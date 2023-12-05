"""Enum improved class."""

from enum import Enum, EnumMeta


class EnumMetaClass(EnumMeta):
    """Enum metaclass for values / list ability."""

    def __contains__(cls, item):
        """Contains magic function."""
        try:
            cls(item)
        except ValueError:
            return False
        return True


class EnumImproved(Enum, metaclass=EnumMetaClass):
    """Enum base with values / list ability."""

    @classmethod
    def list(cls):
        """List function."""
        return list(map(lambda c: c.value, cls))

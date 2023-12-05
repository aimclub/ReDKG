"""Reference Base module."""


class ReferenceBase:
    """Base dataclass for references (enums, info classes)."""

    @property
    def values(self):
        """Define 'values' method."""
        properties = []
        for attribute_name, attribute_value in self.__dict__.items():
            if not callable(attribute_value):
                properties.append(attribute_value)
        return properties

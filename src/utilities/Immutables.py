from dataclasses import dataclass

@dataclass(frozen=True)
class ImmutableArray:
    data: list

    def __post_init__(self):
        if not isinstance(self.data, list):
            raise ValueError("Data must be a nested list.")
        if len(self.data) == 0 or not all(isinstance(row, list) for row in self.data):
            raise ValueError("Data must be a non-empty nested list.")
        if not all(len(row) == len(self.data[0]) for row in self.data):
            raise ValueError("All rows in data must have the same length.")
        if not all(isinstance(value, (int, float)) for row in self.data for value in row):
            raise ValueError("All elements in data must be integers or floats.")

    @property
    def shape(self):
        height = len(self.data)
        width = len(self.data[0])
        channels = len(self.data[0][0])
        return (height, width, channels)

    def __repr__(self):
        return f"ImmutableArray(shape={self.shape})"
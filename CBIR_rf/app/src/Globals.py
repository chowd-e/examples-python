import enum

# Determines search method for distance computation
class Mode(enum.Enum):
   COLOR = enum.auto()
   INTENSITY = enum.auto()
   COLOR_INTENSITY = enum.auto()
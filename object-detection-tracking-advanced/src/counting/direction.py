"""
Direction Enumeration Module
Defines movement directions for counting
"""
from enum import Enum


class Direction(Enum):
    """Direction of movement for counting"""
    
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    IN = "in"
    OUT = "out"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"Direction.{self.name}"
    
    @classmethod
    def from_string(cls, direction_str: str) -> "Direction":
        """
        Create Direction from string
        
        Args:
            direction_str: Direction as string
            
        Returns:
            Direction enum
        """
        direction_str = direction_str.lower()
        for direction in cls:
            if direction.value == direction_str:
                return direction
        return cls.UNKNOWN
    
    def is_inward(self) -> bool:
        """Check if direction is inward"""
        return self in [Direction.IN, Direction.DOWN, Direction.RIGHT]
    
    def is_outward(self) -> bool:
        """Check if direction is outward"""
        return self in [Direction.OUT, Direction.UP, Direction.LEFT]

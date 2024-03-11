import math
import random

class LimitWorkspace:
    def __init__(self, point1, point2):
        self.points = [point1, point2]

    def check_limits(self, x, y):
        """Determine is the point is in the workspace range (x[0], y[0]) (x[1], y[1]) 
           Returns 1 if inside or 0 if outside.
        """
        # Extract the x and y values of the points
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]

        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        x = float(x)
        y = float(y)
        
        # Determine the min and max for x and y
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        
        # Check if the given point (x, y) is within the bounds
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return 1  # Inside
        else:
            return 0  # Outside
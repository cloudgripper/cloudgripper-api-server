#  also fixes the origin 
class CoordinateNormalizer:
    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    # y = (x-a)/(b-a)
    def x_to_norm(self, x_mm: float) -> float:
        return float(x_mm - self.xmin)/float(self.xmax - self.xmin)

    def y_to_norm(self, y_mm: float) -> float:
        return float(y_mm - self.ymin)/float(self.ymax - self.ymin)

    # x = (b-a)y+a
    def x_to_mm(self, x_norm: float) -> float:
        return float(x_norm * (self.xmax - self.xmin) + self.xmin)

    def dx_to_dmm(self, dx_norm: float) -> float:
        return dx_norm / (self.xmax - self.xmin)

    def y_to_mm(self, y_norm: float) -> float:
        return float(y_norm * (self.ymax - self.ymin) + self.ymin)

    def dy_to_dmm(self, dy_norm: float) -> float:
        return dy_norm / (self.ymax - self.ymin)
    
    def xy_to_norm(self, x_mm: float, y_mm: float):
        return self.x_to_norm(x_mm), self.y_to_norm(y_mm)

    def xy_to_mm(self, x_norm: float, y_norm: float):
        return self.x_to_mm(x_norm), self.y_to_mm(y_norm)

    def dxdy_to_dmm(self, dx_norm: float, dy_norm: float):
        return self.dx_to_dmm(dx_norm), self.dy_to_dmm(dy_norm)
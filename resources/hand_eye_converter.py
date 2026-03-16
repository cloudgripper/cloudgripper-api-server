import yaml
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class HandEyeConverter:
    def __init__(self, filepath):
        self.px_Y_model = None
        self.py_X_model = None
        self.Y_px_model = None
        self.X_py_model = None
        self.data_dict = self._yaml_to_dict(filepath)
        self._fit_data()

    def _yaml_to_dict(self, yaml_file_path):
        # Load data from the YAML file
        with open(yaml_file_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.SafeLoader)
        
        data_dict = {}
        for key, values in data.items():
            formatted_key = tuple(map(int, key.strip('()').split(',')))
            # Since the values are already lists, we convert them directly to tuples
            formatted_values = tuple(map(float, values))
            data_dict[formatted_key] = formatted_values
    
        return data_dict

    def _fit_data(self):
        px_values = []
        py_values = []
        X_values = []
        Y_values = []

        for key, value in self.data_dict.items():
            px, py = key
            X, Y = value
            px_values.append(px)
            py_values.append(py)
            X_values.append(X)
            Y_values.append(Y)

        px_values = np.array(px_values).reshape(-1, 1)
        py_values = np.array(py_values).reshape(-1, 1)
        X_values = np.array(X_values).reshape(-1, 1)
        Y_values = np.array(Y_values).reshape(-1, 1)

        # Fit px to Y and Y to px
        self.px_Y_model = self._polynomial_regression(px_values, Y_values)
        self.Y_px_model = self._polynomial_regression(Y_values, px_values)

        # Fit py to X and X to py
        self.py_X_model = self._polynomial_regression(py_values, X_values)
        self.X_py_model = self._polynomial_regression(X_values, py_values)

    def _polynomial_regression(self, x, y, degree=4):
        poly_features = PolynomialFeatures(degree=degree)
        x_poly = poly_features.fit_transform(x)
        model = LinearRegression().fit(x_poly, y)
        return {"poly_features": poly_features, "model": model}

    def px_py_to_x_y(self, px, py):
        px = np.array([[px]])
        py = np.array([[py]])

        y_poly = self.px_Y_model["poly_features"].transform(px)
        Y = self.px_Y_model["model"].predict(y_poly)[0][0]

        x_poly = self.py_X_model["poly_features"].transform(py)
        X = self.py_X_model["model"].predict(x_poly)[0][0]

        return X, Y
    
    def x_y_to_px_py(self, X, Y):
        X = np.array([[X]])
        Y = np.array([[Y]])

        y_poly = self.Y_px_model["poly_features"].transform(Y)
        px = self.Y_px_model["model"].predict(y_poly)[0][0]

        x_poly = self.X_py_model["poly_features"].transform(X)
        py = self.X_py_model["model"].predict(x_poly)[0][0]

        # Clipping the values to ensure they are within valid pixel boundaries
        px = max(0, min(639, int(round(px))))  # Assuming 1920 as max x-resolution
        py = max(0, min(479, int(round(py))))   # Assuming 480 as max y-resolution
        
        return int(round(px)), int(round(py))

if __name__ == "__main__":
    # Testing
    converter = HandEyeConverter('./calibration/camera-to-robot-cr1.yaml')

    x, y = 0.011338939475859733, 0.13144378681284707
    print(converter.x_y_to_px_py(x, y)) 

import numpy as np
import collections


class Line(object):
    # This class is to draw 2D polynomial approximation lines on the
    def __init__(self, x, y, height, width):
        self.height = height
        self.width = width
        self.recent_x_fitted = collections.deque(maxlen=5)
        self.recent_fits = collections.deque(maxlen=5)
        self.current_fit = None

        self.fit(x, y)

    def fit(self, x, y):
        if len(y) > 0 and (self.current_fit is None or np.max(y) - np.min(y) > self.height * 0.625):
            self.current_fit = np.polyfit(y, x, 2)
            self.recent_fits.append(self.current_fit)
            self.recent_x_fitted.append(x)

    def points(self):
        # For y points between 0 and height, we find the x-coordinates
        y_points = np.linspace(0, self.height - 1, self.height)
        best_fit = np.array(self.recent_fits).mean(axis=0)
        best_fit_x = best_fit[0] * y_points ** 2 + best_fit[1] * y_points + best_fit[2]
        return np.stack((best_fit_x, y_points)).astype(int).T

    def measure_curvature(self):
        points = self.points()
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        x = points[:, 0]
        y = points[:, 1]

        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        curve_radius = ((1 + (2 * fit_cr[0] * 720 * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) \
                       / np.absolute(2 * fit_cr[0])
        return int(curve_radius)

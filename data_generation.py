import numpy as np
from tqdm import tqdm

from config import *


#general simulation parameters



#magnetic field data generation
class magnetic_field_simulator:
    from config import MAGNET_CONFIG, AOI_CONFIG
    def __init__(self):
        self.height = MAGNET_CONFIG['height']
        self.min_side_length = MAGNET_CONFIG['min_side_length']
        self.max_side_length = MAGNET_CONFIG['max_side_length']
        self.min_M = MAGNET_CONFIG['min_M']
        self.max_M = MAGNET_CONFIG['max_M']

        self.x_start = AOI_CONFIG['x_start']
        self.y_start = AOI_CONFIG['y_start']
        self.x_end = AOI_CONFIG['x_end']
        self.y_end = AOI_CONFIG['y_end']

    def calculate_field(self,
                        x_grid: np.ndarray,
                        y_grid: np.ndarray,
                        magnet_params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate magnetic field H at grid points from a magnetized prism.

        Args:
            x_grid: X coordinates of evaluation points [cm]
            y_grid: Y coordinates of evaluation points [cm]
            magnet_params: Dictionary with keys:
                - x, y: center position [cm]
                - a: side length [cm]
                - Mx, My: magnetization components [T]

        Returns:
            Hx, Hy: Magnetic field components [T] (scaled by mu_0)
        """
        # Convert cm to meters
        x_eval = x_grid / 100.0
        y_eval = y_grid / 100.0
        x_mag = magnet_params['x'] / 100.0
        y_mag = magnet_params['y'] / 100.0
        z_mag = 0.005  # 0.5 cm in meters
        z_eval = 0.005  # Evaluation at z = 0.5 cm

        # Magnet dimensions
        a = magnet_params['a'] / 100.0  # side length [m]
        h = 0.01  # height 1 cm [m]

        # Magnetization vector (convert back from normalized form)
        # Paper uses: Mx = (|M| - 0.5) * cos(phi), My = (|M| - 0.5) * sin(phi)
        # So |M| = sqrt(Mx^2 + My^2) + 0.5
        Mx = magnet_params['Mx']
        My = magnet_params['My']
        M_mag = np.sqrt(Mx ** 2 + My ** 2) + 0.5  # T

        if M_mag > 0.5:  # Avoid division by zero
            mx = Mx / (M_mag - 0.5)
            my = My / (M_mag - 0.5)
        else:
            mx, my = 0.0, 0.0

        M = np.array([mx * M_mag, my * M_mag, 0.0])  # [T]

        # Use magnetic dipole approximation for efficiency
        # For more accuracy, implement full analytical solution or use MagTense
        Hx = np.zeros_like(x_eval)
        Hy = np.zeros_like(y_eval)

        # Magnetic moment [A⋅m²]
        volume = a * a * h  # [m³]
        m = (M / self.mu_0) * volume  # [A⋅m²]

        # Position vector from magnet center to evaluation points
        rx = x_eval - x_mag
        ry = y_eval - y_mag
        rz = z_eval - z_mag

        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        r3 = r ** 3
        r5 = r ** 5

        # Avoid singularities
        mask = r > a / 2

        # Dipole field formula: H = (1/4π) * [(3(m⋅r̂)r̂ - m) / r³]
        m_dot_r = m[0] * rx + m[1] * ry + m[2] * rz

        Hx[mask] = (1 / (4 * np.pi)) * (
                (3 * m_dot_r[mask] * rx[mask] / r5[mask]) - (m[0] / r3[mask])
        )
        Hy[mask] = (1 / (4 * np.pi)) * (
                (3 * m_dot_r[mask] * ry[mask] / r5[mask]) - (m[1] / r3[mask])
        )

        # Scale by mu_0 to get units of [T] as in paper
        Hx_scaled = Hx * self.mu_0
        Hy_scaled = Hy * self.mu_0

        return Hx_scaled, Hy_scaled
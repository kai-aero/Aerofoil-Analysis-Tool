# -*- coding: utf-8 -*-
"""
Generates and plots the coordinates for a 4-digit NACA aerofoil.

This script contains functions to parse a NACA 4-digit code, calculate the
camber line and thickness distribution, and plot the final aerofoil geometry.
"""
import numpy as np
import matplotlib.pyplot as plt

def parse_naca_code(naca_code: str) -> tuple[float, float, float]:
    """
    Parses a 4-digit NACA string into its aerodynamic parameters.

    Args:
        naca_code (str): The 4-digit NACA code (e.g., "2412").

    Returns:
        A tuple containing:
        - m (float): Maximum camber as a fraction of the chord.
        - p (float): Position of maximum camber as a tenth of the chord.
        - t (float): Maximum thickness as a fraction of the chord.
    """
    if len(naca_code) != 4:
        raise ValueError("NACA code must be 4 digits.")
    
    m = int(naca_code[0]) / 100.0
    p = int(naca_code[1]) / 10.0
    t = int(naca_code[2:]) / 100.0
    
    return m, p, t

def create_cosine_spacing(num_points: int) -> np.ndarray:
    """
    Creates an array of x-coordinates with cosine spacing.

    This spacing clusters points at the leading and trailing edges of the aerofoil,
    which is beneficial for aerodynamic calculations.

    Args:
        num_points (int): The number of points to generate.

    Returns:
        np.ndarray: An array of x-coordinates from 0 to 1.
    """
    angles = np.linspace(0, np.pi, num_points)
    x_coords = 0.5 * (1 - np.cos(angles))
    return x_coords

def calculate_camber_line(m: float, p: float, x_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the camber line (yc) and its gradient (dyc_dx).

    Args:
        m (float): Maximum camber.
        p (float): Position of maximum camber.
        x_coords (np.ndarray): Array of x-coordinates.

    Returns:
        A tuple containing:
        - yc (np.ndarray): The y-coordinates of the camber line.
        - dyc_dx (np.ndarray): The gradient of the camber line.
    """
    # Return zero arrays if the airfoil is symmetric to avoid division by zero.
    if p == 0:
        return np.zeros_like(x_coords), np.zeros_like(x_coords)
    
    yc = np.zeros_like(x_coords)
    dyc_dx = np.zeros_like(x_coords)
    
    # Differentiate calculations for the forward and aft sections of the camber line.
    front_mask = x_coords <= p
    rear_mask = x_coords > p
    
    yc[front_mask] = (m / p**2) * (2 * p * x_coords[front_mask] - x_coords[front_mask]**2)
    yc[rear_mask] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x_coords[rear_mask] - x_coords[rear_mask]**2)
    
    dyc_dx[front_mask] = (2 * m / p**2) * (p - x_coords[front_mask])
    dyc_dx[rear_mask] = (2 * m / (1 - p)**2) * (p - x_coords[rear_mask])

    return yc, dyc_dx

def calculate_thickness_distribution(t: float, x_coords: np.ndarray) -> np.ndarray:
    """
    Calculates the thickness distribution (yt) along the chord line.

    Args:
        t (float): Maximum thickness.
        x_coords (np.ndarray): Array of x-coordinates.

    Returns:
        np.ndarray: The thickness distribution (yt).
    """
    # Coefficients for the NACA 4-digit thickness polynomial.
    a0 =  0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 =  0.2843
    a4 = -0.1015 # Standard coefficient for a physically closed trailing edge.
    
    yt = 5 * t * (
        a0 * np.sqrt(x_coords)
        + a1 * x_coords
        + a2 * x_coords**2
        + a3 * x_coords**3
        + a4 * x_coords**4
    )
    
    # Manually enforce a closed trailing edge to handle floating-point inaccuracies.
    yt[-1] = 0.0
    
    return yt

def apply_thickness(x_coords: np.ndarray, yc: np.ndarray, yt: np.ndarray, dyc_dx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the final upper and lower surface coordinates by applying the
    thickness distribution perpendicular to the camber line.

    Args:
        x_coords (np.ndarray): Array of x-coordinates.
        yc (np.ndarray): y-coordinates of the camber line.
        yt (np.ndarray): Thickness distribution.
        dyc_dx (np.ndarray): Gradient of the camber line.

    Returns:
        A tuple containing the coordinates for the upper and lower surfaces:
        (x_upper, y_upper, x_lower, y_lower)
    """
    theta = np.arctan(dyc_dx)
    x_upper = x_coords - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    x_lower = x_coords + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)
    
    return x_upper, y_upper, x_lower, y_lower

if __name__ == "__main__":
    
    # --- Configuration ---
    naca_code = "2412"
    num_points = 201
    
    # --- Generation ---
    m, p, t = parse_naca_code(naca_code)
    x_coords = create_cosine_spacing(num_points)
    yc, dyc_dx = calculate_camber_line(m, p, x_coords)
    yt = calculate_thickness_distribution(t, x_coords)
    x_upper, y_upper, x_lower, y_lower = apply_thickness(x_coords, yc, yt, dyc_dx)
    
    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(x_upper, y_upper, 'b-', label='Upper Surface')
    plt.plot(x_lower, y_lower, 'r-', label='Lower Surface')
    plt.plot(x_coords, yc, 'k--', label='Camber Line')
    plt.title(f'NACA {naca_code} Aerofoil', fontsize=16)
    plt.xlabel('Chord')
    plt.ylabel('Thickness')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()
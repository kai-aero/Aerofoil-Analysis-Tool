# -*- coding: utf-8 -*-
"""
Main script to run the NACA aerofoil flow simulation.

This script orchestrates the entire process:
1.  Sets the configuration for the simulation.
2.  Generates the aerofoil geometry using the `aerofoil_generator` module.
3.  Calculates the flow properties using the `panel_method_solver` module.
4.  Visualises the final airflow streamlines over the aerofoil.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from aerofoil_generator import create_cosine_spacing, calculate_camber_line, calculate_thickness_distribution, apply_thickness, parse_naca_code
from panel_method_solver import solve_panel_method, get_velocities

def main():
    """Main function to run the simulation and plot the results."""
    # --- 1. Configuration ---
    naca_code = "2412"
    num_points = 101       # Number of points defining the aerofoil shape
    N = num_points - 1     # Number of panels is one less than the number of points
    alpha_deg = 8.0        # Angle of attack in degrees
    Uinf = 1.0             # Freestream velocity

    # --- 2. Generate Aerofoil Geometry ---
    print("Generating aerofoil coordinates...")
    alpha_rad = np.deg2rad(alpha_deg)
    
    m, p, t = parse_naca_code(naca_code)
    x_coords_raw = create_cosine_spacing(num_points)
    yc, dyc_dx = calculate_camber_line(m, p, x_coords_raw)
    yt = calculate_thickness_distribution(t, x_coords_raw)
    x_upper, y_upper, x_lower, y_lower = apply_thickness(x_coords_raw, yc, yt, dyc_dx)
    
    # The solver needs a single clockwise array of points
    x_nodes = np.concatenate((np.flip(x_upper), x_lower[1:]))
    y_nodes = np.concatenate((np.flip(y_upper), y_lower[1:]))
    print("Aerofoil generation complete.")

    # --- 3. Run the Panel Method Solver ---
    print("Running panel method solver...")
    q, g = solve_panel_method(x_nodes, y_nodes, N, Uinf, alpha_rad)
    print("Solver finished successfully.")

    # --- 4. Visualise the Results ---
    print("Generating flow visualisation...")
    # Create a grid of points around the aerofoil
    x_start, x_end = -0.5, 1.5
    y_start, y_end = -0.5, 0.5
    grid_size = 100
    X, Y = np.meshgrid(np.linspace(x_start, x_end, grid_size), np.linspace(y_start, y_end, grid_size))

    # Calculate the velocity at each grid point
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            u[i, j], v[i, j] = get_velocities(X[i, j], Y[i, j], x_nodes, y_nodes, N, q, g, Uinf, alpha_rad)
    
    # Create a path from the aerofoil coordinates to mask the interior
    aerofoil_path = Path(np.vstack((x_nodes, y_nodes)).T)
    mask = aerofoil_path.contains_points(np.vstack([X.ravel(), Y.ravel()]).T).reshape(X.shape)
    u[mask] = 0
    v[mask] = 0

    # --- 5. Plot the Final Result ---
    plt.figure(figsize=(12, 6))
    # The 'color' keyword is part of the Matplotlib library and must be spelled this way.
    plt.streamplot(X, Y, u, v, density=2, linewidth=0.8, color='k')
    plt.fill(x_nodes, y_nodes, 'k')
    plt.title(f'Flow around NACA {naca_code} at {alpha_deg}° Angle of Attack', fontsize=16)
    plt.xlabel('Chord', fontsize=12)
    plt.ylabel('Thickness', fontsize=12)
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
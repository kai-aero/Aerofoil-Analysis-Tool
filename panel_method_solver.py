# -*- coding: utf-8 -*-
"""
Core functions for a 2D Hess-Smith panel method solver.

This module provides the low-level functions required to calculate the influence
matrix, solve for source and circulation strengths, and determine velocities in
the flow field around an aerofoil.

Original code by Jonny de Cruz, University of Southampton.
Refactored for modular use.
"""
import numpy as np

def solve_panel_method(x_nodes: np.ndarray, y_nodes: np.ndarray, N: int, Uinf: float, alpha_rad: float) -> tuple[np.ndarray, float]:
    """
    Solves the Hess-Smith panel method for a given aerofoil geometry.

    Args:
        x_nodes (np.ndarray): Array of x-coordinates for the aerofoil nodes (N+1 points).
        y_nodes (np.ndarray): Array of y-coordinates for the aerofoil nodes (N+1 points).
        N (int): The number of panels.
        Uinf (float): The freestream velocity.
        alpha_rad (float): The angle of attack in radians.

    Returns:
        A tuple containing:
        - q (np.ndarray): The source strength for each of the N panels.
        - g (float): The circulation for the aerofoil.
    """
    b = _calculate_b_vector(x_nodes, y_nodes, N, Uinf, alpha_rad)
    A = _infl_coeffs(x_nodes, y_nodes, N)
    
    qg = np.linalg.solve(A, b)
    
    q = qg[:-1]
    g = qg[-1]
    
    return q, g

def _panel_params(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, tuple[float, float]]:
    """Computes the length, angle, and midpoint of a single panel."""
    h = np.sqrt(((x1 - x0)**2) + ((y1 - y0)**2))
    th = np.arctan2(y1 - y0, x1 - x0)
    c = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    return h, th, c

def _vel_coeffs(xp: float, yp: float, x_nodes: np.ndarray, y_nodes: np.ndarray, j: int) -> tuple[float, float]:
    """Computes the velocity coefficients ln_ij and beta_ij."""
    dx0 = xp - x_nodes[j]
    dy0 = yp - y_nodes[j]
    dx1 = xp - x_nodes[j+1]
    dy1 = yp - y_nodes[j+1]
    
    ln_ij = 0.5 * np.log((dx1**2 + dy1**2) / (dx0**2 + dy0**2))
    beta_ij = np.arctan2((dx0 * dy1) - (dy0 * dx1), (dx0 * dx1) + (dy0 * dy1))
    
    return ln_ij, beta_ij

def _infl_coeffs(x_nodes: np.ndarray, y_nodes: np.ndarray, N: int) -> np.ndarray:
    """Computes the influence coefficients matrix A."""
    A = np.zeros((N + 1, N + 1))
    
    for i in range(N):
        (hi, thi, ci) = _panel_params(x_nodes[i], y_nodes[i], x_nodes[i+1], y_nodes[i+1])
        
        for j in range(N):
            if i == j:
                ln_ij, beta_ij = 0.0, np.pi
            else:
                (ln_ij, beta_ij) = _vel_coeffs(ci[0], ci[1], x_nodes, y_nodes, j)
            
            (hj, thj, cj) = _panel_params(x_nodes[j], y_nodes[j], x_nodes[j+1], y_nodes[j+1])
            
            A[i, j] = (ln_ij * np.sin(thi - thj)) + (beta_ij * np.cos(thi - thj))
            A[i, N] += (ln_ij * np.cos(thi - thj)) - (beta_ij * np.sin(thi - thj))
            
            if i == 0 or i == N - 1:
                A[N, j] += (beta_ij * np.sin(thi - thj)) - (ln_ij * np.cos(thi - thj))
                A[N, N] += (beta_ij * np.cos(thi - thj)) + (ln_ij * np.sin(thi - thj))
                
    return A

def _calculate_b_vector(x_nodes: np.ndarray, y_nodes: np.ndarray, N: int, Uinf: float, alpha_rad: float) -> np.ndarray:
    """Calculates the right-hand side vector b of the matrix equation."""
    b = np.zeros(N + 1)
    for i in range(N):
        (hi, thi, ci) = _panel_params(x_nodes[i], y_nodes[i], x_nodes[i+1], y_nodes[i+1])
        b[i] = 2 * np.pi * Uinf * np.sin(thi - alpha_rad)
    
    (h0, th0, c0) = _panel_params(x_nodes[0], y_nodes[0], x_nodes[1], y_nodes[1])
    (h1N, th1N, c1N) = _panel_params(x_nodes[N-1], y_nodes[N-1], x_nodes[N], y_nodes[N])
    b[N] = -2 * np.pi * Uinf * (np.cos(th0 - alpha_rad) + np.cos(th1N - alpha_rad))
    
    return b

def get_velocities(xp: float, yp: float, x_nodes: np.ndarray, y_nodes: np.ndarray, N: int, q: np.ndarray, g: float, Uinf: float, alpha_rad: float) -> tuple[float, float]:
    """
    Computes the velocity components at a single point (xp, yp) in the flow field.

    Args:
        xp (float): x-coordinate of the point of interest.
        yp (float): y-coordinate of the point of interest.
        x_nodes (np.ndarray): Array of x-coordinates for the aerofoil nodes.
        y_nodes (np.ndarray): Array of y-coordinates for the aerofoil nodes.
        N (int): The number of panels.
        q (np.ndarray): Array of source strengths for each panel.
        g (float): The circulation of the aerofoil.
        Uinf (float): The freestream velocity.
        alpha_rad (float): The angle of attack in radians.

    Returns:
        A tuple containing the velocity components (u, v).
    """
    u_inf = Uinf * np.cos(alpha_rad)
    v_inf = Uinf * np.sin(alpha_rad)
    
    u_induced, v_induced = 0, 0

    for j in range(N):
        th_j = _panel_params(x_nodes[j], y_nodes[j], x_nodes[j+1], y_nodes[j+1])[1]
        (ln_pj, beta_pj) = _vel_coeffs(xp, yp, x_nodes, y_nodes, j)
        
        # Contribution from sources and vortices
        u_induced += (q[j] / (2 * np.pi)) * ((-ln_pj * np.cos(th_j)) - (beta_pj * np.sin(th_j)))
        u_induced += (g / (2 * np.pi)) * ((beta_pj * np.cos(th_j)) - (ln_pj * np.sin(th_j)))
        
        v_induced += (q[j] / (2 * np.pi)) * ((-ln_pj * np.sin(th_j)) + (beta_pj * np.cos(th_j)))
        v_induced += (g / (2 * np.pi)) * ((beta_pj * np.sin(th_j)) + (ln_pj * np.cos(th_j)))

    u = u_inf + u_induced
    v = v_inf + v_induced
    
    return u, v
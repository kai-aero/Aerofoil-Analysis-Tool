# Python Aerofoil Analysis Suite

A tool to programmatically generate and analyse the geometry of 4-digit NACA aerofoils. This project applies foundational aerodynamic theory to create a practical, visual tool using Python.

## Current Development

* **Geometry Engine (✅ Implemented):** The core functionality to generate precise coordinates and plot the shape for any NACA 4-digit aerofoil is complete.
* **Flow Solver (▶️ Next Step):** The next phase is to build a panel method solver to simulate and visualise potential flow over these aerofoils.

## NACA Aerofoil Generator

The `aerofoil_generator.py` script can generate the coordinates for a given 4-digit NACA code and plot its shape, camber line, and thickness distribution.

### How to Run
1.  Clone the repository and install the required libraries (`numpy`, `matplotlib`).
2.  Run the script from the command line:
    ```bash
    python aerofoil_generator.py
    ```
# Python Aerofoil Analysis Suite

A tool to generate and analyse the geometry and potential flow around 4-digit NACA aerofoils. This project applies foundational aerodynamic theory to create a practical, visual tool using Python.

## Project Status

This project was built in two phases:

* **Phase 1: Geometry Engine:** A script that generates and plots the precise geometry for any 4-digit NACA aerofoil.
* **Phase 2: Flow Solver:** Implementation of a panel method to simulate and visualise potential flow over the generated aerofoils.

## How to Run the Simulation

The main script `run_simulation.py` generates the aerofoil geometry and runs the flow solver.

1.  Clone the repository and install the required libraries (`numpy`, `matplotlib`).
2.  Run the main simulation script from the command line:
    ```bash
    python run_simulation.py
    ```
3.  The script will generate and display a plot of the streamlines for the default NACA 2412 aerofoil at an 8-degree angle of attack. You can change these parameters inside the script.

## Acknowledgements

The core panel method solver in `panel_method_solver.py` is based on the excellent work of **Jonny de Cruz** from the University of Southampton. His original, self-contained script was refactored for use in this project.

* **Original Repository:** [Hess-Smith-Panel-Method](https://github.com/jonny-dc/Hess-Smith-Panel-Method)
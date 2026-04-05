# Taichi Physical Simulation

A personal gallery of physics simulations implemented in Taichi.

This repository documents my process of building physical simulation systems from scratch. It starts from simple models like mass-spring systems, and gradually explores more advanced topics such as constraints, contact, and friction.

Each module is designed to be small, self-contained, and visualized with demos.

## 🎬 Demo
<p align="center">
  <img src="4_moving_dirichlet/moving_dirichlet.gif" width="300"/>
  <img src="3_friction/miu_0.01_friction.gif" width="300"/>
</p>

<p align="center">
  <em>Moving Dirichlet (left) vs Low Friction μ=0.01 (right)</em>
</p>

## 📂 Structure

- `0_mass_spring/`  
  Explicit vs implicit integration and stability comparison

- `1_dirichlet/`  
  Dirichlet boundary conditions (sticky constraints)

- `2_contact/`  
  Contact handling using barrier methods

- `3_friction/`  
  Friction behavior under different coefficients

- `4_moving_dirichlet/`  
  Moving boundary constraints with soft penalty formulation

- `implementation_notes/`  
  Implementation documents for code designing


## 🚀 Future Work

- Continuum mechanics (strain / stress, FEM)
- SPH (Smoothed Particle Hydrodynamics)
- MPM (Material Point Method)
- More advanced solvers and optimization methods

## 📝 Notes

This is an ongoing project. Some parts are experimental and may change over time.
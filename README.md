# RKF45
This program performs the following tasks: 
1. Uses scipy's library function called solve_ivp with the RK45 algorithm.
2. Implements the RKF45 algorithm (not from the library).
3. Compares the results of the two algorithms.

Dependencies:
- math
- numpy
- matplotlib.pyplot
- scipy.integrate.solve_ivp
- plotly.graph_objects
- json

Usage:
- Ensure all dependencies are installed.
- Run the script.

Functions:
- dynamics(t, x): Calculates the derivative of the state variable x with respect to time t.
- simulation_solve_ivp(dynamics, t0, tf, initial_conditions): Simulates the system using solve_ivp.
- rk4(dynamics, t, y, h): Performs a fourth-order Runge-Kutta integration step.
- rk5(dynamics, t, y, h): Performs a fifth-order Runge-Kutta integration step.
- rkf45(dynamics, t_span, y0, tol): Implements the Runge-Kutta-Fehlberg 45 algorithm.
- simulation(dynamics, t_span, initial_conditions, tol=0.5): Simulates the system using RKF45.
- interpolation_rkf45(t_values,y_values,t_interp): Interpolates RKF45 solutions to match solve_ivp.

Current pylint evaluation: 8.31

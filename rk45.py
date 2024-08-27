import math
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

with open('params_rk45.json', 'r') as file:
    data = json.load(file)

num = []

# Accessing each number in k1, k2, k3, ...
for key in data.keys():
    key_numbers = []  # List to store numbers for the current key
    for value in data[key]:
        numbers = value.split('/')  # Split the string by '/'
        if len(numbers) == 1:  # If it's a single number
            key_numbers.append(float(numbers[0]))  # Append the single number as a float
        else:  # If it's a fraction
            numerator_str, denominator_str = numbers
            numerator = float(numerator_str)
            denominator = float(denominator_str)
            key_numbers.append(numerator / denominator)  # Append the fraction as a float
    num.append(key_numbers)  # Append the list of numbers for the current key to the 2D array

initial_conditions = np.linspace(2.1, 2.3, 50)
t_span = (0, 50)
t0, tf = t_span
def dynamics(t, x):
    """
    Calculates the derivative of the state variable x with respect to time t.
    Parameters:
    - t (float): The current time.
    - x (float): The current state variable.
    Returns:
    - dxdt (float): The derivative of x with respect to t.
    """
    n = 4
    k = 30
    kdeg = 0.2
    dxdt = x**n / (x**n + k) - kdeg*x
    return dxdt

def simulation_solve_ivp(dynamics, t0, tf, initial_conditions):
    """
    Simulates the system using solve_ivp (RK45 algorithm from scipy).
    Parameters:
    - dynamics (function): The dynamics function describing the system.
    - t0 (float): The initial time.
    - tf (float): The final time.
    - initial_conditions (array-like): The array of initial conditions.
    Returns:
    - values (list): A list containing the results of the simulations.
    """
    values = []
    for y0 in initial_conditions:
        y_array = np.array([y0])
        y_result = solve_ivp(dynamics, (t0, tf), y_array, dense_output=True)
        values.append(y_result)
    return values

t_values_ivp = np.linspace(t0, tf, 50)
x_values_ivp = []
solution_ivp = simulation_solve_ivp(dynamics, t0, tf, initial_conditions)
for y_ivp in solution_ivp:
    x_values_ivp.append(y_ivp.sol(t_values_ivp).flatten())

# Heatmap for solve_ivp
data_ivp = np.array(x_values_ivp)
heatmap_trace_ivp = go.Heatmap(z=data_ivp,x=t_values_ivp,y=initial_conditions)
layout_ivp = go.Layout(
    title='Heatmap of solve_ivp',
    xaxis={'title': 'Time'},
    yaxis={'title': 'Initial State x'},
    width=800,
    height=600
)
fig_solve_ivp = go.Figure(data=[heatmap_trace_ivp], layout=layout_ivp)

# Graph for solve_ivp
plt.figure(figsize=(10, 6))
for x_ivp in x_values_ivp:
    plt.plot(t_values_ivp, x_ivp)
plt.xlabel('Time')
plt.ylabel('State x')
plt.title('Solve_ivp')
plt.legend()
plt.grid(True)
plt.show()

# RKF45

def optimal_step(h, tol, z, y):
    """
    Calculates the optimal step size for the RKF45 integration method.
    Parameters:
    - h (float): The current step size.
    - tol (float): The tolerance for the integration method.
    - z (float): Estimate of the state variable using a higher-order method.
    - y (float): Estimate of the state variable using a lower-order method.
    Returns:
    - float: The optimal step size for the next iteration.
    """
    s = ((tol * h)/(2*abs(z-y)))**0.25
    return s * h

def k_steps(dynamics, ts, ys, hs):
    """
    Performs intermediate steps for the Runge-Kutta integration.
    Parameters:
    - dynamics (function): The dynamics function describing the system.
    - ts (float): The current time.
    - ys (float): The current state variable.
    - hs (float): The step size.
    Returns:
    - k1_s, k3_s, k4_s, k5_s, k6_s (tuple): Intermediate steps for the RK4 and RK5 methods.
    """
    k1_s=hs*dynamics(ts,ys)
    k2_s=hs*dynamics(ts,ys+num[1][1]*k1_s)
    k3_s=hs*dynamics(ts,ys+num[2][1]*k1_s+num[2][2]*k2_s)
    k4_s=hs*dynamics(ts,ys+num[3][1]*k1_s+num[3][2]*k2_s+num[3][3]*k3_s)
    k5_s=hs*dynamics(ts,ys+num[4][1]*k1_s+num[4][2]*k2_s+num[4][3]*k3_s+num[4][4]*k4_s)
    k6_s=hs*dynamics(ts,ys+num[5][1]*k1_s+num[5][2]*k2_s+num[5][3]*k3_s+num[5][4]*k4_s+num[5][5]*k5_s)
    return k1_s, k3_s, k4_s, k5_s, k6_s

def rk4(dynamics, t, y, h):
    """
    Performs a fourth-order Runge-Kutta integration step.
    Parameters:
    - dynamics (function): The dynamics function describing the system.
    - t (float): The current time.
    - y (float): The current state variable.
    - h (float): The step size.
    Returns:
    - y_new (float): The new state variable after the integration step.
    """
    k1_s, k3_s, k4_s, k5_s, _ = k_steps(dynamics, t, y, h)
    return y+num[6][0]*k1_s+num[6][2]*k3_s+num[6][3]*k4_s+num[6][4]*k5_s

def rk5(dynamics, t, y, h):
    """
    Performs a fifth-order Runge-Kutta integration step.
    Parameters:
    - dynamics (function): The dynamics function describing the system.
    - t (float): The current time.
    - y (float): The current state variable.
    - h (float): The step size.
    Returns:
    - y_new (float): The new state variable after the integration step.
    """
    k1_s, k3_s, k4_s, k5_s, k6_s = k_steps(dynamics, t, y, h)
    return y+num[7][0]*k1_s+num[7][2]*k3_s+num[7][3]*k4_s+num[7][4]*k5_s+num[7][5]*k6_s

def rkf45(dynamics, t_span, y0, tol):
    """
    Performs the Runge-Kutta-Fehlberg integration method (RKF45).
    Parameters:
    - dynamics (function): The dynamics function describing the system.
    - t_span (tuple): The time span for integration (t0, tf).
    - y0 (float): The initial state variable.
    - tol (float): The tolerance for the integration method.
    Returns:
    - t_values (array-like): Array containing the time values.
    - y_values (array-like): Array containing the state variable values.
    """
    t0_step, tf_step = t_span
    t_values_step = [t0_step]
    y_values_step = [y0]
    t_step = t0_step
    y_step = y0
    h_step = 0.2  # Initial step size, normally it should be h = (tf - t0) / 100
    while t_step < tf_step:
        yn_step = rk4(dynamics, t_step, y_step, h_step)
        zn_step = rk5(dynamics, t_step, y_step, h_step)

        if np.max(np.abs(zn_step - yn_step)) < tol:
            if t_step + h_step >= tf_step:
                t_step = tf_step
            else:
                t_step += h_step
            t_values_step.append(t_step)
            y_step = yn_step
            y_values_step.append(y_step)
        h_step = optimal_step(tol, h_step, zn_step, yn_step)
    return np.array(t_values_step), np.array(y_values_step)

def simulation(dynamics, t_span, initial_conditions, tol=0.5):
    """
    Performs simulation using the RKF45 integration method.
    Parameters:
    - dynamics (function): The dynamics function describing the system.
    - t_span (tuple): The time span for integration (t0, tf).
    - initial_conditions (array-like): The array of initial conditions.
    - tol (float, optional): The tolerance for the integration method. Default is 0.5.
    Returns:
    - t_values (array-like): Array containing the time values.
    - y_values (array-like): Array containing the state variable values.
    """
    t_values = []
    y_values = []
    for y0 in initial_conditions:
        t, y = rkf45(dynamics, t_span, y0, tol)
        t_values.append(t)
        y_values.append(y)
    return t_values, y_values
t_values_step, y_values_step = simulation(dynamics, t_span, initial_conditions)

#Heatmap
heatmap_trace = go.Heatmap(z=y_values_step, x=t_values_step[0], y=initial_conditions)
layout = go.Layout(
    title='Heatmap of RKF45',
    xaxis={'title': 'Time', 'range': [t0, tf]},
    yaxis={'title': 'Initial state X'},
    width=800,
    height=600
)
fig_RKF45 = go.Figure(data=[heatmap_trace], layout=layout)

#Graph
plt.figure(figsize=(10, 6))
for t_step, y_step in zip(t_values_step, y_values_step):
    plt.plot(t_step, y_step)
plt.xlabel('Time')
plt.ylabel('State x')
plt.title('RKF45')
plt.grid(True)
plt.show()


t_interp_step = np.linspace(t0, tf, 50)
#The difference between RKF45 and solve_ivp:
def interpolation_rkf45(t_values_step, y_values_step, t_interp_step):
    """
    Interpolates RKF45 simulation results to match a specific set of time points.
    Parameters:
    - t_values_step (array-like): Array containing the original time values.
    - y_values_step (array-like): Array containing the original state variable values.
    - t_interp_step (array-like): Array containing the time points for interpolation.
    Returns:
    - interp_y_step (array-like): Interpolated state variable values.
    """
    interp_y_step = []
    for t_step, y_step in zip(t_values_step, y_values_step):
        #From numpy library interp function
        #https://numpy.org/doc/stable/reference/generated/numpy.interp.html
        interp_y_step.append(np.interp(t_interp_step, t_step, y_step))
    return interp_y_step


y_values_interp_step = interpolation_rkf45(t_values_step, y_values_step, t_interp_step)

#Difference from logorithmic function for the solution
difference = []
tmp = []
MIN_VALUE = 0
for y_step, x_ivp in zip(y_values_interp_step, x_values_ivp):
    for y_point_step, x_point_ivp in zip(y_step, x_ivp):
        tmp.append(math.log(abs(y_point_step - x_point_ivp) + 1e-6))
        if MIN_VALUE>min(tmp):
            MIN_VALUE = min(tmp)
    difference.append(tmp)
    tmp = []

#Heatmap
heatmap_trace = go.Heatmap(z=difference, x=t_interp_step)

layout = go.Layout(
    title='Heatmap of log(abs(y - x) + 1e-6)',
    xaxis={'title': 'Time', 'range': [t0, tf]},
    yaxis={'title': 'Functions count'},
    width=800,
    height=600
)

fig_difference = go.Figure(data=[heatmap_trace], layout=layout)

#Graph
t_values_step = np.linspace(t0, tf, 50)
plt.figure(figsize=(10, 6))
#zip - pairs the values
for y_step in difference:
    plt.plot(t_values_step, y_step)
plt.xlabel('Time')
plt.ylabel('State x')
plt.title('log(abs(y - x) + 1e-6) graph')
plt.grid(True)
plt.show()


plt.show()

fig_solve_ivp.show()
fig_RKF45.show()
fig_difference.show()

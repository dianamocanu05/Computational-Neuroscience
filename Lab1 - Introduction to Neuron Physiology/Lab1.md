# Laboratory 1: Introduction to Neuron Physiology

## Exercise 1: Simple Neuron Modeling

**Objective:** Implement a mathematical model of a neuron and visualize its membrane potential over time.

**Exercise:**

1. Use the integrate-and-fire model to simulate the membrane potential of a neuron.
2. Plot the membrane potential against time using Matplotlib.

**Hints:**

- Start with a basic integrate-and-fire equation.
- Define constants such as capacitance, leak conductance, resting potential, and input current.
- Implement a simple threshold mechanism for action potentials.

**Code Snippet:**

```py
import numpy as np
import matplotlib.pyplot as plt

# Constants
C = 1.0  # Capacitance
g_L = 0.1  # Leak conductance
E_L = -70.0  # Resting potential
I = 1.5  # Input current

# Simulation parameters
dt = 0.1  # Time step
timesteps = 1000

# Initialize variables
membrane_potential = np.zeros(timesteps)

# Simulation loop
for t in range(1, timesteps):
    membrane_potential[t] = membrane_potential[t-1] + (1/C) * (I - g_L * (membrane_potential[t-1] - E_L)) * dt
    # Add threshold mechanism for simplicity
    if membrane_potential[t] > -55.0:
        membrane_potential[t] = E_L

# Plotting
time = np.arange(0, timesteps * dt, dt)
plt.plot(time, membrane_potential)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Simple Neuron Model')
plt.show()
```

## Exercise 2: Hodgkin-Huxley Model Implementation

**Objective:** Understand the dynamics of an action potential using the Hodgkin-Huxley model.

**Exercise:**

1. Implement the Hodgkin-Huxley equations in Python.
2. Simulate and visualize the generation of an action potential.
3. Explore the impact of changing parameters on the action potential shape.

**Hints:**

- Use the Hodgkin-Huxley model equations for gating variables *m*, *h*, and *n*.
- Simulate the model over time to observe the generation of an action potential.
- Experiment with different input currents.

**Code Snippet:**

```py
import numpy as np
import matplotlib.pyplot as plt

# Constants
C_m = 1.0  # Membrane capacitance
g_Na = 120.0  # Sodium conductance
g_K = 36.0  # Potassium conductance
g_L = 0.3  # Leak conductance
E_Na = 55.0  # Sodium reversal potential
E_K = -72.0  # Potassium reversal potential
E_L = -49.4  # Leak reversal potential

# Simulation parameters
dt = 0.01  # Time step
timesteps = 5000

# Initialize variables
membrane_potential = np.zeros(timesteps)
m = np.zeros(timesteps)
h = np.zeros(timesteps)
n = np.zeros(timesteps)

# Initial conditions
membrane_potential[0] = -65.0
m[0] = 0.05
h[0] = 0.6
n[0] = 0.3

# Input current
I = np.concatenate([np.zeros(1000), np.ones(2000)])

# Hodgkin-Huxley model simulation
for t in range(1, timesteps):
    alpha_m = (0.1 * (membrane_potential[t-1] + 40.0)) / (1.0 - np.exp(-(membrane_potential[t-1] + 40.0) / 10.0))
    beta_m = 4.0 * np.exp(-(membrane_potential[t-1] + 65.0) / 18.0)
    alpha_h = 0.07 * np.exp(-(membrane_potential[t-1] + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(membrane_potential[t-1] + 35.0) / 10.0))
    alpha_n = (0.01 * (membrane_potential[t-1] + 55.0)) / (1.0 - np.exp(-(membrane_potential[t-1] + 55.0) / 10.0))
    beta_n = 0.125 * np.exp(-(membrane_potential[t-1] + 65.0) / 80.0)

    m[t] = m[t-1] + (alpha_m * (1 - m[t-1]) - beta_m * m[t-1]) * dt
    h[t] = h[t-1] + (alpha_h * (1 - h[t-1]) - beta_h * h[t-1]) * dt
    n[t] = n[t-1] + (alpha_n * (1 - n[t-1]) - beta_n * n[t-1]) * dt

    I_Na = g_Na * m[t]**3 * h[t] * (membrane_potential[t-1] - E_Na)
    I_K = g_K * n[t]**4 * (membrane_potential[t-1] - E_K)
    I_L = g_L * (membrane_potential[t-1] - E_L)

    membrane_potential[t] = membrane_potential[t-1] + (I[t-1] - I_Na - I_K - I_L) / C_m * dt

# Plotting
time = np.arange(0, timesteps * dt, dt)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(time, membrane_potential, label='Membrane Potential')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Hodgkin-Huxley Model: Action Potential')

plt.subplot(2, 1, 2)
plt.plot(time, m, label='m')
plt.plot(time, h, label='h')
plt.plot(time, n, label='n')
plt.xlabel('Time (ms)')
plt.ylabel('Gating Variables')
plt.legend()
plt.show()
```

## Exercise 3: Basic Network Simulation

**Objective:** Extend the model to a simple network and observe interactions.

**Exercise:**

1. Create a network of integrate-and-fire neurons with excitatory and inhibitory connections.
2. Simulate the network activity over time.
3. Visualize the spike patterns and analyze how the network responds to different input configurations.

**Hints:**

- Build on the integrate-and-fire model from Exercise 1.
- Include synaptic interactions in the network.
- Visualize the network activity using raster plots or spike histograms.

## Exercise 4: Data Visualization and Analysis

**Objective:** Analyze and interpret neural data.

**Exercise:**

1. Generate synthetic spike train data for multiple neurons.
2. Use Python libraries such as Seaborn or Plotly to create raster plots and spike histograms.
3. Perform basic statistical analysis on the spike train data.

**Hints:**

- Simulate spike trains for multiple neurons using Poisson processes.
- Explore Seaborn or Plotly for data visualization.
- Calculate firing rates, interspike intervals, and coefficient of variation.

## Exercise 5: Explore Different Input Currents

**Objective:** Investigate the effects of various input currents on neuronal behavior.

**Exercise:**

1. Apply step, ramp, and sinusoidal input currents to the neuron model.
2. Record and compare the resulting membrane potential dynamics.
3. Discuss how different input patterns influence the neuron's firing behavior.

**Hints:**

- Design different input current waveforms.
- Use the neuron model from Exercise 1 to observe responses.
- Analyze and compare the impact of different stimuli on the membrane potential.
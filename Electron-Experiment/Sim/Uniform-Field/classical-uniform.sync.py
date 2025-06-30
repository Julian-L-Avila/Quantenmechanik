# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="06549788"
# # Classical Uniform Electron Beam Simulation
#
# This notebook simulates the trajectory of electrons in a time-varying magnetic field and visualizes the pattern formed on a screen. It is based on the `classical-uniform.py` script.

# %% [markdown] id="ad7d98ca"
# ## 1. Imports
#
# Import necessary libraries: `numpy` for numerical operations, `scipy` for solving ordinary differential equations, and `matplotlib` for plotting.

# %% id="9e1ad508"
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import imageio
import os

# %% [markdown] id="a15117ae"
# ## 2. Physical Constants and Simulation Parameters
#
# Define the physical constants (in Gaussian units) and the parameters for the simulation.

# %% colab={"base_uri": "https://localhost:8080/"} id="8361b693" outputId="4b944eb0-0a01-4878-fc01-e85284f71195"
# Physical constants (Gaussian units)
m_e = 9.10938356e-28  # Mass of electron (g)
e_charge = 4.80320425e-10 # Charge of electron (statcoulomb or esu)
c_light = 2.99792458e10   # Speed of light (cm/s)
acc_potential = 4.3e3 # Accelerating Potential (kV)

enegy_kin = acc_potential / 299.79
v_initial = np.sqrt(2 * enegy_kin * e_charge / m_e)
print("Initial v: ", v_initial / c_light, "c")

# Simulation parameters
# CRITICAL FIX: Reduced B-field strength to prevent non-physical velocities.
# The original value of 10.0 G caused numerical instability.
B_02_sim = 0.1   # Gauss
B_03_sim = 0.1   # Gauss
w_2_sim = 61.0 * 2.0 * np.pi    # rad/s
w_3_sim = 63.0 * 2.0 * np.pi    # rad/s
L_sim = 20.0      # cm (length of the beam path to the screen)
v_initial_sim = v_initial # Initial velocity in x1 direction

num_electrons_sim = 1000 # Number of electrons to simulate

# --- Output Control ---
generate_gif = True # Set to False to generate a static plot instead of a GIF


# %% [markdown] id="e3c5bcee"
# ## 3. B-Field Definitions
#
# Functions to define the time-varying magnetic fields $B_2(t)$ and $B_3(t)$ and their time derivatives.

# %% id="a4f32a9e"
def B_sin(t, B_0=1.0, omega=1.0, phase=0.0):
    return B_0 * np.sin(omega * t + phase)

def dB_sin(t, B_0=1.0, omega=1.0, phase=0.0):
    return B_0 * omega * np.cos(omega * t + phase)

def B_square(t, B_0=1.0, omega=1.0, phase=0.0, steepness=100):
    return B_0 * np.tanh(steepness * np.sin(omega * t + phase))

def dB_square(t, B_0=1.0, omega=1.0, phase=0.0, steepness=100):
    sin_val = np.sin(omega * t + phase)
    cos_val = np.cos(omega * t + phase)
    sech_sq_val = 1 - np.tanh(steepness * sin_val)**2
    return B_0 * (steepness * omega * cos_val) * sech_sq_val

def B_trig(t, B_0=1.0, omega=1.0):
    period = 2.0 * np.pi / omega
    slope = 2.0 * B_0 * omega / np.pi

    time_in_period = t % period

    if time_in_period < period / 2.0:
        return - B_0 + slope * (time_in_period)

    else:
        return 3 * B_0 - slope * (time_in_period)

def dB_trig(t, B_0=1.0, omega=1.0):
    period = 2.0 * np.pi / omega
    slope = 2.0 * B_0 * omega / np.pi

    time_in_period = t % period

    if time_in_period < period / 2.0:
        return slope

    else:
        return - slope

# %%
def B2_field(t, B_02_param, w_2_param):
    return B_sin(t, B_02_param, w_2_param)
def dB2_dt_field(t, B_02_param, w_2_param):
    return dB_sin(t, B_02_param, w_2_param)

def B3_field(t, B_03_param, w_3_param):
    return B_sin(t, B_03_param, w_3_param)
def dB3_dt_field(t, B_03_param, w_3_param):
    return dB_sin(t, B_03_param, w_3_param)

# %% [markdown] id="894da472"
# ## 4. Equations of Motion
#
# Defines the system of ordinary differential equations (ODEs) that describe the motion of an electron in the magnetic fields.
#
# The ODE function takes `t_offset` as an argument. This offset represents the launch time of a specific electron on a global clock, allowing us to simulate electrons launched at different phases of the oscillating B-fields.

# %% id="75530bde"
# REFACTORED: The ODE function is now defined once and takes t_offset as an argument.
# This is a cleaner and more robust approach than redefining a function in each loop iteration.
def equations_of_motion(t_particle, y, m, e, c, B_02_param, w_2_param, B_03_param, w_3_param, t_offset):
    # The B-field evolves on a global clock. Its phase depends on when the electron was launched (t_offset).
    # The solver's time, t_particle, is the time since that specific electron's launch.
    global_t = t_particle + t_offset

    x1, x2, x3, v1, v2, v3 = y

    B2_t = B2_field(global_t, B_02_param, w_2_param)
    dB2_dt_t = dB2_dt_field(global_t, B_02_param, w_2_param)
    B3_t = B3_field(global_t, B_03_param, w_3_param)
    dB3_dt_t = dB3_dt_field(global_t, B_03_param, w_3_param)

    # The equations of motion as provided in the notebook
    a1 = (e / (2 * m * c)) * (x3 * dB2_dt_t - x2 * dB3_dt_t)
    a2 = (e / (m * c)) * (0.5 * x1 * dB3_dt_t - v1 * B3_t)
    a3 = (e / (m * c)) * (-0.5 * x1 * dB2_dt_t + v1 * B2_t)

    return [v1, v2, v3, a1, a2, a3]


# %% [markdown] id="0946de0b"
# ## 5. Simulate Electron Trajectories
#
# This section simulates the trajectories of multiple electrons. Each electron is launched at a slightly different time (`t_offset`) to capture the effect of the oscillating fields. The final positions of the electrons on a screen located at $x_1 = L$ are recorded.

# %% colab={"base_uri": "https://localhost:8080/"} id="a735ce0f" outputId="52f965e4-f3c4-4db8-b60d-b3ddd7ccd570"
screen_positions_x2 = []
screen_positions_x3 = []

# Approximate time of flight for one electron. Used to set the integration interval.
t_flight_approx = L_sim / v_initial_sim
t_span = (0, t_flight_approx)

# REVISED LOGIC/COMMENT: The following clarifies how electrons are launched.
# To generate a pattern, we simulate electrons launched at different times into the
# oscillating magnetic fields. This is modeled with a 't_offset' for each electron,
# representing its launch time on a global clock. This is equivalent to applying
# a different initial phase of the B-fields for each electron.
characteristic_period = 2 * np.pi / max(w_2_sim, w_3_sim)
total_launch_time = characteristic_period * 5 # Launch electrons over 5 cycles of the faster B-field

print(f"Approximate time of flight: {t_flight_approx:.2e} s")
print(f"Simulating {num_electrons_sim} electrons...")

for i in range(num_electrons_sim):
    # This offset simulates launching electrons sequentially into the evolving B-fields.
    t_offset = (i / num_electrons_sim) * total_launch_time

    # Initial conditions: [x1, x2, x3, v1, v2, v3]
    y0 = [0, 0, 0, v_initial_sim, 0, 0]

    # Set evaluation times for dense output. Ensures smooth trajectory data.
    num_time_points = int(max(w_2_sim, w_3_sim) * t_flight_approx / (2*np.pi) * 30)
    num_time_points = max(num_time_points, 200)
    t_eval = np.linspace(t_span[0], t_span[1], num_time_points)

    # REFACTORED: Use the 'args' parameter to pass arguments to the ODE function.
    # This is the standard, efficient, and correct way to handle changing parameters in a loop.
    sol = solve_ivp(
        fun=equations_of_motion,
        t_span=t_span,
        y0=y0,
        method='RK45',
        dense_output=True, # Recommended for accurate final point interpolation
        t_eval=t_eval,
        args=(m_e, e_charge, c_light, B_02_sim, w_2_sim, B_03_sim, w_3_sim, t_offset)
    )

    # Extract the final position on the screen
    final_state = sol.y[:, -1]
    screen_positions_x2.append(final_state[1])
    screen_positions_x3.append(final_state[2])

    if (i+1) % (num_electrons_sim // 10) == 0:
        print(f"  Simulated {i+1}/{num_electrons_sim} electrons...")

print("Simulation complete.")

# %% [markdown] id="245d8154"
# ## 6. Plot the Screen Pattern
#
# Visualize the positions of the electrons on the screen. This creates a scatter plot of $(x_2, x_3)$ coordinates.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1a4365b1" outputId="e1a81646-ddcf-42c9-f0de-b87742bab2f8"
plt.figure(figsize=(9, 9))
plt.scatter(screen_positions_x2, screen_positions_x3, s=10, alpha=0.7)
plt.title(f'Electron Beam Pattern on Screen (L={L_sim} cm, $B_0$={B_02_sim} G)')
plt.xlabel('$x_2$ position (cm)')
plt.ylabel('$x_3$ position (cm)')
plt.grid(True)
plt.axis('equal')
# plt.show() # Original static plot show command, moved to conditional block

if generate_gif:
    # --- GIF Generation Parameters ---
    gif_filename = 'electron-beam-uniform-sin.gif'
    gif_path = os.path.join(os.getcwd(), gif_filename) # Save in the current notebook directory
    num_frames = num_electrons_sim # One frame per electron for simplicity, can be adjusted
    decay_factor = 0.9 # How much the alpha of a point decreases each frame it persists
    min_alpha = 0.1 # Minimum alpha before a point is removed
    fps = 10 # Frames per second for the GIF

    # --- Generate Frames for GIF ---
    writer = imageio.get_writer(gif_path, fps=fps)
    history_points = [] # Stores (x2, x3, initial_alpha, current_alpha, age)

    print(f"\nGenerating GIF frames for {gif_filename}...")

    fig_gif, ax_gif = plt.subplots(figsize=(9, 9))

    for i in range(num_electrons_sim):
        # Add the new point from this electron
        new_x2 = screen_positions_x2[i]
        new_x3 = screen_positions_x3[i]
        history_points.append([new_x2, new_x3, 1.0, 1.0, 0]) # x2, x3, initial_alpha, current_alpha, age

        ax_gif.clear()
        ax_gif.set_facecolor('black') # Set background to black for better visibility of points
        fig_gif.patch.set_facecolor('black')

        # Update and draw points
        current_frame_points_x2 = []
        current_frame_points_x3 = []
        current_frame_alphas = []

        next_history_points = []
        for point_data in history_points:
            px, py, initial_a, current_a, age = point_data
            current_frame_points_x2.append(px)
            current_frame_points_x3.append(py)
            current_frame_alphas.append(current_a)

            # Decay alpha for next frame
            next_alpha = current_a * decay_factor
            if next_alpha >= min_alpha:
                next_history_points.append([px, py, initial_a, next_alpha, age + 1])

        history_points = next_history_points

        if current_frame_points_x2: # Check if there are any points to plot
            ax_gif.scatter(current_frame_points_x2, current_frame_points_x3, s=15,
                           c=current_frame_alphas, cmap='Blues_r', vmin=0, vmax=1.0,
                           edgecolors='cyan', linewidths=0.5) # Brighter points with cyan edges

        ax_gif.set_title(f'Electron Beam Pattern (Frame {i+1}/{num_electrons_sim})', color='white')
        ax_gif.set_xlabel('$x_2$ position (cm)', color='white')
        ax_gif.set_ylabel('$x_3$ position (cm)', color='white')
        ax_gif.grid(True, color='gray', linestyle=':', linewidth=0.5)
        ax_gif.tick_params(axis='x', colors='white')
        ax_gif.tick_params(axis='y', colors='white')

        # Determine plot limits dynamically based on all points simulated so far
        if screen_positions_x2 and screen_positions_x3: # Ensure lists are not empty
            all_x2 = np.array(screen_positions_x2[:i+1])
            all_x3 = np.array(screen_positions_x3[:i+1])
            margin = 0.1 * max( (all_x2.max()-all_x2.min() if all_x2.size >1 else 1),
                                 (all_x3.max()-all_x3.min() if all_x3.size >1 else 1) )
            margin = max(margin, 0.1) # Ensure a minimum margin

            ax_gif.set_xlim(all_x2.min() - margin, all_x2.max() + margin)
            ax_gif.set_ylim(all_x3.min() - margin, all_x3.max() + margin)
        else: # Default limits if no points yet (should not happen if simulation runs correctly)
            ax_gif.set_xlim(-L_sim*0.1, L_sim*0.1)
            ax_gif.set_ylim(-L_sim*0.1, L_sim*0.1)

        ax_gif.set_aspect('equal', adjustable='box')

        fig_gif.canvas.draw()
        # Convert the canvas to an RGB numpy array
        image_rgba = np.asarray(fig_gif.canvas.buffer_rgba())
        image_rgb = image_rgba[:, :, :3] # Drop the alpha channel
        writer.append_data(image_rgb)

        if (i+1) % (num_electrons_sim // 10) == 0:
            print(f"  Generated frame {i+1}/{num_electrons_sim}...")

    writer.close()
    plt.close(fig_gif) # Close the figure used for GIF generation
    print(f"GIF saved to {gif_path}")

    # Display the GIF in the notebook (optional, may not work in all environments)
    from IPython.display import Image as IPImage
    if os.path.exists(gif_path):
        display(IPImage(filename=gif_path))
    else:
        print(f"Could not find GIF at {gif_path} to display.")
else:
    # --- Generate Static Plot ---
    print("\nGenerating static plot...")
    plt.figure(figsize=(9, 9))
    plt.scatter(screen_positions_x2, screen_positions_x3, s=10, alpha=0.7)
    plt.title(f'Electron Beam Pattern on Screen (L={L_sim} cm, $B_0$={B_02_sim} G)')
    plt.xlabel('$x_2$ position (cm)')
    plt.ylabel('$x_3$ position (cm)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    print("Static plot displayed.")


# %% [markdown] id="K0r0zpRNiQyC"
# ## Square Wave

# %% id="L8njBA8Xi32O"
def B2_field(t, B_02_param, w_2_param):
    return B_square(t, B_02_param, w_2_param)
def dB2_dt_field(t, B_02_param, w_2_param):
    return dB_square(t, B_02_param, w_2_param)

def B3_field(t, B_03_param, w_3_param):
    return B_square(t, B_03_param, w_3_param)
def dB3_dt_field(t, B_03_param, w_3_param):
    return dB_square(t, B_03_param, w_3_param)

# %% id="lYIBravnkmKG"
# REFACTORED: The ODE function is now defined once and takes t_offset as an argument.
# This is a cleaner and more robust approach than redefining a function in each loop iteration.
def equations_of_motion(t_particle, y, m, e, c, B_02_param, w_2_param, B_03_param, w_3_param, t_offset):
    # The B-field evolves on a global clock. Its phase depends on when the electron was launched (t_offset).
    # The solver's time, t_particle, is the time since that specific electron's launch.
    global_t = t_particle + t_offset

    x1, x2, x3, v1, v2, v3 = y

    B2_t = B2_field(global_t, B_02_param, w_2_param)
    dB2_dt_t = dB2_dt_field(global_t, B_02_param, w_2_param)
    B3_t = B3_field(global_t, B_03_param, w_3_param)
    dB3_dt_t = dB3_dt_field(global_t, B_03_param, w_3_param)

    # The equations of motion as provided in the notebook
    a1 = (e / (2 * m * c)) * (x3 * dB2_dt_t - x2 * dB3_dt_t)
    a2 = (e / (m * c)) * (0.5 * x1 * dB3_dt_t - v1 * B3_t)
    a3 = (e / (m * c)) * (-0.5 * x1 * dB2_dt_t + v1 * B2_t)

    return [v1, v2, v3, a1, a2, a3]


# %% outputId="09d022d6-8b13-4dc2-ae2f-a8f910517eda" colab={"base_uri": "https://localhost:8080/"} id="YQRcprunkaMc"
screen_positions_x2 = []
screen_positions_x3 = []

# Approximate time of flight for one electron. Used to set the integration interval.
t_flight_approx = L_sim / v_initial_sim
t_span = (0, t_flight_approx)

# REVISED LOGIC/COMMENT: The following clarifies how electrons are launched.
# To generate a pattern, we simulate electrons launched at different times into the
# oscillating magnetic fields. This is modeled with a 't_offset' for each electron,
# representing its launch time on a global clock. This is equivalent to applying
# a different initial phase of the B-fields for each electron.
characteristic_period = 2 * np.pi / max(w_2_sim, w_3_sim)
total_launch_time = characteristic_period * 5 # Launch electrons over 5 cycles of the faster B-field

print(f"Approximate time of flight: {t_flight_approx:.2e} s")
print(f"Simulating {num_electrons_sim} electrons...")

for i in range(num_electrons_sim):
    # This offset simulates launching electrons sequentially into the evolving B-fields.
    t_offset = (i / num_electrons_sim) * total_launch_time

    # Initial conditions: [x1, x2, x3, v1, v2, v3]
    y0 = [0, 0, 0, v_initial_sim, 0, 0]

    # Set evaluation times for dense output. Ensures smooth trajectory data.
    num_time_points = int(max(w_2_sim, w_3_sim) * t_flight_approx / (2*np.pi) * 30)
    num_time_points = max(num_time_points, 200)
    t_eval = np.linspace(t_span[0], t_span[1], num_time_points)

    # REFACTORED: Use the 'args' parameter to pass arguments to the ODE function.
    # This is the standard, efficient, and correct way to handle changing parameters in a loop.
    sol = solve_ivp(
        fun=equations_of_motion,
        t_span=t_span,
        y0=y0,
        method='RK45',
        dense_output=True, # Recommended for accurate final point interpolation
        t_eval=t_eval,
        args=(m_e, e_charge, c_light, B_02_sim, w_2_sim, B_03_sim, w_3_sim, t_offset)
    )

    # Extract the final position on the screen
    final_state = sol.y[:, -1]
    screen_positions_x2.append(final_state[1])
    screen_positions_x3.append(final_state[2])

    if (i+1) % (num_electrons_sim // 10) == 0:
        print(f"  Simulated {i+1}/{num_electrons_sim} electrons...")

print("Simulation complete.")

# %% outputId="3a5a335c-ccf8-4851-84fc-78dccf6d0b62" colab={"base_uri": "https://localhost:8080/", "height": 1000} id="4EFoH7T3k3n8"
plt.figure(figsize=(9, 9))
plt.scatter(screen_positions_x2, screen_positions_x3, s=10, alpha=0.7)
plt.title(f'Electron Beam Pattern on Screen (L={L_sim} cm, $B_0$={B_02_sim} G)')
plt.xlabel('$x_2$ position (cm)')
plt.ylabel('$x_3$ position (cm)')
plt.grid(True)
plt.axis('equal')
# plt.show() # Original static plot show command, moved to conditional block

if generate_gif:
    # --- GIF Generation Parameters ---
    gif_filename = 'electron-beam-uniform-square.gif'
    gif_path = os.path.join(os.getcwd(), gif_filename) # Save in the current notebook directory
    num_frames = num_electrons_sim # One frame per electron for simplicity, can be adjusted
    decay_factor = 0.9 # How much the alpha of a point decreases each frame it persists
    min_alpha = 0.1 # Minimum alpha before a point is removed
    fps = 10 # Frames per second for the GIF

    # --- Generate Frames for GIF ---
    writer = imageio.get_writer(gif_path, fps=fps)
    history_points = [] # Stores (x2, x3, initial_alpha, current_alpha, age)

    print(f"\nGenerating GIF frames for {gif_filename}...")

    fig_gif, ax_gif = plt.subplots(figsize=(9, 9))

    for i in range(num_electrons_sim):
        # Add the new point from this electron
        new_x2 = screen_positions_x2[i]
        new_x3 = screen_positions_x3[i]
        history_points.append([new_x2, new_x3, 1.0, 1.0, 0]) # x2, x3, initial_alpha, current_alpha, age

        ax_gif.clear()
        ax_gif.set_facecolor('black') # Set background to black for better visibility of points
        fig_gif.patch.set_facecolor('black')

        # Update and draw points
        current_frame_points_x2 = []
        current_frame_points_x3 = []
        current_frame_alphas = []

        next_history_points = []
        for point_data in history_points:
            px, py, initial_a, current_a, age = point_data
            current_frame_points_x2.append(px)
            current_frame_points_x3.append(py)
            current_frame_alphas.append(current_a)

            # Decay alpha for next frame
            next_alpha = current_a * decay_factor
            if next_alpha >= min_alpha:
                next_history_points.append([px, py, initial_a, next_alpha, age + 1])

        history_points = next_history_points

        if current_frame_points_x2: # Check if there are any points to plot
            ax_gif.scatter(current_frame_points_x2, current_frame_points_x3, s=15,
                           c=current_frame_alphas, cmap='Blues_r', vmin=0, vmax=1.0,
                           edgecolors='cyan', linewidths=0.5) # Brighter points with cyan edges

        ax_gif.set_title(f'Electron Beam Pattern (Frame {i+1}/{num_electrons_sim})', color='white')
        ax_gif.set_xlabel('$x_2$ position (cm)', color='white')
        ax_gif.set_ylabel('$x_3$ position (cm)', color='white')
        ax_gif.grid(True, color='gray', linestyle=':', linewidth=0.5)
        ax_gif.tick_params(axis='x', colors='white')
        ax_gif.tick_params(axis='y', colors='white')

        # Determine plot limits dynamically based on all points simulated so far
        if screen_positions_x2 and screen_positions_x3: # Ensure lists are not empty
            all_x2 = np.array(screen_positions_x2[:i+1])
            all_x3 = np.array(screen_positions_x3[:i+1])
            margin = 0.1 * max( (all_x2.max()-all_x2.min() if all_x2.size >1 else 1),
                                 (all_x3.max()-all_x3.min() if all_x3.size >1 else 1) )
            margin = max(margin, 0.1) # Ensure a minimum margin

            ax_gif.set_xlim(all_x2.min() - margin, all_x2.max() + margin)
            ax_gif.set_ylim(all_x3.min() - margin, all_x3.max() + margin)
        else: # Default limits if no points yet (should not happen if simulation runs correctly)
            ax_gif.set_xlim(-L_sim*0.1, L_sim*0.1)
            ax_gif.set_ylim(-L_sim*0.1, L_sim*0.1)

        ax_gif.set_aspect('equal', adjustable='box')

        fig_gif.canvas.draw()
        # Convert the canvas to an RGB numpy array
        image_rgba = np.asarray(fig_gif.canvas.buffer_rgba())
        image_rgb = image_rgba[:, :, :3] # Drop the alpha channel
        writer.append_data(image_rgb)

        if (i+1) % (num_electrons_sim // 10) == 0:
            print(f"  Generated frame {i+1}/{num_electrons_sim}...")

    writer.close()
    plt.close(fig_gif) # Close the figure used for GIF generation
    print(f"GIF saved to {gif_path}")

    # Display the GIF in the notebook (optional, may not work in all environments)
    from IPython.display import Image as IPImage
    if os.path.exists(gif_path):
        display(IPImage(filename=gif_path))
    else:
        print(f"Could not find GIF at {gif_path} to display.")
else:
    # --- Generate Static Plot ---
    print("\nGenerating static plot...")
    plt.figure(figsize=(9, 9))
    plt.scatter(screen_positions_x2, screen_positions_x3, s=10, alpha=0.7)
    plt.title(f'Electron Beam Pattern on Screen (L={L_sim} cm, $B_0$={B_02_sim} G)')
    plt.xlabel('$x_2$ position (cm)')
    plt.ylabel('$x_3$ position (cm)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    print("Static plot displayed.")

# %% [markdown] id="pWlWU0vBvzmX"
# ### Trig Wave

# %%
def B2_field(t, B_02_param, w_2_param):
    return B_trig(t, B_02_param, w_2_param)
def dB2_dt_field(t, B_02_param, w_2_param):
    return dB_trig(t, B_02_param, w_2_param)

def B3_field(t, B_03_param, w_3_param):
    return B_trig(t, B_03_param, w_3_param)
def dB3_dt_field(t, B_03_param, w_3_param):
    return dB_trig(t, B_03_param, w_3_param)

# %% id="75530bde"
# REFACTORED: The ODE function is now defined once and takes t_offset as an argument.
# This is a cleaner and more robust approach than redefining a function in each loop iteration.
def equations_of_motion(t_particle, y, m, e, c, B_02_param, w_2_param, B_03_param, w_3_param, t_offset):
    # The B-field evolves on a global clock. Its phase depends on when the electron was launched (t_offset).
    # The solver's time, t_particle, is the time since that specific electron's launch.
    global_t = t_particle + t_offset

    x1, x2, x3, v1, v2, v3 = y

    B2_t = B2_field(global_t, B_02_param, w_2_param)
    dB2_dt_t = dB2_dt_field(global_t, B_02_param, w_2_param)
    B3_t = B3_field(global_t, B_03_param, w_3_param)
    dB3_dt_t = dB3_dt_field(global_t, B_03_param, w_3_param)

    # The equations of motion as provided in the notebook
    a1 = (e / (2 * m * c)) * (x3 * dB2_dt_t - x2 * dB3_dt_t)
    a2 = (e / (m * c)) * (0.5 * x1 * dB3_dt_t - v1 * B3_t)
    a3 = (e / (m * c)) * (-0.5 * x1 * dB2_dt_t + v1 * B2_t)

    return [v1, v2, v3, a1, a2, a3]

# %% colab={"base_uri": "https://localhost:8080/"} id="a735ce0f" outputId="52f965e4-f3c4-4db8-b60d-b3ddd7ccd570"
screen_positions_x2 = []
screen_positions_x3 = []

# Approximate time of flight for one electron. Used to set the integration interval.
t_flight_approx = L_sim / v_initial_sim
t_span = (0, t_flight_approx)

# REVISED LOGIC/COMMENT: The following clarifies how electrons are launched.
# To generate a pattern, we simulate electrons launched at different times into the
# oscillating magnetic fields. This is modeled with a 't_offset' for each electron,
# representing its launch time on a global clock. This is equivalent to applying
# a different initial phase of the B-fields for each electron.
characteristic_period = 2 * np.pi / max(w_2_sim, w_3_sim)
total_launch_time = characteristic_period * 5 # Launch electrons over 5 cycles of the faster B-field

print(f"Approximate time of flight: {t_flight_approx:.2e} s")
print(f"Simulating {num_electrons_sim} electrons...")

for i in range(num_electrons_sim):
    # This offset simulates launching electrons sequentially into the evolving B-fields.
    t_offset = (i / num_electrons_sim) * total_launch_time

    # Initial conditions: [x1, x2, x3, v1, v2, v3]
    y0 = [0, 0, 0, v_initial_sim, 0, 0]

    # Set evaluation times for dense output. Ensures smooth trajectory data.
    num_time_points = int(max(w_2_sim, w_3_sim) * t_flight_approx / (2*np.pi) * 30)
    num_time_points = max(num_time_points, 200)
    t_eval = np.linspace(t_span[0], t_span[1], num_time_points)

    # REFACTORED: Use the 'args' parameter to pass arguments to the ODE function.
    # This is the standard, efficient, and correct way to handle changing parameters in a loop.
    sol = solve_ivp(
        fun=equations_of_motion,
        t_span=t_span,
        y0=y0,
        method='RK45',
        dense_output=True, # Recommended for accurate final point interpolation
        t_eval=t_eval,
        args=(m_e, e_charge, c_light, B_02_sim, w_2_sim, B_03_sim, w_3_sim, t_offset)
    )

    # Extract the final position on the screen
    final_state = sol.y[:, -1]
    screen_positions_x2.append(final_state[1])
    screen_positions_x3.append(final_state[2])

    if (i+1) % (num_electrons_sim // 10) == 0:
        print(f"  Simulated {i+1}/{num_electrons_sim} electrons...")

print("Simulation complete.")

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1a4365b1" outputId="e1a81646-ddcf-42c9-f0de-b87742bab2f8"
plt.figure(figsize=(9, 9))
plt.scatter(screen_positions_x2, screen_positions_x3, s=10, alpha=0.7)
plt.title(f'Electron Beam Pattern on Screen (L={L_sim} cm, $B_0$={B_02_sim} G)')
plt.xlabel('$x_2$ position (cm)')
plt.ylabel('$x_3$ position (cm)')
plt.grid(True)
plt.axis('equal')
# plt.show() # Original static plot show command, moved to conditional block

if generate_gif:
    # --- GIF Generation Parameters ---
    gif_filename = 'electron-beam-uniform-tri.gif'
    gif_path = os.path.join(os.getcwd(), gif_filename) # Save in the current notebook directory
    num_frames = num_electrons_sim # One frame per electron for simplicity, can be adjusted
    decay_factor = 0.9 # How much the alpha of a point decreases each frame it persists
    min_alpha = 0.1 # Minimum alpha before a point is removed
    fps = 10 # Frames per second for the GIF

    # --- Generate Frames for GIF ---
    writer = imageio.get_writer(gif_path, fps=fps)
    history_points = [] # Stores (x2, x3, initial_alpha, current_alpha, age)

    print(f"\nGenerating GIF frames for {gif_filename}...")

    fig_gif, ax_gif = plt.subplots(figsize=(9, 9))

    for i in range(num_electrons_sim):
        # Add the new point from this electron
        new_x2 = screen_positions_x2[i]
        new_x3 = screen_positions_x3[i]
        history_points.append([new_x2, new_x3, 1.0, 1.0, 0]) # x2, x3, initial_alpha, current_alpha, age

        ax_gif.clear()
        ax_gif.set_facecolor('black') # Set background to black for better visibility of points
        fig_gif.patch.set_facecolor('black')

        # Update and draw points
        current_frame_points_x2 = []
        current_frame_points_x3 = []
        current_frame_alphas = []

        next_history_points = []
        for point_data in history_points:
            px, py, initial_a, current_a, age = point_data
            current_frame_points_x2.append(px)
            current_frame_points_x3.append(py)
            current_frame_alphas.append(current_a)

            # Decay alpha for next frame
            next_alpha = current_a * decay_factor
            if next_alpha >= min_alpha:
                next_history_points.append([px, py, initial_a, next_alpha, age + 1])

        history_points = next_history_points

        if current_frame_points_x2: # Check if there are any points to plot
            ax_gif.scatter(current_frame_points_x2, current_frame_points_x3, s=15,
                           c=current_frame_alphas, cmap='Blues_r', vmin=0, vmax=1.0,
                           edgecolors='cyan', linewidths=0.5) # Brighter points with cyan edges

        ax_gif.set_title(f'Electron Beam Pattern (Frame {i+1}/{num_electrons_sim})', color='white')
        ax_gif.set_xlabel('$x_2$ position (cm)', color='white')
        ax_gif.set_ylabel('$x_3$ position (cm)', color='white')
        ax_gif.grid(True, color='gray', linestyle=':', linewidth=0.5)
        ax_gif.tick_params(axis='x', colors='white')
        ax_gif.tick_params(axis='y', colors='white')

        # Determine plot limits dynamically based on all points simulated so far
        if screen_positions_x2 and screen_positions_x3: # Ensure lists are not empty
            all_x2 = np.array(screen_positions_x2[:i+1])
            all_x3 = np.array(screen_positions_x3[:i+1])
            margin = 0.1 * max( (all_x2.max()-all_x2.min() if all_x2.size >1 else 1),
                                 (all_x3.max()-all_x3.min() if all_x3.size >1 else 1) )
            margin = max(margin, 0.1) # Ensure a minimum margin

            ax_gif.set_xlim(all_x2.min() - margin, all_x2.max() + margin)
            ax_gif.set_ylim(all_x3.min() - margin, all_x3.max() + margin)
        else: # Default limits if no points yet (should not happen if simulation runs correctly)
            ax_gif.set_xlim(-L_sim*0.1, L_sim*0.1)
            ax_gif.set_ylim(-L_sim*0.1, L_sim*0.1)

        ax_gif.set_aspect('equal', adjustable='box')

        fig_gif.canvas.draw()
        # Convert the canvas to an RGB numpy array
        image_rgba = np.asarray(fig_gif.canvas.buffer_rgba())
        image_rgb = image_rgba[:, :, :3] # Drop the alpha channel
        writer.append_data(image_rgb)

        if (i+1) % (num_electrons_sim // 10) == 0:
            print(f"  Generated frame {i+1}/{num_electrons_sim}...")

    writer.close()
    plt.close(fig_gif) # Close the figure used for GIF generation
    print(f"GIF saved to {gif_path}")

    # Display the GIF in the notebook (optional, may not work in all environments)
    from IPython.display import Image as IPImage
    if os.path.exists(gif_path):
        display(IPImage(filename=gif_path))
    else:
        print(f"Could not find GIF at {gif_path} to display.")
else:
    # --- Generate Static Plot ---
    print("\nGenerating static plot...")
    plt.figure(figsize=(9, 9))
    plt.scatter(screen_positions_x2, screen_positions_x3, s=10, alpha=0.7)
    plt.title(f'Electron Beam Pattern on Screen (L={L_sim} cm, $B_0$={B_02_sim} G)')
    plt.xlabel('$x_2$ position (cm)')
    plt.ylabel('$x_3$ position (cm)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    print("Static plot displayed.")

# %% [markdown]
## Sin-Square

# %%
def B2_field(t, B_02_param, w_2_param):
    return B_sin(t, B_02_param, w_2_param)
def dB2_dt_field(t, B_02_param, w_2_param):
    return dB_sin(t, B_02_param, w_2_param)

def B3_field(t, B_03_param, w_3_param):
    return B_square(t, B_03_param, w_3_param)
def dB3_dt_field(t, B_03_param, w_3_param):
    return dB_square(t, B_03_param, w_3_param)

# %% id="75530bde"
# REFACTORED: The ODE function is now defined once and takes t_offset as an argument.
# This is a cleaner and more robust approach than redefining a function in each loop iteration.
def equations_of_motion(t_particle, y, m, e, c, B_02_param, w_2_param, B_03_param, w_3_param, t_offset):
    # The B-field evolves on a global clock. Its phase depends on when the electron was launched (t_offset).
    # The solver's time, t_particle, is the time since that specific electron's launch.
    global_t = t_particle + t_offset

    x1, x2, x3, v1, v2, v3 = y

    B2_t = B2_field(global_t, B_02_param, w_2_param)
    dB2_dt_t = dB2_dt_field(global_t, B_02_param, w_2_param)
    B3_t = B3_field(global_t, B_03_param, w_3_param)
    dB3_dt_t = dB3_dt_field(global_t, B_03_param, w_3_param)

    # The equations of motion as provided in the notebook
    a1 = (e / (2 * m * c)) * (x3 * dB2_dt_t - x2 * dB3_dt_t)
    a2 = (e / (m * c)) * (0.5 * x1 * dB3_dt_t - v1 * B3_t)
    a3 = (e / (m * c)) * (-0.5 * x1 * dB2_dt_t + v1 * B2_t)

    return [v1, v2, v3, a1, a2, a3]

# %% colab={"base_uri": "https://localhost:8080/"} id="a735ce0f" outputId="52f965e4-f3c4-4db8-b60d-b3ddd7ccd570"
screen_positions_x2 = []
screen_positions_x3 = []

# Approximate time of flight for one electron. Used to set the integration interval.
t_flight_approx = L_sim / v_initial_sim
t_span = (0, t_flight_approx)

# REVISED LOGIC/COMMENT: The following clarifies how electrons are launched.
# To generate a pattern, we simulate electrons launched at different times into the
# oscillating magnetic fields. This is modeled with a 't_offset' for each electron,
# representing its launch time on a global clock. This is equivalent to applying
# a different initial phase of the B-fields for each electron.
characteristic_period = 2 * np.pi / max(w_2_sim, w_3_sim)
total_launch_time = characteristic_period * 5 # Launch electrons over 5 cycles of the faster B-field

print(f"Approximate time of flight: {t_flight_approx:.2e} s")
print(f"Simulating {num_electrons_sim} electrons...")

for i in range(num_electrons_sim):
    # This offset simulates launching electrons sequentially into the evolving B-fields.
    t_offset = (i / num_electrons_sim) * total_launch_time

    # Initial conditions: [x1, x2, x3, v1, v2, v3]
    y0 = [0, 0, 0, v_initial_sim, 0, 0]

    # Set evaluation times for dense output. Ensures smooth trajectory data.
    num_time_points = int(max(w_2_sim, w_3_sim) * t_flight_approx / (2*np.pi) * 30)
    num_time_points = max(num_time_points, 200)
    t_eval = np.linspace(t_span[0], t_span[1], num_time_points)

    # REFACTORED: Use the 'args' parameter to pass arguments to the ODE function.
    # This is the standard, efficient, and correct way to handle changing parameters in a loop.
    sol = solve_ivp(
        fun=equations_of_motion,
        t_span=t_span,
        y0=y0,
        method='RK45',
        dense_output=True, # Recommended for accurate final point interpolation
        t_eval=t_eval,
        args=(m_e, e_charge, c_light, B_02_sim, w_2_sim, B_03_sim, w_3_sim, t_offset)
    )

    # Extract the final position on the screen
    final_state = sol.y[:, -1]
    screen_positions_x2.append(final_state[1])
    screen_positions_x3.append(final_state[2])

    if (i+1) % (num_electrons_sim // 10) == 0:
        print(f"  Simulated {i+1}/{num_electrons_sim} electrons...")

print("Simulation complete.")

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1a4365b1" outputId="e1a81646-ddcf-42c9-f0de-b87742bab2f8"
plt.figure(figsize=(9, 9))
plt.scatter(screen_positions_x2, screen_positions_x3, s=10, alpha=0.7)
plt.title(f'Electron Beam Pattern on Screen (L={L_sim} cm, $B_0$={B_02_sim} G)')
plt.xlabel('$x_2$ position (cm)')
plt.ylabel('$x_3$ position (cm)')
plt.grid(True)
plt.axis('equal')
# plt.show() # Original static plot show command, moved to conditional block

if generate_gif:
    # --- GIF Generation Parameters ---
    gif_filename = 'electron-beam-uniform-sin-sqr.gif'
    gif_path = os.path.join(os.getcwd(), gif_filename) # Save in the current notebook directory
    num_frames = num_electrons_sim # One frame per electron for simplicity, can be adjusted
    decay_factor = 0.9 # How much the alpha of a point decreases each frame it persists
    min_alpha = 0.1 # Minimum alpha before a point is removed
    fps = 10 # Frames per second for the GIF

    # --- Generate Frames for GIF ---
    writer = imageio.get_writer(gif_path, fps=fps)
    history_points = [] # Stores (x2, x3, initial_alpha, current_alpha, age)

    print(f"\nGenerating GIF frames for {gif_filename}...")

    fig_gif, ax_gif = plt.subplots(figsize=(9, 9))

    for i in range(num_electrons_sim):
        # Add the new point from this electron
        new_x2 = screen_positions_x2[i]
        new_x3 = screen_positions_x3[i]
        history_points.append([new_x2, new_x3, 1.0, 1.0, 0]) # x2, x3, initial_alpha, current_alpha, age

        ax_gif.clear()
        ax_gif.set_facecolor('black') # Set background to black for better visibility of points
        fig_gif.patch.set_facecolor('black')

        # Update and draw points
        current_frame_points_x2 = []
        current_frame_points_x3 = []
        current_frame_alphas = []

        next_history_points = []
        for point_data in history_points:
            px, py, initial_a, current_a, age = point_data
            current_frame_points_x2.append(px)
            current_frame_points_x3.append(py)
            current_frame_alphas.append(current_a)

            # Decay alpha for next frame
            next_alpha = current_a * decay_factor
            if next_alpha >= min_alpha:
                next_history_points.append([px, py, initial_a, next_alpha, age + 1])

        history_points = next_history_points

        if current_frame_points_x2: # Check if there are any points to plot
            ax_gif.scatter(current_frame_points_x2, current_frame_points_x3, s=15,
                           c=current_frame_alphas, cmap='Blues_r', vmin=0, vmax=1.0,
                           edgecolors='cyan', linewidths=0.5) # Brighter points with cyan edges

        ax_gif.set_title(f'Electron Beam Pattern (Frame {i+1}/{num_electrons_sim})', color='white')
        ax_gif.set_xlabel('$x_2$ position (cm)', color='white')
        ax_gif.set_ylabel('$x_3$ position (cm)', color='white')
        ax_gif.grid(True, color='gray', linestyle=':', linewidth=0.5)
        ax_gif.tick_params(axis='x', colors='white')
        ax_gif.tick_params(axis='y', colors='white')

        # Determine plot limits dynamically based on all points simulated so far
        if screen_positions_x2 and screen_positions_x3: # Ensure lists are not empty
            all_x2 = np.array(screen_positions_x2[:i+1])
            all_x3 = np.array(screen_positions_x3[:i+1])
            margin = 0.1 * max( (all_x2.max()-all_x2.min() if all_x2.size >1 else 1),
                                 (all_x3.max()-all_x3.min() if all_x3.size >1 else 1) )
            margin = max(margin, 0.1) # Ensure a minimum margin

            ax_gif.set_xlim(all_x2.min() - margin, all_x2.max() + margin)
            ax_gif.set_ylim(all_x3.min() - margin, all_x3.max() + margin)
        else: # Default limits if no points yet (should not happen if simulation runs correctly)
            ax_gif.set_xlim(-L_sim*0.1, L_sim*0.1)
            ax_gif.set_ylim(-L_sim*0.1, L_sim*0.1)

        ax_gif.set_aspect('equal', adjustable='box')

        fig_gif.canvas.draw()
        # Convert the canvas to an RGB numpy array
        image_rgba = np.asarray(fig_gif.canvas.buffer_rgba())
        image_rgb = image_rgba[:, :, :3] # Drop the alpha channel
        writer.append_data(image_rgb)

        if (i+1) % (num_electrons_sim // 10) == 0:
            print(f"  Generated frame {i+1}/{num_electrons_sim}...")

    writer.close()
    plt.close(fig_gif) # Close the figure used for GIF generation
    print(f"GIF saved to {gif_path}")

    # Display the GIF in the notebook (optional, may not work in all environments)
    from IPython.display import Image as IPImage
    if os.path.exists(gif_path):
        display(IPImage(filename=gif_path))
    else:
        print(f"Could not find GIF at {gif_path} to display.")
else:
    # --- Generate Static Plot ---
    print("\nGenerating static plot...")
    plt.figure(figsize=(9, 9))
    plt.scatter(screen_positions_x2, screen_positions_x3, s=10, alpha=0.7)
    plt.title(f'Electron Beam Pattern on Screen (L={L_sim} cm, $B_0$={B_02_sim} G)')
    plt.xlabel('$x_2$ position (cm)')
    plt.ylabel('$x_3$ position (cm)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    print("Static plot displayed.")

# %% [markdown]
## Sin-Trig

# %%
def B2_field(t, B_02_param, w_2_param):
    return B_sin(t, B_02_param, w_2_param)
def dB2_dt_field(t, B_02_param, w_2_param):
    return dB_sin(t, B_02_param, w_2_param)

def B3_field(t, B_03_param, w_3_param):
    return B_trig(t, B_03_param, w_3_param)
def dB3_dt_field(t, B_03_param, w_3_param):
    return dB_trig(t, B_03_param, w_3_param)

# %% id="75530bde"
# REFACTORED: The ODE function is now defined once and takes t_offset as an argument.
# This is a cleaner and more robust approach than redefining a function in each loop iteration.
def equations_of_motion(t_particle, y, m, e, c, B_02_param, w_2_param, B_03_param, w_3_param, t_offset):
    # The B-field evolves on a global clock. Its phase depends on when the electron was launched (t_offset).
    # The solver's time, t_particle, is the time since that specific electron's launch.
    global_t = t_particle + t_offset

    x1, x2, x3, v1, v2, v3 = y

    B2_t = B2_field(global_t, B_02_param, w_2_param)
    dB2_dt_t = dB2_dt_field(global_t, B_02_param, w_2_param)
    B3_t = B3_field(global_t, B_03_param, w_3_param)
    dB3_dt_t = dB3_dt_field(global_t, B_03_param, w_3_param)

    # The equations of motion as provided in the notebook
    a1 = (e / (2 * m * c)) * (x3 * dB2_dt_t - x2 * dB3_dt_t)
    a2 = (e / (m * c)) * (0.5 * x1 * dB3_dt_t - v1 * B3_t)
    a3 = (e / (m * c)) * (-0.5 * x1 * dB2_dt_t + v1 * B2_t)

    return [v1, v2, v3, a1, a2, a3]

# %% colab={"base_uri": "https://localhost:8080/"} id="a735ce0f" outputId="52f965e4-f3c4-4db8-b60d-b3ddd7ccd570"
screen_positions_x2 = []
screen_positions_x3 = []

# Approximate time of flight for one electron. Used to set the integration interval.
t_flight_approx = L_sim / v_initial_sim
t_span = (0, t_flight_approx)

# REVISED LOGIC/COMMENT: The following clarifies how electrons are launched.
# To generate a pattern, we simulate electrons launched at different times into the
# oscillating magnetic fields. This is modeled with a 't_offset' for each electron,
# representing its launch time on a global clock. This is equivalent to applying
# a different initial phase of the B-fields for each electron.
characteristic_period = 2 * np.pi / max(w_2_sim, w_3_sim)
total_launch_time = characteristic_period * 5 # Launch electrons over 5 cycles of the faster B-field

print(f"Approximate time of flight: {t_flight_approx:.2e} s")
print(f"Simulating {num_electrons_sim} electrons...")

for i in range(num_electrons_sim):
    # This offset simulates launching electrons sequentially into the evolving B-fields.
    t_offset = (i / num_electrons_sim) * total_launch_time

    # Initial conditions: [x1, x2, x3, v1, v2, v3]
    y0 = [0, 0, 0, v_initial_sim, 0, 0]

    # Set evaluation times for dense output. Ensures smooth trajectory data.
    num_time_points = int(max(w_2_sim, w_3_sim) * t_flight_approx / (2*np.pi) * 30)
    num_time_points = max(num_time_points, 200)
    t_eval = np.linspace(t_span[0], t_span[1], num_time_points)

    # REFACTORED: Use the 'args' parameter to pass arguments to the ODE function.
    # This is the standard, efficient, and correct way to handle changing parameters in a loop.
    sol = solve_ivp(
        fun=equations_of_motion,
        t_span=t_span,
        y0=y0,
        method='RK45',
        dense_output=True, # Recommended for accurate final point interpolation
        t_eval=t_eval,
        args=(m_e, e_charge, c_light, B_02_sim, w_2_sim, B_03_sim, w_3_sim, t_offset)
    )

    # Extract the final position on the screen
    final_state = sol.y[:, -1]
    screen_positions_x2.append(final_state[1])
    screen_positions_x3.append(final_state[2])

    if (i+1) % (num_electrons_sim // 10) == 0:
        print(f"  Simulated {i+1}/{num_electrons_sim} electrons...")

print("Simulation complete.")

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1a4365b1" outputId="e1a81646-ddcf-42c9-f0de-b87742bab2f8"
plt.figure(figsize=(9, 9))
plt.scatter(screen_positions_x2, screen_positions_x3, s=10, alpha=0.7)
plt.title(f'Electron Beam Pattern on Screen (L={L_sim} cm, $B_0$={B_02_sim} G)')
plt.xlabel('$x_2$ position (cm)')
plt.ylabel('$x_3$ position (cm)')
plt.grid(True)
plt.axis('equal')
# plt.show() # Original static plot show command, moved to conditional block

if generate_gif:
    # --- GIF Generation Parameters ---
    gif_filename = 'electron-beam-uniform-sin-tri.gif'
    gif_path = os.path.join(os.getcwd(), gif_filename) # Save in the current notebook directory
    num_frames = num_electrons_sim # One frame per electron for simplicity, can be adjusted
    decay_factor = 0.9 # How much the alpha of a point decreases each frame it persists
    min_alpha = 0.1 # Minimum alpha before a point is removed
    fps = 10 # Frames per second for the GIF

    # --- Generate Frames for GIF ---
    writer = imageio.get_writer(gif_path, fps=fps)
    history_points = [] # Stores (x2, x3, initial_alpha, current_alpha, age)

    print(f"\nGenerating GIF frames for {gif_filename}...")

    fig_gif, ax_gif = plt.subplots(figsize=(9, 9))

    for i in range(num_electrons_sim):
        # Add the new point from this electron
        new_x2 = screen_positions_x2[i]
        new_x3 = screen_positions_x3[i]
        history_points.append([new_x2, new_x3, 1.0, 1.0, 0]) # x2, x3, initial_alpha, current_alpha, age

        ax_gif.clear()
        ax_gif.set_facecolor('black') # Set background to black for better visibility of points
        fig_gif.patch.set_facecolor('black')

        # Update and draw points
        current_frame_points_x2 = []
        current_frame_points_x3 = []
        current_frame_alphas = []

        next_history_points = []
        for point_data in history_points:
            px, py, initial_a, current_a, age = point_data
            current_frame_points_x2.append(px)
            current_frame_points_x3.append(py)
            current_frame_alphas.append(current_a)

            # Decay alpha for next frame
            next_alpha = current_a * decay_factor
            if next_alpha >= min_alpha:
                next_history_points.append([px, py, initial_a, next_alpha, age + 1])

        history_points = next_history_points

        if current_frame_points_x2: # Check if there are any points to plot
            ax_gif.scatter(current_frame_points_x2, current_frame_points_x3, s=15,
                           c=current_frame_alphas, cmap='Blues_r', vmin=0, vmax=1.0,
                           edgecolors='cyan', linewidths=0.5) # Brighter points with cyan edges

        ax_gif.set_title(f'Electron Beam Pattern (Frame {i+1}/{num_electrons_sim})', color='white')
        ax_gif.set_xlabel('$x_2$ position (cm)', color='white')
        ax_gif.set_ylabel('$x_3$ position (cm)', color='white')
        ax_gif.grid(True, color='gray', linestyle=':', linewidth=0.5)
        ax_gif.tick_params(axis='x', colors='white')
        ax_gif.tick_params(axis='y', colors='white')

        # Determine plot limits dynamically based on all points simulated so far
        if screen_positions_x2 and screen_positions_x3: # Ensure lists are not empty
            all_x2 = np.array(screen_positions_x2[:i+1])
            all_x3 = np.array(screen_positions_x3[:i+1])
            margin = 0.1 * max( (all_x2.max()-all_x2.min() if all_x2.size >1 else 1),
                                 (all_x3.max()-all_x3.min() if all_x3.size >1 else 1) )
            margin = max(margin, 0.1) # Ensure a minimum margin

            ax_gif.set_xlim(all_x2.min() - margin, all_x2.max() + margin)
            ax_gif.set_ylim(all_x3.min() - margin, all_x3.max() + margin)
        else: # Default limits if no points yet (should not happen if simulation runs correctly)
            ax_gif.set_xlim(-L_sim*0.1, L_sim*0.1)
            ax_gif.set_ylim(-L_sim*0.1, L_sim*0.1)

        ax_gif.set_aspect('equal', adjustable='box')

        fig_gif.canvas.draw()
        # Convert the canvas to an RGB numpy array
        image_rgba = np.asarray(fig_gif.canvas.buffer_rgba())
        image_rgb = image_rgba[:, :, :3] # Drop the alpha channel
        writer.append_data(image_rgb)

        if (i+1) % (num_electrons_sim // 10) == 0:
            print(f"  Generated frame {i+1}/{num_electrons_sim}...")

    writer.close()
    plt.close(fig_gif) # Close the figure used for GIF generation
    print(f"GIF saved to {gif_path}")

    # Display the GIF in the notebook (optional, may not work in all environments)
    from IPython.display import Image as IPImage
    if os.path.exists(gif_path):
        display(IPImage(filename=gif_path))
    else:
        print(f"Could not find GIF at {gif_path} to display.")
else:
    # --- Generate Static Plot ---
    print("\nGenerating static plot...")
    plt.figure(figsize=(9, 9))
    plt.scatter(screen_positions_x2, screen_positions_x3, s=10, alpha=0.7)
    plt.title(f'Electron Beam Pattern on Screen (L={L_sim} cm, $B_0$={B_02_sim} G)')
    plt.xlabel('$x_2$ position (cm)')
    plt.ylabel('$x_3$ position (cm)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    print("Static plot displayed.")

# %% [markdown]
## Square-Tri

# %%
def B2_field(t, B_02_param, w_2_param):
    return B_square(t, B_02_param, w_2_param)
def dB2_dt_field(t, B_02_param, w_2_param):
    return dB_square(t, B_02_param, w_2_param)

def B3_field(t, B_03_param, w_3_param):
    return B_trig(t, B_03_param, w_3_param)
def dB3_dt_field(t, B_03_param, w_3_param):
    return dB_trig(t, B_03_param, w_3_param)

# %% id="75530bde"
# REFACTORED: The ODE function is now defined once and takes t_offset as an argument.
# This is a cleaner and more robust approach than redefining a function in each loop iteration.
def equations_of_motion(t_particle, y, m, e, c, B_02_param, w_2_param, B_03_param, w_3_param, t_offset):
    # The B-field evolves on a global clock. Its phase depends on when the electron was launched (t_offset).
    # The solver's time, t_particle, is the time since that specific electron's launch.
    global_t = t_particle + t_offset

    x1, x2, x3, v1, v2, v3 = y

    B2_t = B2_field(global_t, B_02_param, w_2_param)
    dB2_dt_t = dB2_dt_field(global_t, B_02_param, w_2_param)
    B3_t = B3_field(global_t, B_03_param, w_3_param)
    dB3_dt_t = dB3_dt_field(global_t, B_03_param, w_3_param)

    # The equations of motion as provided in the notebook
    a1 = (e / (2 * m * c)) * (x3 * dB2_dt_t - x2 * dB3_dt_t)
    a2 = (e / (m * c)) * (0.5 * x1 * dB3_dt_t - v1 * B3_t)
    a3 = (e / (m * c)) * (-0.5 * x1 * dB2_dt_t + v1 * B2_t)

    return [v1, v2, v3, a1, a2, a3]

# %% colab={"base_uri": "https://localhost:8080/"} id="a735ce0f" outputId="52f965e4-f3c4-4db8-b60d-b3ddd7ccd570"
screen_positions_x2 = []
screen_positions_x3 = []

# Approximate time of flight for one electron. Used to set the integration interval.
t_flight_approx = L_sim / v_initial_sim
t_span = (0, t_flight_approx)

# REVISED LOGIC/COMMENT: The following clarifies how electrons are launched.
# To generate a pattern, we simulate electrons launched at different times into the
# oscillating magnetic fields. This is modeled with a 't_offset' for each electron,
# representing its launch time on a global clock. This is equivalent to applying
# a different initial phase of the B-fields for each electron.
characteristic_period = 2 * np.pi / max(w_2_sim, w_3_sim)
total_launch_time = characteristic_period * 5 # Launch electrons over 5 cycles of the faster B-field

print(f"Approximate time of flight: {t_flight_approx:.2e} s")
print(f"Simulating {num_electrons_sim} electrons...")

for i in range(num_electrons_sim):
    # This offset simulates launching electrons sequentially into the evolving B-fields.
    t_offset = (i / num_electrons_sim) * total_launch_time

    # Initial conditions: [x1, x2, x3, v1, v2, v3]
    y0 = [0, 0, 0, v_initial_sim, 0, 0]

    # Set evaluation times for dense output. Ensures smooth trajectory data.
    num_time_points = int(max(w_2_sim, w_3_sim) * t_flight_approx / (2*np.pi) * 30)
    num_time_points = max(num_time_points, 200)
    t_eval = np.linspace(t_span[0], t_span[1], num_time_points)

    # REFACTORED: Use the 'args' parameter to pass arguments to the ODE function.
    # This is the standard, efficient, and correct way to handle changing parameters in a loop.
    sol = solve_ivp(
        fun=equations_of_motion,
        t_span=t_span,
        y0=y0,
        method='RK45',
        dense_output=True, # Recommended for accurate final point interpolation
        t_eval=t_eval,
        args=(m_e, e_charge, c_light, B_02_sim, w_2_sim, B_03_sim, w_3_sim, t_offset)
    )

    # Extract the final position on the screen
    final_state = sol.y[:, -1]
    screen_positions_x2.append(final_state[1])
    screen_positions_x3.append(final_state[2])

    if (i+1) % (num_electrons_sim // 10) == 0:
        print(f"  Simulated {i+1}/{num_electrons_sim} electrons...")

print("Simulation complete.")

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1a4365b1" outputId="e1a81646-ddcf-42c9-f0de-b87742bab2f8"
plt.figure(figsize=(9, 9))
plt.scatter(screen_positions_x2, screen_positions_x3, s=10, alpha=0.7)
plt.title(f'Electron Beam Pattern on Screen (L={L_sim} cm, $B_0$={B_02_sim} G)')
plt.xlabel('$x_2$ position (cm)')
plt.ylabel('$x_3$ position (cm)')
plt.grid(True)
plt.axis('equal')
# plt.show() # Original static plot show command, moved to conditional block

if generate_gif:
    # --- GIF Generation Parameters ---
    gif_filename = 'electron-beam-uniform-sqr-tri.gif'
    gif_path = os.path.join(os.getcwd(), gif_filename) # Save in the current notebook directory
    num_frames = num_electrons_sim # One frame per electron for simplicity, can be adjusted
    decay_factor = 0.9 # How much the alpha of a point decreases each frame it persists
    min_alpha = 0.1 # Minimum alpha before a point is removed
    fps = 10 # Frames per second for the GIF

    # --- Generate Frames for GIF ---
    writer = imageio.get_writer(gif_path, fps=fps)
    history_points = [] # Stores (x2, x3, initial_alpha, current_alpha, age)

    print(f"\nGenerating GIF frames for {gif_filename}...")

    fig_gif, ax_gif = plt.subplots(figsize=(9, 9))

    for i in range(num_electrons_sim):
        # Add the new point from this electron
        new_x2 = screen_positions_x2[i]
        new_x3 = screen_positions_x3[i]
        history_points.append([new_x2, new_x3, 1.0, 1.0, 0]) # x2, x3, initial_alpha, current_alpha, age

        ax_gif.clear()
        ax_gif.set_facecolor('black') # Set background to black for better visibility of points
        fig_gif.patch.set_facecolor('black')

        # Update and draw points
        current_frame_points_x2 = []
        current_frame_points_x3 = []
        current_frame_alphas = []

        next_history_points = []
        for point_data in history_points:
            px, py, initial_a, current_a, age = point_data
            current_frame_points_x2.append(px)
            current_frame_points_x3.append(py)
            current_frame_alphas.append(current_a)

            # Decay alpha for next frame
            next_alpha = current_a * decay_factor
            if next_alpha >= min_alpha:
                next_history_points.append([px, py, initial_a, next_alpha, age + 1])

        history_points = next_history_points

        if current_frame_points_x2: # Check if there are any points to plot
            ax_gif.scatter(current_frame_points_x2, current_frame_points_x3, s=15,
                           c=current_frame_alphas, cmap='Blues_r', vmin=0, vmax=1.0,
                           edgecolors='cyan', linewidths=0.5) # Brighter points with cyan edges

        ax_gif.set_title(f'Electron Beam Pattern (Frame {i+1}/{num_electrons_sim})', color='white')
        ax_gif.set_xlabel('$x_2$ position (cm)', color='white')
        ax_gif.set_ylabel('$x_3$ position (cm)', color='white')
        ax_gif.grid(True, color='gray', linestyle=':', linewidth=0.5)
        ax_gif.tick_params(axis='x', colors='white')
        ax_gif.tick_params(axis='y', colors='white')

        # Determine plot limits dynamically based on all points simulated so far
        if screen_positions_x2 and screen_positions_x3: # Ensure lists are not empty
            all_x2 = np.array(screen_positions_x2[:i+1])
            all_x3 = np.array(screen_positions_x3[:i+1])
            margin = 0.1 * max( (all_x2.max()-all_x2.min() if all_x2.size >1 else 1),
                                 (all_x3.max()-all_x3.min() if all_x3.size >1 else 1) )
            margin = max(margin, 0.1) # Ensure a minimum margin

            ax_gif.set_xlim(all_x2.min() - margin, all_x2.max() + margin)
            ax_gif.set_ylim(all_x3.min() - margin, all_x3.max() + margin)
        else: # Default limits if no points yet (should not happen if simulation runs correctly)
            ax_gif.set_xlim(-L_sim*0.1, L_sim*0.1)
            ax_gif.set_ylim(-L_sim*0.1, L_sim*0.1)

        ax_gif.set_aspect('equal', adjustable='box')

        fig_gif.canvas.draw()
        # Convert the canvas to an RGB numpy array
        image_rgba = np.asarray(fig_gif.canvas.buffer_rgba())
        image_rgb = image_rgba[:, :, :3] # Drop the alpha channel
        writer.append_data(image_rgb)

        if (i+1) % (num_electrons_sim // 10) == 0:
            print(f"  Generated frame {i+1}/{num_electrons_sim}...")

    writer.close()
    plt.close(fig_gif) # Close the figure used for GIF generation
    print(f"GIF saved to {gif_path}")

    # Display the GIF in the notebook (optional, may not work in all environments)
    from IPython.display import Image as IPImage
    if os.path.exists(gif_path):
        display(IPImage(filename=gif_path))
    else:
        print(f"Could not find GIF at {gif_path} to display.")
else:
    # --- Generate Static Plot ---
    print("\nGenerating static plot...")
    plt.figure(figsize=(9, 9))
    plt.scatter(screen_positions_x2, screen_positions_x3, s=10, alpha=0.7)
    plt.title(f'Electron Beam Pattern on Screen (L={L_sim} cm, $B_0$={B_02_sim} G)')
    plt.xlabel('$x_2$ position (cm)')
    plt.ylabel('$x_3$ position (cm)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    print("Static plot displayed.")


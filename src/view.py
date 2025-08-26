import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation, PillowWriter
import math
import torch

def spring(start, end, nodes, width):
    """Generate points for a spring."""
    nodes = max(int(nodes), 1)
    length = np.linalg.norm(np.array(end) - np.array(start))
    u_t = (np.array(end) - np.array(start)) / length
    u_n = np.array([-u_t[1], u_t[0]])

    # Initialize spring coordinates
    spring_coords = np.zeros((2, nodes + 2))
    spring_coords[:, 0], spring_coords[:, -1] = start, end

    # Calculate normal distance
    normal_dist = math.sqrt(max(0, width**2 - (length**2 / nodes**2))) / 2

    # Create spring points
    for i in range(1, nodes + 1):
        spring_coords[:, i] = (
            start
            + ((length * (2 * i - 1) * u_t) / (2 * nodes))
            + (normal_dist * (-1) ** i * u_n)
        )

    return spring_coords[0, :], spring_coords[1, :]


def animate_springs(ts, thetas, u_i, y_i, path="anis/animation.gif", frame_skip=1):
    """Animates the spring movement based on the solution array.

    Args:
        ts (np.array): Time steps.
        ys (np.array): Solution array.
        u_i (np.array): Initial positions.
        y_i (np.array): Final positions.
        path (str): Path to save the animation.
        frame_skip (int): Number of frames to skip.

    Returns:
        str: Path to the saved animation.
    """
    time_steps = len(ts)
    # frame_skip = max(1, time_steps // 200)  # Adjust frame skip to manage animation length

    ell = max(u_i) - min(u_i)
    solution = np.array(thetas)

    # Setting up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(u_i, y_i, label="Data points", marker="x", c="black")
    linspace = np.linspace(min(u_i), max(u_i), 1000)

    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=14, color="black"
    )
    (line,) = ax.plot([], [], label="Solution", c="black", linewidth=3)

    global_y_max = y_i.max()
    global_y_min = y_i.min()

    for frame in range(0, time_steps, frame_skip):
        q0f = solution[frame, 0]
        q1f = solution[frame, 1]
        global_y_max = max(global_y_max, np.max([q0f, q1f]))
        global_y_min = min(global_y_min, np.min([q0f, q1f]))

    ax.set_ylim(global_y_min - 0.1, global_y_max + 0.1)
    N = len(u_i)  # Number of springs
    spring_lines = [
        mlines.Line2D([], [], color="grey", linestyle="solid") for _ in range(N)
    ]

    for spring_line in spring_lines:
        ax.add_line(spring_line)

    def init():
        time_text.set_text("")
        line.set_data([], [])
        for spring_line in spring_lines:
            spring_line.set_data([], [])
        return [scatter, time_text, line] + spring_lines

    def update(frame):
        # Assuming 'solution' is a 2D array where each row is a timestep
        q0f = solution[frame * frame_skip, 0]
        q1f = solution[frame * frame_skip, 1]
        line.set_data(linspace, q0f + (q1f - q0f) * (linspace - q0f) / ell)

        for i, spring_line in enumerate(spring_lines):
            x_spring, y_spring = spring(
                [u_i[i], y_i[i]],
                [u_i[i], q0f + (q1f - q0f) * (u_i[i] - q0f) / ell],
                nodes=10,
                width=0.1,
            )
            spring_line.set_data(x_spring, y_spring)

        # Update time text
        current_time = ts[frame * frame_skip]
        time_text.set_text(f"Time: {current_time:.2f}")

        return [scatter, time_text, line] + spring_lines

    ani = FuncAnimation(
        fig, update, frames=time_steps // frame_skip, init_func=init, blit=True
    )
    writer = PillowWriter(fps=10)

    ani.save(path, writer=writer)

    return path


def generalized_animation(ts, thetas, u_i, y_i, path="anis/ganimation.gif", frame_skip=1):
    """Animates the spring movement based on the solution array."""
    time_steps = len(ts)
    # frame_skip = max(1, time_steps // 200)  # Adjust frame skip to manage animation length

    ell = max(u_i) - min(u_i)
    solution = np.array(thetas)

    n_vars = int(solution.shape[1] // 2)
    linspace_points = 1000 if n_vars <= 100 else 10000

    # Setting up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(u_i, y_i, label="Data points", marker="o", c="blue", s=10, zorder=10)
    linspace = np.linspace(min(u_i), max(u_i), linspace_points)
    n_points_seg = linspace_points // (n_vars - 1)
    seg_length = (max(u_i) - min(u_i)) / (n_vars - 1)

    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=14, color="black"
    )
    (lines,) = ax.plot([], [], label="Solution", c="black", linewidth=3)

    global_y_max = y_i.max()
    global_y_min = y_i.min()

    for frame in range(0, time_steps, frame_skip):
        y_vals = [solution[frame, i] for i in range(n_vars)]
        global_y_max = max(global_y_max, np.max(y_vals))
        global_y_min = min(global_y_min, np.min(y_vals))

    ax.set_ylim(global_y_min, global_y_max)

    N = len(u_i)  # Number of springs
    umin = min(u_i)
    spring_lines = [
        mlines.Line2D([], [], color="grey", linestyle="solid") for _ in range(N)
    ]

    for spring_line in spring_lines:
        ax.add_line(spring_line)

    def init():
        time_text.set_text("")
        lines.set_data([], [])
        for spring_line in spring_lines:
            spring_line.set_data([], [])
        return [scatter, time_text, lines] + spring_lines

    def update(frame):
        sticks = []
        for i in range(n_vars - 1):
            st_start = solution[frame * frame_skip, i]
            st_end = solution[frame * frame_skip, i + 1]
            x_seg = linspace[n_points_seg * i : n_points_seg * (i + 1)]
            slope = (st_end - st_start) / seg_length
            dif = x_seg - umin - i * seg_length
            stick = st_start + slope * dif
            sticks.append(stick)

        sticks = np.array(sticks).flatten()
        lines.set_data(linspace, np.array(sticks))

        for i, spring_line in enumerate(spring_lines):
            interval = int((u_i[i] - umin) // seg_length)
            if interval == n_vars - 1:
                interval -= 1

            init_i = solution[frame * frame_skip, interval]
            final_i = solution[frame * frame_skip, interval + 1]

            slope = (final_i - init_i) / seg_length
            dif = u_i[i] - umin - interval * seg_length
            y_val = init_i + slope * dif

            x_spring, y_spring = spring(
                [u_i[i], y_i[i]], [u_i[i], y_val], nodes=10, width=0.08
            )
            spring_line.set_data(x_spring, y_spring)

        # Update time text
        current_time = ts[frame * frame_skip]
        time_text.set_text(f"Time: {current_time:.2f}")

        return [scatter, time_text, lines] + spring_lines

    ani = FuncAnimation(
        fig, update, frames=time_steps // frame_skip, init_func=init, blit=True
    )
    writer = PillowWriter(fps=10)
    ani.save(path, writer=writer)

    return path


def generalized_single_frame(ts, thetas, u_i, y_i, time_index=0, path="figs/single_frame_generalized.pdf", ax=None):
    """Saves a single frame of the system based on the solution array at a specified time index."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    solution = np.array(thetas)

    n_vars = int(solution.shape[1] // 2)
    linspace_points = 1000 if n_vars <= 100 else 10000

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    linspace = np.linspace(min(u_i), max(u_i), linspace_points)
    n_points_seg = linspace_points // (n_vars - 1)
    seg_length = (max(u_i) - min(u_i)) / (n_vars - 1)

    y_vals = [solution[time_index, i] for i in range(n_vars)]
    global_y_max = max(max(y_vals), max(y_i)) * 1.1
    global_y_min = min(min(y_vals), min(y_i)) * 1.1

    ax.set_ylim(global_y_min, global_y_max)

    N = len(u_i)  # Number of springs
    umin = min(u_i)

    # Compute positions and plot
    sticks = []
    endpoints = []
    x_endpoints = []
    for i in range(n_vars - 1):
        st_start = solution[time_index, i]
        st_end = solution[time_index, i + 1]
        x_seg = linspace[n_points_seg * i : n_points_seg * (i + 1)]
        slope = (st_end - st_start) / seg_length
        dif = x_seg - umin - i * seg_length
        stick = st_start + slope * dif
        sticks.append(stick)
        endpoints.append(st_start)
        x_endpoints.append(umin + i * seg_length)
        if i == n_vars - 2:
            endpoints.append(st_end)
            x_endpoints.append(umin + (i + 1) * seg_length)

    
    for i in range(len(u_i)):
        interval = int((u_i[i] - umin) // seg_length)
        if interval == n_vars - 1:
            interval -= 1

        init_i = solution[time_index, interval]
        final_i = solution[time_index, interval + 1]

        slope = (final_i - init_i) / seg_length
        dif = u_i[i] - umin - interval * seg_length
        y_val = init_i + slope * dif

        x_spring, y_spring = spring(
            [u_i[i], y_i[i]], [u_i[i], y_val], nodes=10, width=0.1
        )
        ax.plot(x_spring, y_spring, c="gray")

    sticks = np.array(sticks).flatten()
    ax.plot(linspace, np.array(sticks), label="Solution", c="black", linewidth=1.5)
    # add x markers in the endpoints of the sticks
    ax.scatter(x_endpoints, endpoints, c="black", marker="x", s=5)
    ax.scatter(u_i, y_i, label="Data points", marker="o", c="b", s=10, zorder=10)
    ax.set_xlabel(r"$x$", fontsize=16)
    ax.set_ylabel(r"$f(x)$", fontsize=16)
    
    #ticks
    ax.tick_params(axis='both', which='major', labelsize=16)
    

    # current_time = ts[time_index]
    # ax.text(0.02, 0.95, f"Time: {current_time:.2f}", transform=ax.transAxes, fontsize=14, color="black")

    plt.savefig(path, bbox_inches="tight")

    return fig, ax


def spring_3d(start, end, nodes, width):
    """Generate points for a spring in 3D."""
    nodes = max(int(nodes), 2)
    diff = np.array(end) - np.array(start)
    length = np.linalg.norm(diff)
    u_t = diff / length

    # Find a vector perpendicular to u_t
    # It can be any of the vectors that are orthogonal to u_t. If u_t is not parallel or anti-parallel to the z-axis, 
    # we can take the cross product with z-axis. Otherwise, we can use the y-axis.
    if (u_t == [0, 0, 1]).all() or (u_t == [0, 0, -1]).all():
        u_n = np.cross(u_t, np.array([0, 1, 0]))
    else:
        u_n = np.cross(u_t, np.array([0, 0, 1]))

    u_n = u_n / np.linalg.norm(u_n)
    u_b = np.cross(u_t, u_n)  # Binormal vector

    # Initialize spring coordinates
    spring_coords = np.zeros((3, nodes + 2))
    spring_coords[:, 0], spring_coords[:, -1] = start, end

    # Calculate displacements for the spring "windings"
    for i in range(1, nodes + 1):
        angle = math.pi * i  # Angle for normal/binormal oscillation
        displacement = width * math.cos(angle) * u_n + width * math.sin(angle) * u_b
        spring_coords[:, i] = start + u_t * (length * i / (nodes + 1)) + displacement

    return spring_coords

def spring_3d(start, end, nodes, base_width):
    """Generate points for a spring in 3D with varying coil width."""
    
    nodes = max(int(nodes), 2)
    diff = np.array(end) - np.array(start)
    length = np.linalg.norm(diff)
    u_t = diff / length

    # Find a vector perpendicular to u_t
    if np.allclose(u_t, [0, 0, 1]) or np.allclose(u_t, [0, 0, -1]):
        u_n = np.cross(u_t, np.array([1, 0, 0]))
    else:
        u_n = np.cross(u_t, np.array([0, 0, 1]))
    u_n = u_n / np.linalg.norm(u_n)
    u_b = np.cross(u_t, u_n)

    spring_coords = np.zeros((3, nodes + 2))
    spring_coords[:, 0], spring_coords[:, -1] = start, end

    # Create a varying width effect for the spring
    for i in range(1, nodes + 1):
        # Modulate the width here; for example, by varying it sinusoidally
        width = base_width * (1 + 0.1 * np.sin(2 * math.pi * i / nodes))
        angle = 8 * math.pi * i / nodes
        displacement = width * np.cos(angle) * u_n + width * np.sin(angle) * u_b
        spring_coords[:, i] = start + u_t * (length * i / (nodes + 1)) + displacement

    return spring_coords

def generalized_animation_3d(ts, thetas, u_i, y_i, n_pieces, sde, path="anis/animation_3d.gif", frame_skip=1):
    """Animates the surface movement based on the solution array in 3D."""
    
    time_steps = len(ts)
    n_vars = int(thetas.shape[1] / 2)  # Assuming ys has x and y values for each grid point

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Preparing the grid (assuming it's uniform)
    u_x, u_y = np.meshgrid(np.linspace(sde.u_min[0], sde.u_max[0], n_pieces[0]), 
                           np.linspace(sde.u_min[1], sde.u_max[1], n_pieces[1]))
    
    springs = [ax.plot([], [], [], 'gray')[0] for _ in range(len(u_i[0]))]

    z_min = thetas[:, :n_vars].min()
    z_max = thetas[:, :n_vars].max()

    z_min = np.min([z_min, y_i.min()])
    z_max = np.max([z_max, y_i.max()])

    def get_z_data(frame_number):
        return np.array(thetas[frame_number][:n_vars]).reshape(n_pieces)

    def init():
        return []

    def update(frame):
        # Clearing the axes allows for the new surface to be drawn without residual old data
        ax.clear()
        ax.set_xlabel(r'$y$', fontsize=20)
        ax.set_ylabel(r'$x$', fontsize=20)
        ax.set_zlabel(r'$z$', fontsize=20)

        ax.tick_params(axis='both', which='major', labelsize=20)

        zs = get_z_data(frame)
        ax.set_zlim(z_min, z_max)
        ax.scatter(u_i[:, 1], u_i[:, 0], y_i[:, 0], label="Data points", marker="x", c="black")
        ax.plot_surface(u_x, u_y, zs, cmap='inferno', alpha=0.5, edgecolor='k', linewidth=2)
        time_text.set_text(f"Time: {ts[frame]:.2f}s")

        for i in range(u_i.shape[0]):
            # Start and end points for the springs
            start = np.array([u_i[i, 0], u_i[i, 1], y_i[i, 0]])
            y_pred = sde.num_y_prediction(u_i[i, :], thetas[frame, :])
            end = np.array([u_i[i, 0], u_i[i, 1], y_pred[0][0]])

            # Generate spring coordinates and plot
            spring_coords = spring_3d(start, end, 50, 0.1)
            ax.plot(spring_coords[1], spring_coords[0], spring_coords[2], 'gray')

        plt.tight_layout()

        return fig,

    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color="black")

    ani = FuncAnimation(fig, update, frames=np.arange(0, time_steps, frame_skip), init_func=init, blit=False)

    writer = PillowWriter(fps=20)
    ani.save(path, writer=writer)

    return path


def plot_final_frame_3d(ts, thetas, u_i, y_i, n_pieces, sde, path="figs/general_final_frame.pdf", frame=-1):
    """
    Plots the final frame of the 3D animation and saves it as a PDF.
    
    Parameters:
      ts      : array of time steps
      thetas  : array containing surface data for each time step; assumed to be shaped (T, 2*n_vars)
      u_i     : array of grid point coordinates, shape (N, 2)
      y_i     : array of data point z-values, shape (N, 1) or (N,)
      n_pieces: tuple indicating the grid shape (nx, ny)
      sde     : object with attributes u_min, u_max and method num_y_prediction for computing predictions
      pdf_path: output path for the PDF file
    """
    n_vars = int(thetas.shape[1] / 2)
    
    # Prepare the grid
    u_x, u_y = np.meshgrid(
        np.linspace(sde.u_min[0], sde.u_max[0], n_pieces[0]),
        np.linspace(sde.u_min[1], sde.u_max[1], n_pieces[1])
    )
    
    # Determine z-limits from both thetas and y_i
    # z_min = min(thetas[:, :n_vars].min(), np.min(y_i))
    # z_max = max(thetas[:, :n_vars].max(), np.max(y_i))

    z_min = thetas[frame, :n_vars].min()
    z_max = thetas[frame, :n_vars].max()

    z_min = np.min([z_min, y_i.min()])
    z_max = np.max([z_max, y_i.max()])
    
    # Get the surface data for the final frame and reshape
    zs = np.array(thetas[frame][:n_vars]).reshape(n_pieces)
    
    # Create the figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$y$', fontsize=20, labelpad=20)
    ax.set_ylabel(r'$x$', fontsize=20, labelpad=20)
    ax.set_zlabel(r'$z$', fontsize=20, labelpad=20)

    # ax.set_zlim(z_min, z_max)
    ax.set_xlim(sde.u_min[1], sde.u_max[1])
    ax.set_ylim(sde.u_min[0], sde.u_max[0])
    
    # x and y ticks with fontsize of 20
    ax.tick_params(axis='both', which='major', labelsize=20)

    
    # Scatter plot of data points
    ax.scatter(u_i[:, 1], u_i[:, 0], np.ravel(y_i), label="Data points", marker="x", c="black")

    for i in range(u_i.shape[0]):
        y_pred = sde.num_y_prediction(u_i[i, :], thetas[frame, :])
        # If the measured value is below or equal to the prediction, we want the spring under the surface.
        if y_i[i, 0] <= y_pred[0][0]:
            start = np.array([u_i[i, 0], u_i[i, 1], y_i[i, 0]])
            end = np.array([u_i[i, 0], u_i[i, 1], y_pred[0][0]])
            spring_coords = spring_3d(start, end, 50, 0.1)
            ax.plot(spring_coords[1], spring_coords[0], spring_coords[2], 'gray')
    
    # Plot the surface.
    ax.plot_surface(u_x, u_y, zs, cmap='inferno', alpha=0.5, edgecolor='k', linewidth=2)
    
    # Then, plot springs that should appear on top.
    for i in range(u_i.shape[0]):
        y_pred = sde.num_y_prediction(u_i[i, :], thetas[frame, :])
        if y_i[i, 0] > y_pred[0][0]:
            start = np.array([u_i[i, 0], u_i[i, 1], y_i[i, 0]])
            end = np.array([u_i[i, 0], u_i[i, 1], y_pred[0][0]])
            spring_coords = spring_3d(start, end, 50, 0.1)
            ax.plot(spring_coords[1], spring_coords[0], spring_coords[2], 'gray')
    
    
    # Plot the surface
    # ax.plot_surface(u_x, u_y, zs, cmap='inferno', alpha=0.5, edgecolor='k', linewidth=2)

    # # Plot springs for each data point
    # for i in range(u_i.shape[0]):
    #     start = np.array([u_i[i, 0], u_i[i, 1], y_i[i, 0] if y_i.ndim > 1 else y_i[i]])
    #     y_pred = sde.num_y_prediction(u_i[i, :], thetas[frame, :])
    #     end = np.array([u_i[i, 0], u_i[i, 1], y_pred[0][0]])
    #     # spring_3d should be defined elsewhere; it returns coordinates as (x, y, z)
    #     spring_coords = spring_3d(start, end, 50, 0.1)
    #     ax.plot(spring_coords[1], spring_coords[0], spring_coords[2], 'gray')
    
    # Optionally add a time label
    # ax.text2D(0.05, 0.95, f"Time: {ts[frame]:.2f}s", transform=ax.transAxes, color="black")
    
    # Save the figure as a PDF
    plt.tight_layout()
    fig.savefig(path, format="pdf")
    return path


import numpy as np
from scipy.io.wavfile import write
import subprocess
import os

import numpy as np
from scipy.io.wavfile import write

def generate_sound_from_surface(ts, thetas, n_pieces, path="anis/sounds/spring_sound.wav", frame_skip=1):
    """
    Generates a sound representation for the time evolution of a surface based on the change
    in z-values, where greater changes produce higher pitches and smaller changes produce
    lower pitches.

    :param ts: Time steps array.
    :param ys: Solution array with shape (time_steps, variables).
    :param n_pieces: Tuple representing the grid size used for the surface.
    :param path: Output path for the sound file.
    :param frame_skip: How many frames to skip for each sound generation step.
    :return: Path to the generated sound file.
    """
    # Constants for audio generation
    sampling_rate = 44100  # Hz
    duration = len(ts) / (20 / frame_skip)  # Total duration of animation in seconds
    time_array = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

    # Initialize the audio signal array
    audio_signal = np.zeros_like(time_array)

    if isinstance(thetas, torch.Tensor):
        thetas = thetas.detach().cpu().numpy()

    # Frequency mapping based on changes: define the range
    min_frequency = 800  # Minimum frequency in Hz
    max_frequency = 1500  # Maximum frequency in Hz

    # Calculate changes in z-values between consecutive frames
    delta_zs = np.diff(thetas[:, :n_pieces[0] * n_pieces[1]], axis=0)
    delta_zs = np.vstack([delta_zs[0], delta_zs])  # Duplicate first frame for initial delta

    for i, frame in enumerate(range(0, len(ts), frame_skip)):
        frame_deltas = delta_zs[frame].reshape(n_pieces)

        for delta_z in frame_deltas.flatten():
            # Normalize change to [0, 1] range
            norm_change = (delta_z - np.min(delta_zs)) / (np.max(delta_zs) - np.min(delta_zs))
            freq = min_frequency + norm_change * (max_frequency - min_frequency)
            
            # Create sound wave for this change
            audio_signal += np.sin(2 * np.pi * freq * time_array)

    # Normalize the audio signal
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    max_val = np.iinfo(np.int16).max
    audio_signal = (audio_signal * max_val).astype(np.int16)

    # Save the audio file
    write(path, sampling_rate, audio_signal)

    return path

def create_video_3d(ts, thetas, u_i, y_i, n_pieces, sde, gif_path="anis/temp_animation.gif", sound_path="anis/temp_sound.wav", video_path="anis/video.mp4"):
    """
    Creates a 3D animation, generates an associated sound, and merges both into a video file.
    It then deletes the intermediary GIF and sound files.

    :param ts: Time steps array.
    :param ys: Solution array for the animation.
    :param u_i: Initial positions for the animation.
    :param y_i: Initial y values for the animation.
    :param n_pieces: The grid size for the 3D surface in the animation.
    :param sde: The SDE solver instance or any relevant context needed for animation.
    :param gif_path: The path to save the generated GIF.
    :param sound_path: The path to save the generated sound.
    :param video_path: The path to save the final video.
    """
    # Generate 3D animation GIF
    generalized_animation_3d(ts, thetas, u_i, y_i, n_pieces, sde, path=gif_path)
    
    # Generate sound from the animation data
    generate_sound_from_surface(ts, thetas, n_pieces, path=sound_path)
    
    # Merge GIF and sound into a video
    ffmpeg_command = [
        'ffmpeg', 
        '-i', gif_path, 
        '-i', sound_path, 
        '-c:v', 'libx264', 
        '-c:a', 'libmp3lame', 
        '-shortest', 
        video_path
    ]
    
    subprocess.run(ffmpeg_command, check=True)

    # Delete the GIF and sound files after the video is created
    os.remove(gif_path)
    os.remove(sound_path)
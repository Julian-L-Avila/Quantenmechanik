import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import ellipk, ellipe
import matplotlib.cm as cm
import os # Importar el módulo os para la gestión de carpetas

# --- Constantes Físicas ---
mu0 = 4 * np.pi * 1e-7  # Permeabilidad del vacío (T*m/A)

# --- Parámetros de las Bobinas ---
R = 0.18  # Radio de cada bobina (metros)
N = 1    # Número de espiras por bobina (simplificado a 1 para la fórmula)
a = 0.2 # Distancia del centro de las bobinas al origen (metros). Para un Helmholtz ideal, sería R/2.

# --- Función de Ayuda: Matriz de Rotación ---
def get_rotation_matrix(source_vec, dest_vec):
    source_vec = source_vec / np.linalg.norm(source_vec)
    dest_vec = dest_vec / np.linalg.norm(dest_vec)
    if np.allclose(source_vec, dest_vec):
        return np.eye(3)
    if np.allclose(source_vec, -dest_vec):
        rotation_axis = np.array([1,0,0]) if np.abs(np.dot(source_vec, [1,0,0])) < 0.9 else np.array([0,1,0])
        rotation_axis = np.cross(source_vec, rotation_axis)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.pi
    else:
        rotation_axis = np.cross(source_vec, dest_vec)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.dot(source_vec, dest_vec))
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return rotation_matrix

# --- Función para el Campo Magnético de una Sola Espira Circular en el Plano XY ---
def B_loop_xy_plane(xp, yp, zp, R, I):
    rho = np.sqrt(xp**2 + yp**2)
    if rho < 1e-10:
        Bz = mu0 * I * R**2 / (2 * (R**2 + zp**2)**(1.5))
        return np.array([0.0, 0.0, Bz])
    k_squared_num = 4 * R * rho
    k_squared_den = (R + rho)**2 + zp**2
    if k_squared_den == 0:
        return np.array([0.0, 0.0, 0.0])
    k_squared = k_squared_num / k_squared_den
    if k_squared > 1:
        k_squared = 1.0
    elif k_squared < 0:
        k_squared = 0.0
    K_val = ellipk(k_squared)
    E_val = ellipe(k_squared)
    B_rho = mu0 * I * zp / (2 * np.pi * rho * np.sqrt(k_squared_den)) * \
            ( (R**2 + rho**2 + zp**2) / ((R - rho)**2 + zp**2) * E_val - K_val )
    B_z = mu0 * I / (2 * np.pi * np.sqrt(k_squared_den)) * \
          ( (R**2 - rho**2 - zp**2) / ((R - rho)**2 + zp**2) * E_val + K_val )
    Bx = B_rho * xp / rho
    By = B_rho * yp / rho
    Bz = B_z
    return np.array([Bx, By, Bz])

# --- Definición de las Bobinas ---
I_mag = 30.0
coil1_center = np.array([0.0, a, 0.0])
coil1_normal = np.array([0.0, 1.0, 0.0])
coil1_I_mult = 1.0
coil2_center = np.array([0.0, -a, 0.0])
coil2_normal = np.array([0.0, -1.0, 0.0])
coil2_I_mult = 1.0
coil3_center = np.array([0.0, 0.0, a])
coil3_normal = np.array([0.0, 0.0, 1.0])
coil3_I_mult = 1.0
coil4_center = np.array([0.0, 0.0, -a])
coil4_normal = np.array([0.0, 0.0, -1.0])
coil4_I_mult = 1.0
coils = [
    (coil1_center, coil1_normal, coil1_I_mult * I_mag),
    (coil2_center, coil2_normal, coil2_I_mult * I_mag),
    (coil3_center, coil3_normal, coil3_I_mult * I_mag),
    (coil4_center, coil4_normal, coil4_I_mult * I_mag)
]

# --- Función para Calcular el Campo Magnético Total en un Punto ---
def calculate_total_B(point):
    total_B = np.array([0.0, 0.0, 0.0])
    for center, normal, current in coils:
        vec_to_point = point - center
        rotation_matrix_to_local = get_rotation_matrix(normal, np.array([0,0,1]))
        rotated_point = np.dot(rotation_matrix_to_local, vec_to_point)
        B_coil_local = B_loop_xy_plane(rotated_point[0], rotated_point[1], rotated_point[2], R, current)
        total_B += np.dot(rotation_matrix_to_local.T, B_coil_local)
    return total_B

# --- Parámetros de la Visualización ---
plot_lim = 0.6 # Aumentado ligeramente para el target en x=0.5
quiver_lim = 0.3
num_points = 7

x_vals_3d = np.linspace(-quiver_lim, quiver_lim, num_points)
y_vals_3d = np.linspace(-quiver_lim, quiver_lim, num_points)
z_vals_3d = np.linspace(-quiver_lim, quiver_lim, num_points)

X_3d, Y_3d, Z_3d = np.meshgrid(x_vals_3d, y_vals_3d, z_vals_3d)

# Calcular el campo magnético en toda la cuadrícula (esto solo se hace una vez)
Bx_3d, By_3d, Bz_3d = np.zeros(X_3d.shape), np.zeros(Y_3d.shape), np.zeros(Z_3d.shape)
print("Calculando campo magnético para los frames...")
for i in range(num_points):
    for j in range(num_points):
        for k in range(num_points):
            point = np.array([X_3d[i, j, k], Y_3d[i, j, k], Z_3d[i, j, k]])
            B_field = calculate_total_B(point)
            Bx_3d[i, j, k] = B_field[0]
            By_3d[i, j, k] = B_field[1]
            Bz_3d[i, j, k] = B_field[2]

magnitude_3d = np.sqrt(Bx_3d**2 + By_3d**2 + Bz_3d**2)
norm = plt.Normalize(vmin=magnitude_3d.min(), vmax=magnitude_3d.max())
colors = cm.jet(norm(magnitude_3d))

# --- Configuración para la Generación de Imágenes Individuales ---
output_folder = 'frames_campo_magnetico_haz_x' # Nueva carpeta para reflejar el cambio
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

num_frames = 100
elevation_angle = 10

print(f"Generando {num_frames} imágenes en la carpeta '{output_folder}'...")

for i in range(num_frames):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    azim_angle = i * (360 / num_frames)
    ax.view_init(elev=elevation_angle, azim=azim_angle)

    ax.set_xlim([-plot_lim, 0.7])
    ax.set_ylim([-plot_lim, plot_lim])
    ax.set_zlim([-plot_lim, plot_lim])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Campo Magnético 3D de Bobinas (Azimut: {azim_angle:.1f}°)')

    ax.set_box_aspect([1,1,1])

    # Dibujar los vectores de campo magnético
    ax.quiver(X_3d, Y_3d, Z_3d, Bx_3d, By_3d, Bz_3d,
              colors=colors.reshape(-1, 4),
              length=0.07,
              normalize=True,
              arrow_length_ratio=0.3,
              linewidth=0.7)

    # Dibujar las bobinas
    coil_theta = np.linspace(0, 2 * np.pi, 100)
    coil_xy_plane_points = np.array([R * np.cos(coil_theta), R * np.sin(coil_theta), np.zeros_like(coil_theta)])

    for center, normal, current in coils:
        rotation_matrix_to_global = get_rotation_matrix(np.array([0,0,1]), normal)
        coil_points_global = np.dot(rotation_matrix_to_global, coil_xy_plane_points) + center[:, np.newaxis]
        ax.plot(coil_points_global[0], coil_points_global[1], coil_points_global[2],
                color='red' if normal[2] != 0 else 'blue',
                linewidth=2)

    # Graficar el origen como referencia
    ax.plot([0], [0], [0], 'kx', markersize=10)

    # --- CAMBIOS: HAZ DE ELECTRONES EN EL EJE X Y TARGET EN EL PLANO YZ ---

    # --- AÑADIR LA BARRA DE COLOR ---
    # Crear un ScalarMappable para la barra de color
    sm = cm.ScalarMappable(cmap=cm.jet, norm=norm)
    sm.set_array(magnitude_3d) # Asignar los datos de magnitud
    
    # Añadir la barra de color a la figura
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.1)
    cbar.set_label('Magnitud del Campo Magnético (T)')


    # 1. Haz de Electrones (línea verde fosforescente a lo largo del eje X)
    electron_beam_x = np.linspace(-0.5, 0.7, 50) # Rango de X para el haz
    electron_beam_y = np.zeros_like(electron_beam_x) # Y en 0
    electron_beam_z = np.zeros_like(electron_beam_x) # Z en 0
    ax.plot(electron_beam_x, electron_beam_y, electron_beam_z,
            color='#39FF14',   # Verde fosforescente (neón)
            linestyle='-',    # Línea discontinua
            linewidth=3,       # Grosor de la línea
            label='Haz de Electrones')

    # 2. Target Fluorescente (circunferencia en x=0.5, paralela al plano YZ)
    target_radius = 0.2
    target_x_pos = 0.7 # Posición en X para el target
    target_theta = np.linspace(0, 2 * np.pi, 100)
    target_y = target_radius * np.cos(target_theta) # Y variando con el círculo
    target_z = target_radius * np.sin(target_theta) # Z variando con el círculo
    target_x = np.full_like(target_y, target_x_pos) # Todos los puntos a la misma posición X

    ax.plot(target_x, target_y, target_z,
            color='cyan',      # Color de contraste (azul cian)
            linestyle='-',
            linewidth=2,
            label='Target Fluorescente')
    # Añadir un punto central en el target para mayor claridad
    ax.scatter([target_x_pos], [0], [0], color='cyan', marker='o', s=50) # Punto central del círculo

    # --- Fin de CAMBIOS ---

    # Re-añadir la leyenda para incluir el haz y el target
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(l)] for l in unique_labels]
    ax.legend(unique_handles, unique_labels, loc='upper left')

    # Guardar la imagen
    filename = os.path.join(output_folder, f'frame_azim_{azim_angle:05.1f}.png')
    plt.savefig(filename, dpi=150)

    plt.close(fig)

    if (i + 1) % 10 == 0 or (i + 1) == num_frames:
        print(f"  Guardado frame {i + 1}/{num_frames} ({azim_angle:.1f} grados)")

print(f"\n¡Se han generado {num_frames} imágenes con haz en X y target en YZ en la carpeta '{output_folder}'!")

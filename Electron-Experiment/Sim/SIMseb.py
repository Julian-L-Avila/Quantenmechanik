import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import ellipk, ellipe
import matplotlib.cm as cm
import matplotlib.animation as animation # Importar el módulo de animación

# --- Copia de las funciones y parámetros necesarios (o asegúrate de que estén definidos en celdas anteriores) ---
# Si ejecutas esto en la misma sesión de notebook, no necesitas copiar todo de nuevo,
# pero lo incluyo para que el bloque sea autocontenido si lo ejecutas de forma aislada.

# --- Physical Constants ---
mu0 = 4 * np.pi * 1e-7  # Permeability of vacuum (T*m/A)

# --- Coil Parameters ---
R = 0.1  # Radius of each coil (meters)
N = 4    # Number of turns per coil (simplified to 1 turn for the formula)
a = 0.15 # Distance of coil centers from the origin (meters). For an ideal 2-coil Helmholtz, this would be R/2.

# --- Helper Function: Rotation Matrix from Source Vector to Destination Vector ---
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

# --- Function for Magnetic Field of a Single Circular Loop in XY Plane ---
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

# --- Definition of the Coils ---
I_mag = 3.0
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

# --- Function to Calculate Total Magnetic Field at a Point ---
def calculate_total_B(point):
    total_B = np.array([0.0, 0.0, 0.0])
    for center, normal, current in coils:
        vec_to_point = point - center
        rotation_matrix_to_local = get_rotation_matrix(normal, np.array([0,0,1]))
        rotated_point = np.dot(rotation_matrix_to_local, vec_to_point)
        B_coil_local = B_loop_xy_plane(rotated_point[0], rotated_point[1], rotated_point[2], R, current)
        total_B += np.dot(rotation_matrix_to_local.T, B_coil_local)
    return total_B

# --- Parámetros de la Visualización (los mismos que en el código anterior) ---
plot_lim = 0.3
quiver_lim = 0.2
num_points = 10

x_vals_3d = np.linspace(-quiver_lim, quiver_lim, num_points)
y_vals_3d = np.linspace(-quiver_lim, quiver_lim, num_points)
z_vals_3d = np.linspace(-quiver_lim, quiver_lim, num_points)

X_3d, Y_3d, Z_3d = np.meshgrid(x_vals_3d, y_vals_3d, z_vals_3d)

# Calcular el campo magnético en toda la cuadrícula (esto solo se hace una vez)
Bx_3d, By_3d, Bz_3d = np.zeros(X_3d.shape), np.zeros(Y_3d.shape), np.zeros(Z_3d.shape)
print("Calculando campo magnético para el GIF...")
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
colors = cm.jet(norm(magnitude_3d)) # Puedes usar 'viridis', 'plasma', etc.

# --- Configuración para la Animación ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Establecer los límites de los ejes para el plot completo
ax.set_xlim([-plot_lim, plot_lim])
ax.set_ylim([-plot_lim, plot_lim])
ax.set_zlim([-plot_lim, plot_lim])

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Campo Magnético 3D de Bobinas')
ax.set_box_aspect([1,1,1])

# Dibujar las bobinas (estáticas en cada frame)
coil_theta = np.linspace(0, 2 * np.pi, 100)
coil_xy_plane_points = np.array([R * np.cos(coil_theta), R * np.sin(coil_theta), np.zeros_like(coil_theta)])

for center, normal, current in coils:
    rotation_matrix_to_global = get_rotation_matrix(np.array([0,0,1]), normal)
    coil_points_global = np.dot(rotation_matrix_to_global, coil_xy_plane_points) + center[:, np.newaxis]
    ax.plot(coil_points_global[0], coil_points_global[1], coil_points_global[2],
            color='red' if normal[2] != 0 else 'blue',
            linewidth=2) # Quité el label para evitar duplicados en cada frame

ax.plot([0], [0], [0], 'kx', markersize=10) # Origen

# Inicializar el objeto quiver para la animación
# Necesitamos una referencia a los objetos que se actualizarán
quiver_plot = ax.quiver(X_3d, Y_3d, Z_3d, Bx_3d, By_3d, Bz_3d,
                        colors=colors.reshape(-1, 4),
                        length=0.04,
                        normalize=True,
                        arrow_length_ratio=0.2,
                        linewidth=0.7)

# Opcional: añadir la barra de color (generalmente no se anima)
m = cm.ScalarMappable(cmap='jet', norm=norm)
m.set_array(magnitude_3d)
fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, label='Magnitud del Campo (T)')

# --- Función de Actualización para la Animación ---
def update_rotation(frame):
    # La elevación es fija, el azimut cambia
    ax.view_init(elev=30, azim=frame * 0.36) # Rota 360 grados en 100 frames (3.6 grados por frame)
    return quiver_plot, # Retorna los objetos que se han modificado

# Crear la animación
# frames: número de frames para la rotación completa
# interval: tiempo entre frames en ms
# blit: optimización (True si no hay cambios en el fondo del gráfico)
print("Generando GIF... esto puede tomar un tiempo.")
ani = animation.FuncAnimation(fig, update_rotation, frames=1000, interval=100, blit=False)

# Guardar la animación como un GIF
# Necesitarás 'Pillow' instalado: pip install Pillow
ani.save('campo_magnetico_rotacion.gif', writer='pillow', dpi=150) # dpi controla la resolución del GIF

plt.close(fig) # Cierra la figura para evitar mostrarla estática al final
print("GIF 'campo_magnetico_rotacion.gif' generado con éxito.")

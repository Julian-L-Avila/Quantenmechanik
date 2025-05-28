import numpy as np

def calculate_normal_mode_frequencies(k1, k2, cl, dl, y_cm1, y_cm2, y_cm3, m1, m2, m3, I1, I2, I3, g):
    """
    Berechnet die Eigenfrequenzen eines gekoppelten Oszillatorsystems.

    Args:
        k1, k2 (float): Federkonstanten.
        cl, dl (float): Längenparameter.
        y_cm1, y_cm2, y_cm3 (float): Positionen der Massenschwerpunkte in y-Richtung.
        m1, m2, m3 (float): Massen.
        I1, I2, I3 (float): Trägheitsmomente.
        g (float): Erdbeschleunigung.

    Returns:
        numpy.ndarray: Ein Array mit den Eigenfrequenzen in Hz.
                       Gibt None zurück, wenn ungültige Eigenwerte (nicht-negativ nach Negation) gefunden werden.
    """

    # Definiere die Elemente der Matrix A
    # Beachte: Die Gleichungen sind in der Form \ddot{theta} = A * theta
    # Die Eigenwerte von A sind -omega^2

    A11 = -((cl**2 * k1) + (y_cm1 * m1 * g)) / I1
    A12 = (k1 * cl**2) / I1
    A13 = 0.0  # Da theta_3 in der ersten Gleichung nicht vorkommt

    A21 = (k1 * cl**2) / I2
    A22 = -((k2 * dl**2) + (k1 * cl**2) + (y_cm2 * m2 * g)) / I2
    A23 = (k2 * dl**2) / I2

    A31 = 0.0  # Da theta_1 in der dritten Gleichung nicht vorkommt
    A32 = (k2 * dl**2) / I3
    A33 = -((k2 * dl**2) + (y_cm3 * m3 * g)) / I3

    # Erstelle die Matrix A
    A_matrix = np.array([
        [A11, A12, A13],
        [A21, A22, A23],
        [A31, A32, A33]
    ])

    print("Systemmatrix A:")
    print(A_matrix)

    # Berechne die Eigenwerte der Matrix A
    # Eigenwerte lambda = -omega^2
    eigenvalues = np.linalg.eigvals(A_matrix)
    print("\nEigenwerte (lambda = -omega^2):")
    print(eigenvalues)

    # Berechne omega^2 = -lambda
    # Stelle sicher, dass die Werte nicht-negativ sind, bevor die Wurzel gezogen wird
    omega_squared_values = -eigenvalues

    # Filtere auf physikalisch sinnvolle (nicht-negative) omega^2 Werte
    valid_omega_squared = []
    for val in omega_squared_values:
        if val >= 0:
            valid_omega_squared.append(val)
        elif np.isclose(val, 0): # Behandle numerisch sehr kleine negative Zahlen als Null
             valid_omega_squared.append(0.0)
        else:
            print(f"\nAchtung: Negativer Wert für omega^2 gefunden ({val}). Dies deutet auf eine Instabilität im System hin oder auf einen Fehler in den Parametern.")
            # Du könntest hier entscheiden, ob du einen Fehler auslösen oder mit den imaginären Frequenzen fortfahren möchtest.
            # Für dieses Beispiel geben wir None zurück, wenn eine Instabilität vorliegt.
            # return None # Oder behandle es anders, z.B. durch Rückgabe von np.nan für diese Frequenz

    if not valid_omega_squared or len(valid_omega_squared) != len(omega_squared_values):
        print("\nEinige oder alle omega^2 Werte sind negativ, was auf ein instabiles System hindeutet.")
        # Je nach Bedarf kann man hier auch nur die validen Frequenzen zurückgeben
        # oder eine Fehlermeldung ausgeben.
        # Wir geben für dieses Beispiel die berechneten omega^2 Werte (auch negative) zurück,
        # um die Diagnose zu ermöglichen, aber die Frequenzen in Hz werden nur für positive omega^2 berechnet.

    # Berechne die Kreisfrequenzen omega (rad/s)
    # np.sqrt wird hier elementweise angewendet
    # Wir nehmen den Realteil, falls durch numerische Ungenauigkeiten kleine Imaginärteile entstehen bei Werten nahe Null
    angular_frequencies_rad_s = np.sqrt(np.maximum(0, omega_squared_values.astype(complex))).real


    print("\nKreisfrequenzen (omega in rad/s):")
    print(angular_frequencies_rad_s)

    # Wandle die Kreisfrequenzen in Frequenzen in Hertz (Hz) um
    # f = omega / (2 * pi)
    frequencies_hz = angular_frequencies_rad_s / (2 * np.pi)

    print("\nEigenfrequenzen (f in Hz):")
    print(frequencies_hz)

    return frequencies_hz

# --- Beispielhafte Verwendung ---
# Bitte ersetze diese Werte durch deine tatsächlichen Parameter!
# Physikalische Konstanten
g_const = 9.81  # m/s^2

# Parameter des Systems (Beispielwerte)
k1_val = 3.32   # N/m oder N/rad, je nach Definition von cl
k2_val = 3.04   # N/m oder N/rad
cl_val = 0.046 * 6    # m
dl_val = 0.046 * 6    # m
y_cm1_val = 0.142 # m (angenommen positiv, wenn der Schwerpunkt eine rückstellende Kraft erzeugt)
y_cm2_val = 0.28 # m (kann auch negativ sein, was die Natur des g-Terms ändert)
y_cm3_val = 0.14 # m
m1_val = 0.6008    # kg
m2_val = 1.2163    # kg
m3_val = 0.6018    # kg
I1_val = m1_val * (0.28) ** 2 / 3.0    # kg*m^2
I2_val = m2_val * (0.56) ** 2 / 3.0   # kg*m^2
I3_val = m3_val * (0.28) ** 2 / 3.0   # kg*m^2

# Rufe die Funktion auf, um die Frequenzen zu berechnen
normal_frequencies = calculate_normal_mode_frequencies(
    k1=k1_val, k2=k2_val, cl=cl_val, dl=dl_val,
    y_cm1=y_cm1_val, y_cm2=y_cm2_val, y_cm3=y_cm3_val,
    m1=m1_val, m2=m2_val, m3=m3_val,
    I1=I1_val, I2=I2_val, I3=I3_val,
    g=g_const
)

if normal_frequencies is not None:
    print("\n------------------------------------")
    print("Berechnete Eigenfrequenzen in Hz:")
    for i, freq in enumerate(normal_frequencies):
        print(f"Mode {i+1}: {freq:.4f} Hz")
    print("------------------------------------")
else:
    print("\nDie Berechnung der Eigenfrequenzen war nicht erfolgreich aufgrund von Instabilitäten.")

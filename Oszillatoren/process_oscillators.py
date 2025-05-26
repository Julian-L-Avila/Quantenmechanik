import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def parse_tsv(file_path):
    """Parses a TSV file and returns the data."""
    time, angle1, angle2, angle3 = [], [], [], []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                try:
                    values = line.strip().split('\t')
                    if len(values) < 4:
                        print(f"Warning: Skipping line due to insufficient columns: {line.strip()}")
                        continue
                    time.append(float(values[0]))
                    angle1.append(float(values[1]))
                    angle2.append(float(values[2]))
                    angle3.append(float(values[3]))
                except ValueError:
                    print(f"Warning: Skipping line due to non-numeric data: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return [], [], [], []
    except Exception as e:
        print(f"An unexpected error occurred while parsing {file_path}: {e}")
        return [], [], [], []
    return time, angle1, angle2, angle3

def calculate_fft(time_data, angle_data):
    """Calculates the Fast Fourier Transform (FFT) of the angle data."""
    if not angle_data or len(time_data) < 2:
        print("Warning: Not enough data points to perform FFT.")
        return None, None, None

    # Calculate sampling interval
    dt = np.mean(np.diff(time_data))
    if dt == 0:
        print("Warning: Sampling interval is zero. Cannot perform FFT.")
        return None, None, None

    n_samples = len(angle_data)

    # Perform FFT
    fft_result = np.fft.fft(angle_data)
    frequencies = np.fft.fftfreq(n_samples, d=dt)

    # Consider only positive frequencies
    positive_freq_indices = np.where(frequencies >= 0)
    positive_frequencies = frequencies[positive_freq_indices]
    positive_fft_result = fft_result[positive_freq_indices]

    # Calculate magnitudes
    fft_magnitudes = np.abs(positive_fft_result)

    # Find the main frequency (ignoring DC component at index 0 if it's dominant)
    # We search in the positive frequencies.
    # If the DC component (at index 0 of positive_frequencies) is the max,
    # we check if there's another significant peak.
    # For simplicity, we start search from index 1 of the positive frequencies.
    if len(positive_frequencies) > 1:
        # Search for peak starting from the first non-DC component
        peak_index_in_positive = np.argmax(fft_magnitudes[1:]) + 1
        main_frequency = positive_frequencies[peak_index_in_positive]
    elif len(positive_frequencies) == 1: # Only DC component exists
        main_frequency = positive_frequencies[0]
    else: # Should not happen if previous checks are correct
        main_frequency = None


    return positive_frequencies, fft_magnitudes, main_frequency

def plot_angles_vs_time(time_data, angle1_data, angle2_data, angle3_data, filename):
    """Plots the angles vs. time and returns the figure object."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots()

    ax.plot(time_data, angle1_data, color='Plum', label='Angle 1')
    ax.plot(time_data, angle2_data, color='Green', label='Angle 2')
    ax.plot(time_data, angle3_data, color='RoyalBlue', label='Angle 3')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)") # Assuming radians as it's common in physics/engineering
    ax.set_title(f"Angle Data vs. Time for {filename}")
    ax.legend()
    
    # Consider adjusting font sizes for a Springer-like feel if needed
    # plt.rcParams.update({'font.size': 10}) # Example
    
    return fig

def plot_spectrum(freq_data_list, spectrum_data_list, main_frequencies, filename):
    """Plots the spectrum for all angles and returns the figure object."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots()
    colors = ['Plum', 'Green', 'RoyalBlue']
    labels = ['Angle 1 Spectrum', 'Angle 2 Spectrum', 'Angle 3 Spectrum']

    for i in range(len(freq_data_list)):
        if freq_data_list[i] is not None and spectrum_data_list[i] is not None:
            ax.plot(freq_data_list[i], spectrum_data_list[i], color=colors[i], label=labels[i])
            if main_frequencies[i] is not None:
                ax.axvline(main_frequencies[i], color=colors[i], linestyle='--', label=f'Main Freq Angle {i+1}: {main_frequencies[i]:.2f} Hz')

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Frequency Spectrum for {filename}")
    ax.legend()
    
    # Consider adjusting x-axis limits if spectra are too wide or too narrow
    # Example: ax.set_xlim([0, max_interesting_frequency])
    
    return fig

if __name__ == "__main__":
    data_directory = "Oszillatoren/Data/"
    summary_file_path = "Oszillatoren/frequency_summary.txt"
    pdf_output_dir = "Oszillatoren/Analysis_Results/"
    os.makedirs(pdf_output_dir, exist_ok=True)

    # Initialize or clear the summary file and write a header
    try:
        with open(summary_file_path, 'w') as summary_file:
            summary_file.write("# Filename - Main Frequency Angle 1, Main Frequency Angle 2, Main Frequency Angle 3\n")
    except IOError as e:
        print(f"Error: Could not write to summary file {summary_file_path}: {e}")
        exit() 

    if not os.path.isdir(data_directory):
        print(f"Error: Data directory '{data_directory}' not found.")
        exit() 

    for filename in os.listdir(data_directory):
        if filename.endswith(".tsv"):
            file_path = os.path.join(data_directory, filename)
            base_filename, _ = os.path.splitext(filename)
            pdf_file_path = os.path.join(pdf_output_dir, base_filename + "_analysis.pdf")
            
            print(f"Processing {file_path}...")

            time, angle1, angle2, angle3 = parse_tsv(file_path)

            if not time: 
                print(f"Could not process {filename} due to parsing errors. Skipping PDF generation.")
                continue 

            # Calculate FFT for each angle
            freq_data_list = []
            spectrum_data_list = []
            main_frequencies_list = []

            fft_results = [
                calculate_fft(time, angle1),
                calculate_fft(time, angle2),
                calculate_fft(time, angle3)
            ]

            valid_fft_for_all_angles = True
            for freq, mag, main_freq in fft_results:
                if freq is None or mag is None or main_freq is None:
                    valid_fft_for_all_angles = False
                    break
                freq_data_list.append(freq)
                spectrum_data_list.append(mag)
                main_frequencies_list.append(main_freq)

            # Update summary file
            if valid_fft_for_all_angles:
                output_line = f"{filename} - {main_frequencies_list[0]:.4f}, {main_frequencies_list[1]:.4f}, {main_frequencies_list[2]:.4f}\n"
                try:
                    with open(summary_file_path, 'a') as summary_file:
                        summary_file.write(output_line)
                except IOError as e:
                    print(f"Error: Could not append to summary file {summary_file_path}: {e}")
            else:
                print(f"FFT calculation failed for one or more angles in {filename}. Results for this file will not be added to summary.")

            # Plotting and PDF generation
            if valid_fft_for_all_angles:
                fig_angles = plot_angles_vs_time(time, angle1, angle2, angle3, base_filename)
                fig_spectrum = plot_spectrum(freq_data_list, spectrum_data_list, main_frequencies_list, base_filename)

                if fig_angles and fig_spectrum:
                    try:
                        with PdfPages(pdf_file_path) as pdf:
                            pdf.savefig(fig_angles)
                            pdf.savefig(fig_spectrum)
                        print(f"Saved PDF: {pdf_file_path}")
                    except Exception as e:
                        print(f"Error saving PDF {pdf_file_path}: {e}")
                    finally:
                        plt.close(fig_angles) 
                        plt.close(fig_spectrum)
                else:
                    print(f"Skipping PDF for {filename} due to plotting error (figures not generated).")
                    if fig_angles: plt.close(fig_angles)
                    if fig_spectrum: plt.close(fig_spectrum)
            else:
                print(f"Skipping PDF generation for {filename} due to FFT calculation errors.")

    print(f"Processing complete. Summary saved to {summary_file_path}. PDFs saved in {pdf_output_dir}")

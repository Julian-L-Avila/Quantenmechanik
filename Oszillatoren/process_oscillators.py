import pandas
import os
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# Define the input files and their time shifts
# Format: (filename, time_shift)
# File paths are relative to the script's location (Oszillatoren/)
INPUT_FILES_DATA = [
    ("Data/111_11.tsv", 16.25),
    ("Data/010_15.tsv", 0.3),
    ("Data/010_26.tsv", 10.5),
    ("Data/101_26.tsv", 3.0),
    ("Data/010_66.tsv", 6.6),
    ("Data/001_66.tsv", 2.6),
]

COLUMN_NAMES = ['time', 'angle1', 'angle2', 'angle3']
ANGLE_COLUMNS = ['angle1', 'angle2', 'angle3']

def load_and_process_file(relative_file_path: str, time_shift: float) -> pandas.DataFrame | None:
    """
    Loads a TSV file, adjusts the time column, and filters based on the adjusted time.

    Args:
        relative_file_path: The path to the TSV file, relative to the script's directory (e.g., "Data/filename.tsv").
        time_shift: The time shift value to subtract from the 'time' column.

    Returns:
        A pandas DataFrame with the processed data, or None if the file is not found.
    """
    full_file_path = "" # Initialize in case of early error
    try:
        current_script_path = os.path.realpath(__file__)
        script_directory = os.path.dirname(current_script_path)

        # Path construction should be robust.
        # relative_file_path is like "Data/111_11.tsv"
        # script_directory is the absolute path to the directory containing this script.
        # So, this join gives the absolute path to the data file.
        full_file_path = os.path.join(script_directory, relative_file_path)

        df = pandas.read_csv(
            full_file_path,
            delimiter='\t',
            names=COLUMN_NAMES,
            comment='#',
            header=None
        )

        for col in COLUMN_NAMES:
            df[col] = pandas.to_numeric(df[col], errors='coerce')
        df.dropna(subset=COLUMN_NAMES, inplace=True)

        df['time'] = df['time'] - time_shift
        df_processed = df[df['time'] > 0].copy()

        return df_processed

    except FileNotFoundError:
        err_msg = f"Error: File not found for '{relative_file_path}'."
        if full_file_path: # Add attempted path if available
            err_msg += f" (Attempted path: {full_file_path})"
        print(err_msg)
        return None
    except Exception as e:
        print(f"An error occurred while processing {relative_file_path}: {e}")
        return None

def calculate_fft(processed_dataframe: pandas.DataFrame) -> dict:
    fft_results = {}
    time_data = processed_dataframe['time'].to_numpy()

    for angle_col_name in ANGLE_COLUMNS:
        if angle_col_name not in processed_dataframe.columns:
            print(f"Warning: Column {angle_col_name} not found in DataFrame. Skipping FFT for this column.")
            fft_results[angle_col_name] = {'frequency': np.array([]), 'amplitude': np.array([])}
            continue

        angle_data = processed_dataframe[angle_col_name].to_numpy()
        N = angle_data.size

        if N < 2:
            print(f"Warning: Insufficient data points (N={N}) for FFT on {angle_col_name}. Skipping.")
            fft_results[angle_col_name] = {'frequency': np.array([]), 'amplitude': np.array([])}
            continue

        if time_data.size < 2:
             print(f"Warning: Insufficient time data points to calculate dt for {angle_col_name}. Skipping.")
             fft_results[angle_col_name] = {'frequency': np.array([]), 'amplitude': np.array([])}
             continue

        dt = time_data[1] - time_data[0]
        if dt <= 0:
            print(f"Warning: Non-positive sampling interval dt={dt:.4f} for {angle_col_name}. Skipping FFT.")
            fft_results[angle_col_name] = {'frequency': np.array([]), 'amplitude': np.array([])}
            continue

        yf = fft(angle_data)
        xf = fftfreq(N, dt)
        mask = xf > 0
        fft_results[angle_col_name] = {
            'frequency': xf[mask],
            'amplitude': np.abs(yf[mask])
        }
    return fft_results

def find_top_n_peaks(frequencies: np.ndarray, amplitudes: np.ndarray, n: int = 3, height_threshold: float = 0.001) -> list:
    if frequencies.size == 0 or amplitudes.size == 0 or frequencies.size != amplitudes.size:
        return []
    peak_indices, _ = find_peaks(amplitudes, height=height_threshold)
    if peak_indices.size == 0:
        return []
    peak_amplitudes = amplitudes[peak_indices]
    peak_frequencies = frequencies[peak_indices]
    detected_peaks = [{'frequency': f, 'amplitude': a} for f, a in zip(peak_frequencies, peak_amplitudes)]
    sorted_peaks = sorted(detected_peaks, key=lambda x: x['amplitude'], reverse=True)
    return sorted_peaks[:n]

def transform_filename_parts(original_basename: str) -> str:
    name_without_suffix, _ = os.path.splitext(original_basename)
    parts = name_without_suffix.split('_')
    if len(parts) == 2:
        p1 = parts[0]
        p2 = parts[1]
        reversed_p1 = p1[::-1]
        reversed_p2 = p2[::-1]
        new_base = f"{reversed_p2}-{reversed_p1}"
        return new_base
    else:
        print(f"Warning: Filename '{original_basename}' (base: '{name_without_suffix}') does not match expected 'part1_part2.tsv' format. Cannot transform.")
        return name_without_suffix

def save_peaks_data(original_file_path: str,
                    top_peaks_for_file: dict,
                    output_dir: str):
    try:
        original_basename = os.path.basename(original_file_path)
        transformed_base = transform_filename_parts(original_basename)
        output_filename = transformed_base + '_peaks.tsv'
        full_output_path = os.path.join(output_dir, output_filename)

        rows_for_df = []
        for angle_col in ANGLE_COLUMNS:
            peaks_list = top_peaks_for_file.get(angle_col, [])
            for i, peak_data in enumerate(peaks_list):
                rank = i + 1
                rows_for_df.append({
                    'angle': angle_col,
                    'peak_rank': rank,
                    'frequency': peak_data['frequency'],
                    'amplitude': peak_data['amplitude']
                })
        if rows_for_df:
            peaks_df = pandas.DataFrame(rows_for_df)
            peaks_df = peaks_df[['angle', 'peak_rank', 'frequency', 'amplitude']]
            peaks_df.to_csv(full_output_path, sep='\t', index=False, float_format='%.6e', na_rep='NaN')
            print(f"Saved peak data for {original_file_path} to {full_output_path}")
        else:
            print(f"No peak data to save for {original_file_path}")
    except Exception as e:
        print(f"Error saving peak data for {original_file_path}: {e}")

def save_spectrum_data(original_file_path: str,
                       fft_results_for_file: dict,
                       output_dir: str):
    try:
        original_basename = os.path.basename(original_file_path)
        transformed_base = transform_filename_parts(original_basename)
        output_filename = transformed_base + '_spectrum.tsv'
        full_output_path = os.path.join(output_dir, output_filename)

        df_data = {}
        reference_angle_data = None
        for angle in ANGLE_COLUMNS:
            if angle in fft_results_for_file and \
               fft_results_for_file[angle]['frequency'].size > 0:
                reference_angle_data = fft_results_for_file[angle]
                break

        if reference_angle_data is None:
            print(f"Skipping save for {original_file_path}: No frequency data found for any angle (full spectrum).")
            return

        frequencies = reference_angle_data['frequency']
        df_data['frequency'] = frequencies
        num_freq_points = len(frequencies)

        for angle_col in ANGLE_COLUMNS:
            angle_data = fft_results_for_file.get(angle_col)
            if angle_data and angle_data['amplitude'].size == num_freq_points:
                df_data[angle_col] = angle_data['amplitude']
            else:
                df_data[angle_col] = np.full(num_freq_points, np.nan)
                if angle_data:
                     print(f"Warning for {original_file_path}, {angle_col} (full spectrum): Mismatched data length. Expected {num_freq_points}, got {angle_data['amplitude'].size}. Filling with NaN.")

        spectrum_df = pandas.DataFrame(df_data)
        ordered_columns = ['frequency'] + [col for col in ANGLE_COLUMNS if col in spectrum_df.columns]
        spectrum_df = spectrum_df[ordered_columns]

        spectrum_df.to_csv(full_output_path, sep='\t', index=False, float_format='%.6e', na_rep='NaN')
        print(f"Saved spectrum data for {original_file_path} to {full_output_path}")
    except Exception as e:
        print(f"Error saving spectrum data for {original_file_path}: {e}")

if __name__ == "__main__":
    current_script_abs_path = os.path.abspath(__file__)
    script_parent_dir = os.path.dirname(current_script_abs_path)
    output_directory = os.path.join(script_parent_dir, 'Data', 'Processed_Spectra')
    os.makedirs(output_directory, exist_ok=True)
    print(f"Output directory set to: {os.path.abspath(output_directory)}")

    processed_and_normalized_data_for_saving = []

    if not INPUT_FILES_DATA:
        print("No input files defined.")
    else:
        print(f"Starting processing for {len(INPUT_FILES_DATA)} file(s)...")
        for file_path, time_shift_val in INPUT_FILES_DATA:
            print(f"\nProcessing file: {file_path} with time shift: {time_shift_val} s")
            processed_df = load_and_process_file(file_path, time_shift_val)

            if processed_df is not None and not processed_df.empty:
                if len(processed_df) >= 2:
                    print(f"Successfully loaded and processed {file_path}. DataFrame rows: {len(processed_df)}")
                    current_fft_results = calculate_fft(processed_df)
                    print(f"FFT calculation done for {file_path}.")

                    file_max_amplitude = 0.0
                    for angle_col_name_iter in ANGLE_COLUMNS:
                        if angle_col_name_iter in current_fft_results:
                            angle_data = current_fft_results[angle_col_name_iter]
                            if angle_data['amplitude'].size > 0:
                                current_angle_max = np.max(angle_data['amplitude'])
                                if current_angle_max > file_max_amplitude:
                                    file_max_amplitude = current_angle_max
                    print(f"File: {file_path}, File-specific Max Amplitude: {file_max_amplitude:.4f}")

                    file_normalized_fft_data = {}
                    for angle_col_name_norm, angle_spec_data in current_fft_results.items():
                        original_amplitudes = angle_spec_data['amplitude']
                        frequencies = angle_spec_data['frequency']
                        if file_max_amplitude > 0:
                            normalized_amplitudes = original_amplitudes / file_max_amplitude
                        else:
                            normalized_amplitudes = original_amplitudes
                        file_normalized_fft_data[angle_col_name_norm] = {
                            'frequency': frequencies,
                            'amplitude': normalized_amplitudes
                        }

                    print(f"Verification of normalized amplitudes for {file_path}:")
                    for angle_verify_col in ANGLE_COLUMNS:
                        if angle_verify_col in file_normalized_fft_data:
                            norm_amps_verify = file_normalized_fft_data[angle_verify_col]['amplitude']
                            if norm_amps_verify.size > 0:
                                max_norm_amp_verify = np.max(norm_amps_verify)
                                print(f"  Max normalized amplitude for {angle_verify_col}: {max_norm_amp_verify:.4f}")
                            else:
                                print(f"  No normalized amplitude data for {angle_verify_col} to verify (empty array).")
                        else:
                             print(f"  Angle column {angle_verify_col} not found in file_normalized_fft_data for {file_path}.")

                    top_peaks_data_for_file = {}
                    for angle_col_peak in ANGLE_COLUMNS:
                        if angle_col_peak in file_normalized_fft_data:
                            current_frequencies = file_normalized_fft_data[angle_col_peak]['frequency']
                            current_normalized_amplitudes = file_normalized_fft_data[angle_col_peak]['amplitude']
                            if current_normalized_amplitudes.size > 0 and current_frequencies.size > 0 :
                                top_peaks = find_top_n_peaks(current_frequencies, current_normalized_amplitudes, n=3)
                                top_peaks_data_for_file[angle_col_peak] = top_peaks
                            else:
                                top_peaks_data_for_file[angle_col_peak] = []
                        else:
                            top_peaks_data_for_file[angle_col_peak] = []

                    print(f"Top peaks for {file_path}:")
                    for angle_col_print, peaks_print in top_peaks_data_for_file.items():
                        print(f"  {angle_col_print}:")
                        if peaks_print:
                            for peak in peaks_print:
                                print(f"    Freq: {peak['frequency']:.2f} Hz, Norm_Amp: {peak['amplitude']:.4f}")
                        else:
                            print("    No peaks found or data was empty.")

                    processed_and_normalized_data_for_saving.append({
                        'file_name': file_path,
                        'normalized_fft_results': file_normalized_fft_data,
                        'top_peaks': top_peaks_data_for_file
                    })
                    print(f"File-specific normalization, verification, and peak finding done for {file_path}.")
                else:
                    print(f"Skipping FFT, normalization, verification and peak finding for {file_path}: Not enough data points (less than 2) after processing.")
            else:
                print(f"Skipping {file_path}: Failed to load or DataFrame is empty.")

        print(f"\nFinished processing all files. {len(processed_and_normalized_data_for_saving)} file(s) had successful FFT and normalization.")

        print("\nStarting to save file-specific normalized spectra...")
        if not processed_and_normalized_data_for_saving:
             print("No normalized data to save.")
        else:
            for item in processed_and_normalized_data_for_saving:
                original_input_path = item['file_name']
                normalized_spectrum_to_save = item['normalized_fft_results']
                top_peaks_to_save = item['top_peaks']
                save_spectrum_data(original_input_path, normalized_spectrum_to_save, output_directory)
                save_peaks_data(original_input_path, top_peaks_to_save, output_directory)
            print("Finished saving file-specific normalized spectra and peak data.")

            if processed_and_normalized_data_for_saving:
                if processed_and_normalized_data_for_saving:
                    first_processed_file_data = processed_and_normalized_data_for_saving[0]
                    file_name_for_print = first_processed_file_data['file_name']
                    angle1_norm_fft = first_processed_file_data['normalized_fft_results'].get('angle1')
                    if angle1_norm_fft and angle1_norm_fft['frequency'].size > 0:
                        print(f"\nFile-Specific Normalized FFT results for angle1 of ({file_name_for_print}) (first 5 points - for demonstration):")
                        for i in range(min(5, angle1_norm_fft['frequency'].size)):
                            freq = angle1_norm_fft['frequency'][i]
                            norm_amp = angle1_norm_fft['amplitude'][i]
                            print(f"Freq: {freq:.2f} Hz, File-Norm_Amp: {norm_amp:.4f}")
                    else:
                        print(f"\nNo FFT data with 'angle1' to display for {file_name_for_print} for demonstration.")
                else:
                    print("\nNo processed and normalized data available for demonstration.")
            else:
                print("\nNo file-specific normalized FFT data available to display for demonstration.")

    # Test calls for transform_filename_parts (REMOVED/COMMENTED OUT)
    # print("\n--- Testing transform_filename_parts ---")
    # test_cases = ["001_16.tsv", "01_5.tsv", "badformat.tsv", "001_16_extra.tsv", "another_bad_format_again.tsv"]
    # expected_outputs = ["61-100", "5-10", "badformat", "001_16_extra", "another_bad_format_again"]

    # for i, test_name in enumerate(test_cases):
    #     transformed = transform_filename_parts(test_name)
    #     print(f"Original: '{test_name}', Transformed: '{transformed}', Expected: '{expected_outputs[i]}'")
    #     assert transformed == expected_outputs[i], f"Test failed for {test_name}"

    print("\nScript finished.")

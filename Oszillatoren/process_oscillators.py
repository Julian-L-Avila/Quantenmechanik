import pandas
import os
import numpy as np
from scipy.fft import fft, fftfreq

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
    try:
        # Assuming the script is in Oszillatoren/ and data is in Oszillatoren/Data/
        # Construct the full path relative to the script's location
        script_dir = os.path.dirname(__file__) 
        # If relative_file_path already contains "Data/", os.path.join handles it well.
        # If script_dir is empty (e.g. when running from the same dir as the script),
        # it means the current working directory.
        if script_dir == "":
            full_path = relative_file_path
        else:
            full_path = os.path.join(script_dir, relative_file_path)

        # Correctly handle paths if relative_file_path might not start with "Data/"
        # For this problem, the instruction implies file_path will be like "Data/111_11.tsv"
        # So, if script is in "Oszillatoren/", and file_path is "Data/111_11.tsv",
        # full_path becomes "Oszillatoren/Data/111_11.tsv".
        # However, the problem states "relative to the script's location (e.g., Data/111_11.tsv)"
        # and then "construct the full path to the file (e.g., join with Data/)"
        # This is a bit ambiguous. Let's assume relative_file_path is *just* "111_11.tsv"
        # and we need to prepend "Data/".
        # Or, if relative_file_path is "Data/111_11.tsv", then we don't need to prepend "Data/".

        # Let's stick to the instruction: "file_path (relative to the Oszillatoren/ directory)"
        # and "The file paths should be relative to the script's location (e.g., Data/111_11.tsv)"
        # This means if the script is at Oszillatoren/process_oscillators.py,
        # then Data/111_11.tsv is Oszillatoren/Data/111_11.tsv

        # The argument is relative_file_path, which is stated to be e.g. "Data/111_11.tsv"
        # So, if the script is run from Oszillatoren/, then "Data/111_11.tsv" is correct.
        # If the script is run from /, then full_path needs to be "Oszillatoren/" + "Data/111_11.tsv"

        # __file__ will be 'Oszillatoren/process_oscillators.py' if run from project root,
        # or 'process_oscillators.py' if run from Oszillatoren directory.
        current_script_path = os.path.realpath(__file__) 
        script_directory = os.path.dirname(current_script_path)
        
        # relative_file_path is now like "Data/111_11.tsv".
        # os.path.join correctly constructs the path: <script_directory>/Data/111_11.tsv
        full_file_path = os.path.join(script_directory, relative_file_path)

        # Skip comment lines (starting with '#') and use specified column names
        # as the file does not have a clean header row after comments.
        df = pandas.read_csv(
            full_file_path,
            delimiter='\t',
            names=COLUMN_NAMES,
            comment='#',
            header=None
        )
        
        # Ensure numeric types, especially for time
        for col in COLUMN_NAMES:
            df[col] = pandas.to_numeric(df[col], errors='coerce')
        df.dropna(subset=COLUMN_NAMES, inplace=True)

        df['time'] = df['time'] - time_shift
        df_processed = df[df['time'] > 0].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        return df_processed

    except FileNotFoundError:
        print(f"Error: File not found at {full_file_path}") # full_file_path defined in try block
        return None
    except Exception as e:
        # It's good practice to ensure full_file_path is defined if used in error messages here
        # or pass relative_file_path which is always available.
        print(f"An error occurred while processing {relative_file_path}: {e}")
        return None

def calculate_fft(processed_dataframe: pandas.DataFrame) -> dict:
    """
    Calculates FFT for angle columns in the processed DataFrame.

    Args:
        processed_dataframe: DataFrame output from load_and_process_file.
                           Assumes it contains 'time' and angle columns.

    Returns:
        A dictionary where keys are angle column names, and values are
        dictionaries with 'frequency' and 'amplitude' numpy arrays.
    """
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

        # Calculate sampling interval dt
        # It's crucial that time_data is sorted and reasonably uniform for this dt calculation.
        # load_and_process_file already filters time > 0 and copies, implying order is preserved.
        if time_data.size < 2: # Should be caught by N < 2 already, but defensive check.
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

        # Select positive frequencies and corresponding amplitudes
        mask = xf > 0
        fft_results[angle_col_name] = {
            'frequency': xf[mask],
            'amplitude': np.abs(yf[mask])
        }
        
    return fft_results

# def normalize_spectra(all_files_fft_data: list, global_max_amplitude: float) -> list:
#     """
#     Normalizes the FFT amplitudes in all_files_fft_data using the global_max_amplitude.
# 
#     Args:
#         all_files_fft_data: A list of dictionaries, where each dictionary contains
#                             'file_name' and 'fft_results'.
#         global_max_amplitude: The global maximum amplitude found across all spectra.
# 
#     Returns:
#         A new list with the same structure, but with normalized amplitudes.
#     """
#     normalized_fft_data_all_files = []
#     for item in all_files_fft_data:
#         normalized_file_entry = {
#             'file_name': item['file_name'],
#             'normalized_fft_results': {}
#         }
#         for angle_column, angle_spec_data in item['fft_results'].items():
#             original_amplitudes = angle_spec_data['amplitude']
#             if global_max_amplitude > 0:
#                 normalized_amplitudes = original_amplitudes / global_max_amplitude
#             else:
#                 # Avoid division by zero; if global max is 0, all amplitudes must be 0.
#                 normalized_amplitudes = original_amplitudes 
#             
#             normalized_file_entry['normalized_fft_results'][angle_column] = {
#                 'frequency': angle_spec_data['frequency'],
#                 'amplitude': normalized_amplitudes
#             }
#         normalized_fft_data_all_files.append(normalized_file_entry)
#     return normalized_fft_data_all_files

def save_spectrum_data(original_file_path: str, 
                       fft_results_for_file: dict, # Changed name to reflect it can be unnormalized or normalized
                       output_dir: str):
    """
    Saves the normalized spectrum data for a single file to a TSV file.

    Args:
        original_file_path: The path of the original input file (e.g., "Data/111_11.tsv").
        normalized_fft_results_for_file: Dict with angle columns as keys and
                                         {'frequency': freqs, 'amplitude': norm_amps} as values.
        output_dir: The directory to save the output TSV file.
    """
    try:
        base_name = os.path.basename(original_file_path)
        output_filename = os.path.splitext(base_name)[0] + '_spectrum.tsv'
        full_output_path = os.path.join(output_dir, output_filename)

        df_data = {}
        
        reference_angle_data = None
        # Find a reference angle that has frequency data to determine the frequency column
        for angle in ANGLE_COLUMNS:
            if angle in fft_results_for_file and \
               fft_results_for_file[angle]['frequency'].size > 0:
                reference_angle_data = fft_results_for_file[angle]
                break
        
        if reference_angle_data is None:
            print(f"Skipping save for {original_file_path}: No frequency data found for any angle.")
            return

        frequencies = reference_angle_data['frequency']
        df_data['frequency'] = frequencies
        num_freq_points = len(frequencies)

        for angle_col in ANGLE_COLUMNS:
            angle_data = fft_results_for_file.get(angle_col)
            # Ensure data exists and matches the length of the frequency array
            if angle_data and angle_data['amplitude'].size == num_freq_points:
                df_data[angle_col] = angle_data['amplitude']
            else:
                # Fill with NaNs if data is missing, empty, or mismatched
                df_data[angle_col] = np.full(num_freq_points, np.nan)
                if angle_data: # Data exists but length is mismatched
                     print(f"Warning for {original_file_path}, {angle_col}: Mismatched data length. Expected {num_freq_points}, got {angle_data['amplitude'].size}. Filling with NaN.")
                # else: data for angle_col doesn't exist, already handled by .get() implicitly.
        
        spectrum_df = pandas.DataFrame(df_data)
        
        # Ensure columns are in the desired order: frequency, then angle columns that are present
        ordered_columns = ['frequency'] + [col for col in ANGLE_COLUMNS if col in spectrum_df.columns]
        spectrum_df = spectrum_df[ordered_columns]
        
        spectrum_df.to_csv(full_output_path, sep='\t', index=False, float_format='%.6e', na_rep='NaN')
        print(f"Saved spectrum data for {original_file_path} to {full_output_path}")

    except Exception as e:
        print(f"Error saving spectrum data for {original_file_path}: {e}")


if __name__ == "__main__":
    # Define and create output directory
    # Assuming the script is in Oszillatoren/, output will be Oszillatoren/Data/Processed_Spectra/
    script_dir = os.path.dirname(__file__)
    if script_dir == "": # Handles case where script is run from its own directory
        script_dir = "."
    output_directory = os.path.join(script_dir, 'Data', 'Processed_Spectra')
    os.makedirs(output_directory, exist_ok=True)
    print(f"Output directory set to: {os.path.abspath(output_directory)}")

    processed_and_normalized_data_for_saving = [] # Renamed to reflect its content

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

                    # File-specific normalization
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
                            normalized_amplitudes = original_amplitudes # Keep as is if max_amp is 0
                        
                        file_normalized_fft_data[angle_col_name_norm] = {
                            'frequency': frequencies,
                            'amplitude': normalized_amplitudes
                        }
                    
                    # Verification of normalized amplitudes
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

                    processed_and_normalized_data_for_saving.append({
                        'file_name': file_path,
                        'normalized_fft_results': file_normalized_fft_data 
                    })
                    print(f"File-specific normalization and verification done for {file_path}.")

                else:
                    print(f"Skipping FFT, normalization, and verification for {file_path}: Not enough data points (less than 2) after processing.")
            else:
                print(f"Skipping {file_path}: Failed to load or DataFrame is empty.")
        
        print(f"\nFinished processing all files. {len(processed_and_normalized_data_for_saving)} file(s) had successful FFT and normalization.")

        # Save Normalized Spectra
        print("\nStarting to save file-specific normalized spectra...")
        if not processed_and_normalized_data_for_saving:
             print("No normalized data to save.")
        else:
            for item in processed_and_normalized_data_for_saving:
                original_input_path = item['file_name']
                # The 'fft_results' key in save_spectrum_data now receives file-normalized data
                normalized_results_to_save = item['normalized_fft_results'] 
                save_spectrum_data(original_input_path, normalized_results_to_save, output_directory)
            print("Finished saving file-specific normalized spectra.")

            # Demonstrate File-Specific Normalization for the first successfully processed file
            if processed_and_normalized_data_for_saving:
                # Ensure there's at least one item before trying to access it
                if processed_and_normalized_data_for_saving: 
                    first_processed_file_data = processed_and_normalized_data_for_saving[0]
                    file_name_for_print = first_processed_file_data['file_name']
                    # Accessing the now file-normalized data
                    angle1_norm_fft = first_processed_file_data['normalized_fft_results'].get('angle1') 
                    
                    if angle1_norm_fft and angle1_norm_fft['frequency'].size > 0:
                        print(f"\nFile-Specific Normalized FFT results for angle1 of ({file_name_for_print}) (first 5 points - for demonstration):")
                        for i in range(min(5, angle1_norm_fft['frequency'].size)):
                            freq = angle1_norm_fft['frequency'][i]
                            norm_amp = angle1_norm_fft['amplitude'][i] # This is now file-normalized
                            print(f"Freq: {freq:.2f} Hz, File-Norm_Amp: {norm_amp:.4f}")
                    else:
                        print(f"\nNo FFT data with 'angle1' to display for {file_name_for_print} for demonstration.")
                else: # This case might be redundant due to the outer if, but good for safety
                    print("\nNo processed and normalized data available for demonstration.") 
            else:
                print("\nNo file-specific normalized FFT data available to display for demonstration.")

    print("\nScript finished.")

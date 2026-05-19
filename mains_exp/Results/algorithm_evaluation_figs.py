"""
algorithm_evaluation_figs.py
==============================
Generate evaluation figures from saved direct-response CSV files.
Point CSV_FOLDER at the directory containing *_direct_response.csv files
produced by main.py, then run this script.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dataviz.viz import (
    plot_rms_before_after,
    plot_response_parameter_histograms,
    plot_amplitudes_vs_pulse,
    plot_channel_amplitude_std_by_type,
)

CSV_FOLDER    = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\20hz'
OUTPUT_FOLDER = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\mains_exp\Results'

if __name__ == '__main__':
    # plot_rms_before_after(
    #     csv_folder=CSV_FOLDER,
    #     output_folder=OUTPUT_FOLDER,
    #     save=True,
    # )
    # plot_response_parameter_histograms(
    #     csv_folder=CSV_FOLDER,
    #     output_folder=OUTPUT_FOLDER,
    #     save=True,
    # )
    plot_amplitudes_vs_pulse(
        csv_folder=CSV_FOLDER,
        output_folder=OUTPUT_FOLDER,
        save=True,
    )
    # plot_channel_amplitude_std_by_type(
    #     csv_folder=CSV_FOLDER,
    #     output_folder=OUTPUT_FOLDER,
    #     save=True,
    # )

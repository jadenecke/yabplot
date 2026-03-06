import subprocess
import shutil
import os

def check_workbench():
    """
    Checks if Connectome Workbench (wb_command) is installed and available in PATH.
    
    Raises
    ------
    EnvironmentError
        If wb_command is not found, providing instructions for installation.
    """
    if shutil.which('wb_command') is None:
        raise EnvironmentError(
            "Connectome Workbench ('wb_command') was not found in your system PATH.\n"
            "This is required for volume-to-surface projection (necessary for creating a custom cortical atlas).\n"
            "Please download it from: https://humanconnectome.org/software/get-connectome-workbench\n"
            "After installing, ensure the 'bin' folder is added to your PATH environment variable."
        )

def run_wb_import(input_nii, label_list, output_nii):
    """Wrapper for wb_command -volume-label-import"""
    check_workbench()
    cmd = ["wb_command", "-volume-label-import", input_nii, label_list, output_nii]
    subprocess.run(cmd, check=True)

def run_wb_projection(input_nii, midthickness, output_gii, white, pial):
    """Wrapper for wb_command -volume-label-to-surface-mapping (ribbon-constrained)"""
    check_workbench()
    cmd = [
        "wb_command", "-volume-label-to-surface-mapping",
        input_nii, midthickness, output_gii,
        "-ribbon-constrained", white, pial
    ]
    subprocess.run(cmd, check=True)
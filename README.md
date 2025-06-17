# Stress Test Dashboard

This project provides a **System Stress Test Dashboard** built using **Streamlit**. It allows users to test the performance of their CPU and GPU using stress testing tools. The dashboard is designed to handle PyTorch compatibility issues and provides a user-friendly interface for configuring and running stress tests.

## Features

- **CPU Stress Test**: Uses Mandelbrot set calculations to stress CPU cores.
- **GPU Stress Test**: Performs large matrix multiplications to stress the GPU.
- **Lightweight GPU Mode**: Optimized for high iteration counts with better memory management.
- **System Information**: Displays CPU and GPU details, including GPU memory.
- **Interactive Dashboard**: Built with Streamlit for an intuitive user experience.
- **Early Stop Functionality**: Allows users to stop tests mid-execution.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.7 or higher
- Streamlit
- PyTorch (if GPU stress testing is required)

To install the required Python packages, run:

```bash
pip install streamlit torch torchvision torchaudio
```
## 📁 Project Structure

```
Stress-Test-Dashboard/
├── runner_file.py                 # Main Runner File
├── Stress_code_V2.py              # Main Streamlit File
```
## Key Files
- Stress_code_v2.py : Implementation the CPU and GPU stress test, along with the streamlit UI
- runner_file.py : Handles the environment setup and run the Streamlit Dashboard

# How to run
- Clone the repository or download the project files.
- Navigate to the File stress directory:
- Run the runner_file.py script:
- Open the Streamlit dashboard in your browser. The URL will be displayed in the terminal (e.g., http://localhost:8501).

## Usage
  - CPU Stress Test
  - Navigate to the CPU Stress Test tab.
  - Configure the number of iterations and matrix size.
  - Monitor progress and stop the test if needed.
## GPU Stress Test
  - Navigate to the GPU Stress Test tab.
  - Select the test mode (Standard or Lightweight).
  - Configure the number of iterations.
  - Monitor GPU memory usage and stop the test if needed.
## Results & Analytics
  - View detailed metrics, including average, minimum, and maximum execution times.
## Troubleshooting
  - PyTorch Not Available: Ensure PyTorch is installed and compatible with your system's GPU.
  - No GPU Detected: Verify that your system has a compatible GPU and the necessary drivers are installed.
## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Built with Streamlit for the interactive dashboard.
GPU stress testing powered by PyTorch.

 

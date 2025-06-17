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

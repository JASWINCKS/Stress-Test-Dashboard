import streamlit as st
import time
import threading
import queue
from contextlib import closing
from itertools import islice
from os import cpu_count
import io
import sys
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import torch with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.error("PyTorch not available. GPU testing will be disabled.")

# Streamlit configuration to avoid torch path issues
if 'torch_imported' not in st.session_state:
    st.session_state.torch_imported = TORCH_AVAILABLE

# Configure page
st.set_page_config(
    page_title="System Stress Test Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress .st-bo {
        background-color: #ff4b4b;
    }
    .status-running {
        color: #ff4b4b;
        font-weight: bold;
    }
    .status-completed {
        color: #00cc88;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cpu_running' not in st.session_state:
    st.session_state.cpu_running = False
if 'gpu_running' not in st.session_state:
    st.session_state.gpu_running = False
if 'cpu_results' not in st.session_state:
    st.session_state.cpu_results = []
if 'gpu_results' not in st.session_state:
    st.session_state.gpu_results = []
if 'cpu_progress' not in st.session_state:
    st.session_state.cpu_progress = 0
if 'gpu_progress' not in st.session_state:
    st.session_state.gpu_progress = 0

# CPU Stress Test Functions (Modified from m.py)
def pixels(y, n, abs_func):
    range7 = bytearray(range(7))
    pixel_bits = bytearray(128 >> pos for pos in range(8))
    c1 = 2. / float(n)
    c0 = -1.5 + 1j * y * c1 - 1j
    x = 0
    while True:
        pixel = 0
        c = x * c1 + c0
        for pixel_bit in pixel_bits:
            z = c
            for _ in range7:
                for _ in range7:
                    z = z * z + c
                if abs_func(z) >= 2.: break
            else:
                pixel += pixel_bit
            c += c1
        yield pixel
        x += 8

def compute_row(p):
    y, n = p
    result = bytearray(islice(pixels(y, n, abs), (n + 7) // 8))
    result[-1] &= 0xff << (8 - n % 8)
    return y, result

def ordered_rows(rows, n):
    order = [None] * n
    i = 0
    j = n
    while i < len(order):
        if j > 0:
            row = next(rows)
            order[row[0]] = row
            j -= 1
        if order[i]:
            yield order[i]
            order[i] = None
            i += 1

def compute_rows(n, f):
    row_jobs = ((y, n) for y in range(n))
    if cpu_count() < 2:
        yield from map(f, row_jobs)
    else:
        from multiprocessing import Pool
        with Pool() as pool:
            unordered_rows = pool.imap_unordered(f, row_jobs)
            yield from ordered_rows(unordered_rows, n)

def mandelbrot_cpu_test(n):
    """Modified mandelbrot function for CPU stress testing"""
    with closing(compute_rows(n, compute_row)) as rows:
        for row in rows:
            pass  # Just compute, don't write to stdout

def cpu_stress_test(iterations, matrix_size, progress_callback=None):
    """CPU stress test using Mandelbrot set calculation"""
    results = []
    
    for i in range(iterations):
        start_time = time.time()
        mandelbrot_cpu_test(matrix_size)
        end_time = time.time()
        
        execution_time = end_time - start_time
        results.append({
            'iteration': i + 1,
            'time_minutes': execution_time / 60,
            'time_seconds': execution_time
        })
        
        if progress_callback:
            progress_callback(i + 1, iterations, execution_time)
    
    return results

def gpu_stress_test(iterations, progress_callback=None):
    """GPU stress test using matrix multiplication"""
    if not TORCH_AVAILABLE:
        return None, "PyTorch not available"
    
    if not torch.cuda.is_available():
        return None, "No GPU available"
    
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    
    # Calculate matrix size
    factor = 0.8
    matrix_size = int((total_memory * factor / 4) ** 0.5)
    
    # Initialize matrices
    a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
    
    results = []
    
    for i in range(iterations):
        start_time = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()  # Ensure computation is complete
        end_time = time.time()
        
        execution_time = end_time - start_time
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        
        results.append({
            'iteration': i + 1,
            'time_seconds': execution_time,
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'matrix_size': matrix_size
        })
        
        if progress_callback:
            progress_callback(i + 1, iterations, execution_time, memory_allocated)
    
    return results, device_name

# Streamlit UI
st.title("🔥 System Stress Test Dashboard")
st.markdown("Test your CPU and GPU performance with comprehensive stress testing tools")

# Sidebar for controls
st.sidebar.header("⚙️ Test Configuration")

# System Info
st.sidebar.subheader("💻 System Information")
st.sidebar.info(f"**CPU Cores:** {cpu_count()}")
if TORCH_AVAILABLE and torch.cuda.is_available():
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        st.sidebar.info(f"**GPU:** {gpu_name}")
        st.sidebar.info(f"**GPU Memory:** {gpu_memory:.1f} GB")
    except Exception as e:
        st.sidebar.warning(f"GPU detected but error getting info: {str(e)}")
elif TORCH_AVAILABLE:
    st.sidebar.warning("PyTorch available but no GPU detected")
else:
    st.sidebar.warning("PyTorch not available - GPU testing disabled")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["🖥️ CPU Stress Test", "🎮 GPU Stress Test", "📊 Results & Analytics"])

# CPU Stress Test Tab
with tab1:
    st.header("CPU Stress Test (Mandelbrot Set)")
    st.markdown("This test uses Mandelbrot set calculations to stress your CPU cores.")
    st.markdown("**Note:** Dont Change the Iterations value from Default Untill you are sure about it")

    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        cpu_iterations = st.number_input("Number of Iterations", min_value=1, max_value=100, value=10, key="cpu_iter")
        cpu_matrix_size = st.selectbox("Matrix Size (n×n)", [500, 1000, 1500, 2000, 2500,5000,10000,16000], index=1, key="cpu_size")
        
        if st.button("🚀 Start CPU Test", disabled=st.session_state.cpu_running):
            st.session_state.cpu_running = True
            st.session_state.cpu_progress = 0
            st.session_state.cpu_results = []
            
            # Create progress containers
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def cpu_progress_callback(current, total, exec_time):
                progress = current / total
                st.session_state.cpu_progress = progress
                progress_bar.progress(progress)
                status_text.text(f"Iteration {current}/{total} - Last: {exec_time:.2f}s")
            
            # Run test in thread to avoid blocking UI
            with st.spinner("Running CPU stress test..."):
                results = cpu_stress_test(cpu_iterations, cpu_matrix_size, cpu_progress_callback)
                st.session_state.cpu_results = results
                st.session_state.cpu_running = False
                st.success("CPU test completed!")
    
    with col2:
        st.subheader("Live Results")
        if st.session_state.cpu_results:
            df = pd.DataFrame(st.session_state.cpu_results)
            
            # Real-time chart
            fig = px.line(df, x='iteration', y='time_seconds', 
                         title='CPU Performance Over Time',
                         labels={'time_seconds': 'Execution Time (seconds)', 'iteration': 'Iteration'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            avg_time = df['time_seconds'].mean()
            min_time = df['time_seconds'].min()
            max_time = df['time_seconds'].max()
            
            col1_stat, col2_stat, col3_stat = st.columns(3)
            col1_stat.metric("Average Time", f"{avg_time:.3f}s")
            col2_stat.metric("Min Time", f"{min_time:.3f}s")
            col3_stat.metric("Max Time", f"{max_time:.3f}s")

# GPU Stress Test Tab
# with tab2:
#     st.header("GPU Stress Test (Matrix Multiplication)")
#     st.markdown("This test uses large matrix multiplications to stress your GPU.")
#     st.markdown("**Note:** Dont Change the Iterations value from Default Untill you are sure about it")
    
#     if not TORCH_AVAILABLE:
#         st.error("❌ PyTorch not available. Please install PyTorch to enable GPU stress testing.")
#         st.code("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
#     elif not torch.cuda.is_available():
#         st.error("❌ No GPU detected. GPU stress testing is not available.")
#     else:
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             st.subheader("Configuration")
#             gpu_iterations = st.number_input("Number of Iterations", min_value=1, max_value=1000, value=1000, key="gpu_iter")
            
#             # Show estimated matrix size
#             if TORCH_AVAILABLE and torch.cuda.is_available():
#                 try:
#                     device = torch.device("cuda")
#                     total_memory = torch.cuda.get_device_properties(device).total_memory
#                     matrix_size = int((total_memory * 0.8 / 4) ** 0.5)
#                     st.info(f"Estimated matrix size: {matrix_size}×{matrix_size}")
#                 except Exception as e:
#                     st.warning(f"Could not calculate matrix size: {str(e)}")
            
#             if st.button("🚀 Start GPU Test", disabled=st.session_state.gpu_running):
#                 st.session_state.gpu_running = True
#                 st.session_state.gpu_progress = 0
#                 st.session_state.gpu_results = []
                
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
#                 memory_text = st.empty()
                
#                 def gpu_progress_callback(current, total, exec_time, memory_mb):
#                     progress = current / total
#                     st.session_state.gpu_progress = progress
#                     progress_bar.progress(progress)
#                     status_text.text(f"Iteration {current}/{total} - Last: {exec_time:.4f}s")
#                     memory_text.text(f"GPU Memory: {memory_mb:.1f} MB")
                
#                 with st.spinner("Running GPU stress test..."):
#                     results, device_name = gpu_stress_test(gpu_iterations, gpu_progress_callback)
#                     if results:
#                         st.session_state.gpu_results = results
#                         st.success(f"GPU test completed on {device_name}!")
#                     else:
#                         st.error("GPU test failed!")
#                     st.session_state.gpu_running = False
        
#         with col2:
#             st.subheader("Live Results")
#             if st.session_state.gpu_results:
#                 df = pd.DataFrame(st.session_state.gpu_results)
                
#                 # Performance chart
#                 fig = px.line(df, x='iteration', y='time_seconds',
#                              title='GPU Performance Over Time',
#                              labels={'time_seconds': 'Execution Time (seconds)', 'iteration': 'Iteration'})
#                 st.plotly_chart(fig, use_container_width=True)
                
#                 # Memory usage chart
#                 fig_mem = px.line(df, x='iteration', y='memory_allocated_mb',
#                                  title='GPU Memory Usage Over Time',
#                                  labels={'memory_allocated_mb': 'Memory (MB)', 'iteration': 'Iteration'})
#                 st.plotly_chart(fig_mem, use_container_width=True)
                
#                 # Statistics
#                 avg_time = df['time_seconds'].mean()
#                 avg_memory = df['memory_allocated_mb'].mean()
                
#                 col1_stat, col2_stat = st.columns(2)
#                 col1_stat.metric("Avg Execution Time", f"{avg_time:.6f}s")
#                 col2_stat.metric("Avg Memory Usage", f"{avg_memory:.1f} MB")

def gpu_stress_test_universal(iterations, progress_callback=None):
    """
    Universal GPU stress test that works with CUDA, MPS, and CPU
    """
    import torch
    import time
    import gc
    
    results = []
    
    try:
        # Determine the best available device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            supports_memory_stats = True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Apple Silicon GPU (MPS)"
            supports_memory_stats = False  # MPS doesn't support memory stats yet
        else:
            device = torch.device("cpu")
            device_name = "CPU"
            supports_memory_stats = False
        
        # Determine matrix size based on device
        if device.type == "cuda":
            try:
                total_memory = torch.cuda.get_device_properties(device).total_memory
                matrix_size = int((total_memory * 0.6 / 4) ** 0.5)  # More conservative
                matrix_size = min(matrix_size, 8192)  # Cap at 8192
            except:
                matrix_size = 4096
        elif device.type == "mps":
            matrix_size = 4096  # Conservative for MPS
        else:  # CPU
            matrix_size = 1024  # Much smaller for CPU
        
        print(f"Using device: {device}, Matrix size: {matrix_size}x{matrix_size}")
        
        for i in range(iterations):
            # Clear cache before each iteration
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            
            gc.collect()
            
            start_time = time.time()
            
            # Create random matrices
            try:
                a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
                b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
                
                # Perform matrix multiplication
                c = torch.matmul(a, b)
                
                # Ensure computation is complete
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Get memory usage if supported
                memory_mb = 0
                if supports_memory_stats and device.type == "cuda":
                    try:
                        memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
                    except:
                        memory_mb = 0
                
                results.append({
                    'iteration': i + 1,
                    'time_seconds': execution_time,
                    'memory_allocated_mb': memory_mb,
                    'matrix_size': matrix_size
                })
                
                # Clean up
                del a, b, c
                
                if progress_callback:
                    progress_callback(i + 1, iterations, execution_time, memory_mb)
                    
            except Exception as e:
                print(f"Error in iteration {i + 1}: {str(e)}")
                # Try with smaller matrix size
                matrix_size = int(matrix_size * 0.8)
                if matrix_size < 512:
                    break
                continue
        
        return results, device_name
        
    except Exception as e:
        print(f"GPU stress test failed: {str(e)}")
        return None, None

with tab2:
    st.header("GPU Stress Test (Matrix Multiplication)")
    st.markdown("This test uses large matrix multiplications to stress your GPU.")
    st.markdown("**Note:** Don't Change the Iterations value from Default Until you are sure about it")
    
    if not TORCH_AVAILABLE:
        st.error("❌ PyTorch not available. Please install PyTorch to enable GPU stress testing.")
        st.code("pip install torch torchvision torchaudio")
    else:
        # Check for available GPU backends
        gpu_available = False
        device_type = None
        device_name = "Unknown"
        
        if torch.cuda.is_available():
            gpu_available = True
            device_type = "cuda"
            device_name = torch.cuda.get_device_name(0)
            st.success(f"✅ NVIDIA GPU detected: {device_name}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_available = True
            device_type = "mps"
            device_name = "Apple Silicon GPU (MPS)"
            st.success(f"✅ Apple Silicon GPU detected: {device_name}")
        else:
            st.error("❌ No GPU detected. GPU stress testing will run on CPU (not recommended).")
            st.info("💡 For Apple Silicon Macs, ensure you have PyTorch with MPS support installed.")
        
        if gpu_available or st.checkbox("Run on CPU (slower)", help="Check this to run the test on CPU"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Configuration")
                
                # Adjust default iterations based on device
                if device_type == "mps":
                    default_iterations = 500  # Lower default for MPS
                    max_iterations = 1000
                elif device_type == "cuda":
                    default_iterations = 1000
                    max_iterations = 2000
                else:  # CPU
                    default_iterations = 10
                    max_iterations = 100
                
                gpu_iterations = st.number_input(
                    "Number of Iterations", 
                    min_value=1, 
                    max_value=max_iterations, 
                    value=default_iterations, 
                    key="gpu_iter"
                )
                
                # Show estimated matrix size
                try:
                    if device_type == "cuda":
                        device = torch.device("cuda")
                        total_memory = torch.cuda.get_device_properties(device).total_memory
                        matrix_size = int((total_memory * 0.8 / 4) ** 0.5)
                        st.info(f"Estimated matrix size: {matrix_size}×{matrix_size}")
                    elif device_type == "mps":
                        # For MPS, we use a conservative estimate since we can't query memory directly
                        # Apple Silicon typically has unified memory
                        matrix_size = 4096  # Conservative estimate
                        st.info(f"Estimated matrix size: {matrix_size}×{matrix_size}")
                        st.warning("⚠️ MPS memory usage estimation is approximate")
                    else:  # CPU
                        matrix_size = 1024  # Much smaller for CPU
                        st.info(f"CPU matrix size: {matrix_size}×{matrix_size}")
                except Exception as e:
                    st.warning(f"Could not calculate matrix size: {str(e)}")
                
                if st.button("🚀 Start GPU Test", disabled=st.session_state.get('gpu_running', False)):
                    st.session_state.gpu_running = True
                    st.session_state.gpu_progress = 0
                    st.session_state.gpu_results = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    memory_text = st.empty()
                    
                    def gpu_progress_callback(current, total, exec_time, memory_mb):
                        progress = current / total
                        st.session_state.gpu_progress = progress
                        progress_bar.progress(progress)
                        status_text.text(f"Iteration {current}/{total} - Last: {exec_time:.4f}s")
                        if memory_mb > 0:
                            memory_text.text(f"GPU Memory: {memory_mb:.1f} MB")
                        else:
                            memory_text.text("Memory usage: Not available")
                    
                    with st.spinner(f"Running GPU stress test on {device_name}..."):
                        results, final_device_name = gpu_stress_test_universal(gpu_iterations, gpu_progress_callback)
                        if results:
                            st.session_state.gpu_results = results
                            st.success(f"GPU test completed on {final_device_name}!")
                        else:
                            st.error("GPU test failed!")
                        st.session_state.gpu_running = False
            
            with col2:
                st.subheader("Live Results")
                if st.session_state.get('gpu_results', []):
                    df = pd.DataFrame(st.session_state.gpu_results)
                    
                    # Performance chart
                    fig = px.line(df, x='iteration', y='time_seconds',
                                 title='GPU Performance Over Time',
                                 labels={'time_seconds': 'Execution Time (seconds)', 'iteration': 'Iteration'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Memory usage chart (only if memory data is available)
                    if 'memory_allocated_mb' in df.columns and df['memory_allocated_mb'].max() > 0:
                        fig_mem = px.line(df, x='iteration', y='memory_allocated_mb',
                                         title='GPU Memory Usage Over Time',
                                         labels={'memory_allocated_mb': 'Memory (MB)', 'iteration': 'Iteration'})
                        st.plotly_chart(fig_mem, use_container_width=True)
                    
                    # Statistics
                    avg_time = df['time_seconds'].mean()
                    min_time = df['time_seconds'].min()
                    max_time = df['time_seconds'].max()
                    
                    col1_stat, col2_stat, col3_stat = st.columns(3)
                    col1_stat.metric("Avg Execution Time", f"{avg_time:.6f}s")
                    col2_stat.metric("Min Time", f"{min_time:.6f}s")
                    col3_stat.metric("Max Time", f"{max_time:.6f}s")
                    
                    if 'memory_allocated_mb' in df.columns and df['memory_allocated_mb'].max() > 0:
                        avg_memory = df['memory_allocated_mb'].mean()
                        max_memory = df['memory_allocated_mb'].max()
                        
                        col1_mem, col2_mem = st.columns(2)
                        col1_mem.metric("Avg Memory Usage", f"{avg_memory:.1f} MB")
                        col2_mem.metric("Peak Memory Usage", f"{max_memory:.1f} MB")



# Results & Analytics Tab
with tab3:
    st.header("📊 Test Results & Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CPU Test Summary")
        if st.session_state.cpu_results:
            df_cpu = pd.DataFrame(st.session_state.cpu_results)
            st.dataframe(df_cpu)
            
            # Download button for CPU results
            csv_cpu = df_cpu.to_csv(index=False)
            st.download_button(
                label="📥 Download CPU Results (CSV)",
                data=csv_cpu,
                file_name=f"cpu_stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No CPU test results available. Run a CPU test first.")
    
    with col2:
        st.subheader("GPU Test Summary")
        if st.session_state.gpu_results:
            df_gpu = pd.DataFrame(st.session_state.gpu_results)
            st.dataframe(df_gpu)
            
            # Download button for GPU results
            csv_gpu = df_gpu.to_csv(index=False)
            st.download_button(
                label="📥 Download GPU Results (CSV)",
                data=csv_gpu,
                file_name=f"gpu_stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No GPU test results available. Run a GPU test first.")
    
    # Combined Analysis
    if st.session_state.cpu_results and st.session_state.gpu_results:
        st.subheader("🔍 Comparative Analysis")
        
        df_cpu = pd.DataFrame(st.session_state.cpu_results)
        df_gpu = pd.DataFrame(st.session_state.gpu_results)
        
        col1_comp, col2_comp = st.columns(2)
        
        with col1_comp:
            st.metric("CPU Avg Time", f"{df_cpu['time_seconds'].mean():.3f}s")
            st.metric("CPU Total Iterations", len(df_cpu))
            
        with col2_comp:
            st.metric("GPU Avg Time", f"{df_gpu['time_seconds'].mean():.6f}s")
            st.metric("GPU Total Iterations", len(df_gpu))

# Footer
st.markdown("---")
st.markdown("**System Stress Test Dashboard** - Monitor your hardware performance in real-time")

# Auto-refresh for live updates (commented out to avoid constant rerunning)
# if st.session_state.cpu_running or st.session_state.gpu_running:
#     time.sleep(1)
#     st.rerun()
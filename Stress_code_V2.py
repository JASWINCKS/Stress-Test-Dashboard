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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
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
if 'gpu_stop_requested' not in st.session_state:
    st.session_state.gpu_stop_requested = False
if 'cpu_stop_requested' not in st.session_state:
    st.session_state.cpu_stop_requested = False

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

def cpu_stress_test(iterations, matrix_size, progress_callback=None, early_stop_check=None):
    """CPU stress test using Mandelbrot set calculation with early stopping"""
    results = []
    
    for i in range(iterations):
        # Check for early stopping
        if early_stop_check and early_stop_check():
            break
            
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

def gpu_stress_test(iterations, progress_callback=None, early_stop_check=None):
    """Optimized GPU stress test with better memory management and performance"""
    if not TORCH_AVAILABLE:
        return None, "PyTorch not available"
    
    if not torch.cuda.is_available():
        return None, "No GPU available"
    
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    
    # More conservative memory usage (reduced from 0.8 to 0.6)
    factor = 0.6  # Use 60% of GPU memory instead of 80%
    matrix_size = int((total_memory * factor / 4) ** 0.5)
    
    # Clamp matrix size to reasonable bounds
    matrix_size = min(matrix_size, 8192)  # Max 8192x8192
    matrix_size = max(matrix_size, 1024)  # Min 1024x1024
    
    try:
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        
        # Pre-allocate matrices once (major optimization)
        a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
        b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
        c = torch.zeros((matrix_size, matrix_size), device=device, dtype=torch.float32)  # Pre-allocate result
        
        results = []
        
        # Batch processing for better performance
        batch_size = min(10, iterations)  # Process in batches of 10 or less
        
        for batch_start in range(0, iterations, batch_size):
            batch_end = min(batch_start + batch_size, iterations)
            batch_results = []
            
            # Process batch
            for i in range(batch_start, batch_end):
                # Check for early stopping
                if early_stop_check and early_stop_check():
                    results.extend(batch_results)
                    return results, f"{device_name} (stopped early)"
                
                start_time = time.time()
                
                # Use torch.mm with out parameter to avoid new memory allocation
                torch.mm(a, b, out=c)
                torch.cuda.synchronize()  # Ensure computation is complete
                
                end_time = time.time()
                
                execution_time = end_time - start_time
                memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
                memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                
                batch_results.append({
                    'iteration': i + 1,
                    'time_seconds': execution_time,
                    'memory_allocated_mb': memory_allocated,
                    'memory_reserved_mb': memory_reserved,
                    'matrix_size': matrix_size
                })
                
                if progress_callback:
                    progress_callback(i + 1, iterations, execution_time, memory_allocated)
            
            results.extend(batch_results)
            
            # Optional: Small delay between batches to prevent overheating
            if batch_end < iterations:
                time.sleep(0.01)  # 10ms pause between batches
        
        return results, device_name
        
    except torch.cuda.OutOfMemoryError:
        # Handle OOM gracefully
        torch.cuda.empty_cache()
        return None, f"GPU Out of Memory - try reducing matrix size or iterations"
    except Exception as e:
        torch.cuda.empty_cache()
        return None, f"GPU test failed: {str(e)}"
    finally:
        # Always clean up
        torch.cuda.empty_cache()

def gpu_stress_test_lightweight(iterations, progress_callback=None, early_stop_check=None):
    """Lightweight GPU stress test for very high iteration counts"""
    if not TORCH_AVAILABLE:
        return None, "PyTorch not available"
    
    if not torch.cuda.is_available():
        return None, "No GPU available"
    
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    
    # Much smaller matrices for high iteration testing
    matrix_size = 2048  # Fixed smaller size
    
    try:
        torch.cuda.empty_cache()
        
        # Smaller pre-allocated matrices
        a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)  # Use half precision
        b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)
        c = torch.zeros((matrix_size, matrix_size), device=device, dtype=torch.float16)
        
        results = []
        
        # Process in larger batches for efficiency
        batch_size = min(100, iterations)
        update_frequency = max(1, iterations // 100)  # Update progress less frequently
        
        for batch_start in range(0, iterations, batch_size):
            batch_end = min(batch_start + batch_size, iterations)
            
            for i in range(batch_start, batch_end):
                if early_stop_check and early_stop_check():
                    return results, f"{device_name} (stopped early)"
                
                start_time = time.perf_counter()  # More precise timing
                torch.mm(a, b, out=c)
                # Skip synchronize for better performance, only sync occasionally
                if i % 50 == 0:  # Sync every 50 iterations
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                
                # Only collect detailed memory info occasionally to improve performance
                if i % update_frequency == 0 or i == iterations - 1:
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
        
    except Exception as e:
        torch.cuda.empty_cache()
        return None, f"Lightweight GPU test failed: {str(e)}"
    finally:
        torch.cuda.empty_cache()

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
    
    # Warning box
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Performance Warning:</strong> High iteration counts or large matrix sizes can take significant time to complete. 
        Start with default values and use the stop button if needed.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        cpu_iterations = st.number_input("Number of Iterations", min_value=1, max_value=100, value=10, key="cpu_iter")
        cpu_matrix_size = st.selectbox("Matrix Size (n×n)", [500, 1000, 1500, 2000, 2500, 5000, 10000, 16000], index=1, key="cpu_size")
        
        # Estimated time calculation
        est_time_per_iter = {500: 0.1, 1000: 0.5, 1500: 1.2, 2000: 2.1, 2500: 3.3, 5000: 13, 10000: 52, 16000: 134}
        estimated_total_time = cpu_iterations * est_time_per_iter.get(cpu_matrix_size, 1)
        st.info(f"Estimated time: ~{estimated_total_time:.1f}s ({estimated_total_time/60:.1f} min)")
        
        # Control buttons
        col_start, col_stop = st.columns(2)
        
        with col_start:
            start_cpu_clicked = st.button("🚀 Start CPU Test", disabled=st.session_state.cpu_running)
        
        with col_stop:
            if st.button("🛑 Stop CPU Test", disabled=not st.session_state.cpu_running):
                st.session_state.cpu_stop_requested = True
        
        if start_cpu_clicked:
            st.session_state.cpu_running = True
            st.session_state.cpu_progress = 0
            st.session_state.cpu_results = []
            st.session_state.cpu_stop_requested = False
            
            # Create progress containers
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def cpu_progress_callback(current, total, exec_time):
                progress = current / total
                st.session_state.cpu_progress = progress
                progress_bar.progress(progress)
                status_text.text(f"Iteration {current}/{total} - Last: {exec_time:.2f}s")
            
            def cpu_early_stop_check():
                return st.session_state.cpu_stop_requested
            
            # Run test in thread to avoid blocking UI
            with st.spinner("Running CPU stress test..."):
                results = cpu_stress_test(cpu_iterations, cpu_matrix_size, cpu_progress_callback, cpu_early_stop_check)
                st.session_state.cpu_results = results
                st.session_state.cpu_running = False
                
                if st.session_state.cpu_stop_requested:
                    st.warning(f"CPU test stopped early after {len(results)} iterations!")
                else:
                    st.success("CPU test completed!")
                
                st.session_state.cpu_stop_requested = False
    
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
with tab2:
    st.header("GPU Stress Test (Matrix Multiplication)")
    st.markdown("This test uses large matrix multiplications to stress your GPU.")
    
    if not TORCH_AVAILABLE:
        st.error("❌ PyTorch not available. Please install PyTorch to enable GPU stress testing.")
        st.code("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    elif not torch.cuda.is_available():
        st.error("❌ No GPU detected. GPU stress testing is not available.")
    else:
        # Warning box
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>GPU Warning:</strong> High iteration counts can stress your GPU significantly. 
            Monitor temperatures and use the stop button if needed. Lightweight mode is recommended for >1000 iterations.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuration")
            gpu_iterations = st.number_input("Number of Iterations", min_value=1, max_value=10000, value=100, key="gpu_iter")
            
            # Test mode selection
            test_mode = st.selectbox(
                "Test Mode",
                ["Standard (Full Memory)", "Lightweight (High Iterations)"],
                help="Standard: Uses more GPU memory, better for stress testing. Lightweight: Uses less memory, better for high iteration counts."
            )
            
            # Show estimated matrix size and time
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    device = torch.device("cuda")
                    total_memory = torch.cuda.get_device_properties(device).total_memory
                    if test_mode == "Standard (Full Memory)":
                        matrix_size = int((total_memory * 0.6 / 4) ** 0.5)
                        matrix_size = min(matrix_size, 8192)
                        matrix_size = max(matrix_size, 1024)
                        est_time = gpu_iterations * 0.01
                    else:
                        matrix_size = 2048
                        est_time = gpu_iterations * 0.001
                    
                    st.info(f"Matrix size: {matrix_size}×{matrix_size}")
                    st.info(f"Estimated time: ~{est_time:.1f}s")
                except Exception as e:
                    st.warning(f"Could not calculate estimates: {str(e)}")
            
            # Control buttons
            col_start, col_stop = st.columns(2)
            
            with col_start:
                start_gpu_clicked = st.button("🚀 Start GPU Test", disabled=st.session_state.gpu_running)
            
            with col_stop:
                if st.button("🛑 Stop GPU Test", disabled=not st.session_state.gpu_running):
                    st.session_state.gpu_stop_requested = True
            
            if start_gpu_clicked:
                st.session_state.gpu_running = True
                st.session_state.gpu_progress = 0
                st.session_state.gpu_results = []
                st.session_state.gpu_stop_requested = False
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                memory_text = st.empty()
                
                def gpu_progress_callback(current, total, exec_time, memory_mb):
                    progress = current / total
                    st.session_state.gpu_progress = progress
                    progress_bar.progress(progress)
                    status_text.text(f"Iteration {current}/{total} - Last: {exec_time:.4f}s")
                    memory_text.text(f"GPU Memory: {memory_mb:.1f} MB")
                
                def gpu_early_stop_check():
                    return st.session_state.gpu_stop_requested
                
                with st.spinner("Running GPU stress test..."):
                    if test_mode == "Lightweight (High Iterations)":
                        results, device_name = gpu_stress_test_lightweight(
                            gpu_iterations, gpu_progress_callback, gpu_early_stop_check
                        )
                    else:
                        results, device_name = gpu_stress_test(
                            gpu_iterations, gpu_progress_callback, gpu_early_stop_check
                        )
                    
                    if results:
                        st.session_state.gpu_results = results
                        if st.session_state.gpu_stop_requested:
                            st.warning(f"GPU test stopped early after {len(results)} iterations on {device_name}")
                        else:
                            st.success(f"GPU test completed on {device_name}!")
                    else:
                        st.error(f"GPU test failed: {device_name}")
                    
                    st.session_state.gpu_running = False
                    st.session_state.gpu_stop_requested = False
        
        with col2:
            st.subheader("Live Results")
            if st.session_state.gpu_results:
                df = pd.DataFrame(st.session_state.gpu_results)
                
                # Performance chart
                fig = px.line(df, x='iteration', y='time_seconds',
                             title='GPU Performance Over Time',
                             labels={'time_seconds': 'Execution Time (seconds)', 'iteration': 'Iteration'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Memory usage chart
                fig_mem = px.line(df, x='iteration', y='memory_allocated_mb',
                                 title='GPU Memory Usage Over Time',
                                 labels={'memory_allocated_mb': 'Memory (MB)', 'iteration': 'Iteration'})
                st.plotly_chart(fig_mem, use_container_width=True)
                
                # Statistics
                avg_time = df['time_seconds'].mean()
                avg_memory = df['memory_allocated_mb'].mean()
                
                col1_stat, col2_stat = st.columns(2)
                col1_stat.metric("Avg Execution Time", f"{avg_time:.6f}s")
                col2_stat.metric("Avg Memory Usage", f"{avg_memory:.1f} MB")

# Results & Analytics Tab
with tab3:
    st.header("📊 Test Results & Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CPU Test Summary")
        if st.session_state.cpu_results:
            df_cpu = pd.DataFrame(st.session_state.cpu_results)
            st.dataframe(df_cpu)
            
            # Additional CPU statistics
            st.write("**Performance Statistics:**")
            st.write(f"- Total Iterations: {len(df_cpu)}")
            st.write(f"- Average Time: {df_cpu['time_seconds'].mean():.3f}s")
            st.write(f"- Standard Deviation: {df_cpu['time_seconds'].std():.3f}s")
            st.write(f"- Total Test Time: {df_cpu['time_seconds'].sum():.1f}s")
            
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
            
            # Additional GPU statistics
            st.write("**Performance Statistics:**")
            st.write(f"- Total Iterations: {len(df_gpu)}")
            st.write(f"- Average Time: {df_gpu['time_seconds'].mean():.6f}s")
            st.write(f"- Matrix Size: {df_gpu['matrix_size'].iloc[0]}×{df_gpu['matrix_size'].iloc[0]}")
            st.write(f"- Peak Memory: {df_gpu['memory_allocated_mb'].max():.1f} MB")
            st.write(f"- Total Test Time: {df_gpu['time_seconds'].sum():.1f}s")
            
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
        
        col1_comp, col2_comp, col3_comp, col4_comp = st.columns(4)
        
        with col1_comp:
            st.metric("CPU Avg Time", f"{df_cpu['time_seconds'].mean():.3f}s")
            
        with col2_comp:
            st.metric("CPU Total Iterations", len(df_cpu))
            
        with col3_comp:
            st.metric("GPU Avg Time", f"{df_gpu['time_seconds'].mean():.6f}s")
            
        with col4_comp:
            st.metric("GPU Total Iterations", len(df_gpu))
        
        # Performance comparison chart
        if len(df_cpu) > 0 and len(df_gpu) > 0:
            st.subheader("Performance Comparison")
            
            # Normalize iteration counts for comparison
            cpu_normalized = df_cpu.copy()
            gpu_normalized = df_gpu.copy()
            
            # Create comparison chart
            fig_comp = go.Figure()
            
            fig_comp.add_trace(go.Scatter(
                x=cpu_normalized['iteration'],
                y=cpu_normalized['time_seconds'],
                mode='lines+markers',
                name='CPU',
                line=dict(color='blue')
            ))
            
            fig_comp.add_trace(go.Scatter(
                x=gpu_normalized['iteration'],
                y=gpu_normalized['time_seconds'],
                mode='lines+markers',
                name='GPU',
                line=dict(color='red'),
                yaxis='y2'
            ))
            
            fig_comp.update_layout(
                title='CPU vs GPU Performance Comparison',
                xaxis_title='Iteration',
                yaxis=dict(title='CPU Time (seconds)', side='left', color='blue'),
                yaxis2=dict(title='GPU Time (seconds)', side='right', overlaying='y', color='red'),
                legend=dict(x=0.7, y=1)
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**System Stress Test Dashboard** - Monitor your hardware performance in real-time")
st.markdown("**Version 2.0** - Optimized for better performance and user control")

# Performance tips in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Performance Tips")
st.sidebar.markdown("""
**CPU Testing:**
- Start with 10 iterations
- Matrix size 1000-2000 for quick tests
- Use higher values for stress testing

**GPU Testing:**
- Standard mode: 100-1000 iterations
- Lightweight mode: 1000+ iterations
- Monitor GPU temperature during tests

**General:**
- Use stop buttons for long tests
- Close other applications for accuracy
- Save results before running new tests
""")

# Auto-refresh for live updates (commented out to avoid constant rerunning)
# if st.session_state.cpu_running or st.session_state.gpu_running:
#     time.sleep(1)
#     st.rerun()
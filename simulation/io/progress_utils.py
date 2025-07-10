"""Progress monitoring utilities with tqdm-like formatting for training data collection."""

import time
import threading
from queue import Empty


def format_time(seconds):
    """Format seconds into human-readable time string"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m{secs:02.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h{minutes:02.0f}m"


def format_rate(rate):
    """Format rate in tasks per second"""
    if rate >= 1.0:
        return f"{rate:.2f}it/s"
    elif rate > 0:
        return f"{1/rate:.2f}s/it"
    else:
        return "0.00it/s"


def create_progress_bar(completed, total, width=50):
    """Create tqdm-style progress bar"""
    if total <= 0:
        return "[" + " " * width + "]"
    
    filled_length = int(width * completed / total)
    bar = "█" * filled_length + "░" * (width - filled_length)
    return f"[{bar}]"


def report_progress(completed_samples, progress_queue=None, shutdown_event=None):
    """Report progress to main process via queue (non-blocking)"""
    if progress_queue and (shutdown_event is None or not shutdown_event.is_set()):
        try:
            progress_queue.put(('progress', completed_samples), timeout=0.1)
        except:
            pass  # Queue full or other error, skip


def progress_monitor(progress_queue, total_expected, interval=5, shutdown_event=None):
    """
    Monitor progress from worker processes with tqdm-style visualization and graceful shutdown support
    
    Args:
        progress_queue: Queue for receiving progress updates
        total_expected: Total number of expected tasks
        interval: Minimum interval between progress reports (seconds)
        shutdown_event: Event to signal shutdown
    """
    completed = 0
    last_report = time.time()
    start_time = time.time()
    
    # Print initial state
    print(f"Progress: {create_progress_bar(0, total_expected)} 0/{total_expected} [00:00<?, 0.00it/s]", end="", flush=True)
    
    while completed < total_expected and (shutdown_event is None or not shutdown_event.is_set()):
        try:
            # Use very short timeout to be highly responsive to shutdown
            timeout = 0.5  # Much shorter timeout for faster shutdown response
            msg_type, value = progress_queue.get(timeout=timeout)
            
            if msg_type == 'progress':
                completed += value
                
                now = time.time()
                if now - last_report >= interval or completed >= total_expected:
                    elapsed = now - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    
                    # Calculate ETA
                    if rate > 0 and completed < total_expected:
                        eta_seconds = (total_expected - completed) / rate
                        eta_str = format_time(eta_seconds)
                    else:
                        eta_str = "?"
                    
                    # Create progress bar
                    progress_bar = create_progress_bar(completed, total_expected)
                    percentage = 100 * completed / total_expected if total_expected > 0 else 0
                    elapsed_str = format_time(elapsed)
                    rate_str = format_rate(rate)
                    
                    # Print tqdm-style progress (overwrite previous line)
                    progress_line = (f"\rProgress: {progress_bar} {completed}/{total_expected} "
                                   f"({percentage:.1f}%) [{elapsed_str}<{eta_str}, {rate_str}]")
                    print(progress_line, end="", flush=True)
                    
                    last_report = now
                    
            elif msg_type == 'stop':
                print(f"\n[PROGRESS] Received stop signal.", flush=True)
                break
                    
        except Empty:
            # Check for shutdown during timeout
            if shutdown_event and shutdown_event.is_set():
                print(f"\n[PROGRESS] Shutdown event detected during timeout.", flush=True)
                break
                
            # Timeout - print current status if we have progress
            if completed > 0:
                now = time.time()
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                
                # Calculate ETA
                if rate > 0 and completed < total_expected:
                    eta_seconds = (total_expected - completed) / rate
                    eta_str = format_time(eta_seconds)
                else:
                    eta_str = "?"
                
                # Create progress bar
                progress_bar = create_progress_bar(completed, total_expected)
                percentage = 100 * completed / total_expected if total_expected > 0 else 0
                elapsed_str = format_time(elapsed)
                rate_str = format_rate(rate)
                
                # Print tqdm-style progress (overwrite previous line)
                progress_line = (f"\rProgress: {progress_bar} {completed}/{total_expected} "
                               f"({percentage:.1f}%) [{elapsed_str}<{eta_str}, {rate_str}]")
                print(progress_line, end="", flush=True)
    
    # Final status report
    final_time = time.time()
    elapsed = final_time - start_time
    
    if completed > 0:
        rate = completed / elapsed if elapsed > 0 else 0
        status = "interrupted" if (shutdown_event and shutdown_event.is_set()) else "completed"
        progress_bar = create_progress_bar(completed, total_expected)
        percentage = 100 * completed / total_expected if total_expected > 0 else 0
        elapsed_str = format_time(elapsed)
        rate_str = format_rate(rate)
        
        # Final progress line
        progress_line = (f"\rProgress: {progress_bar} {completed}/{total_expected} "
                       f"({percentage:.1f}%) [{elapsed_str}, {rate_str}]")
        print(progress_line, flush=True)
        print(f"Progress monitor {status}: {completed}/{total_expected} tasks in {elapsed_str}")
    else:
        print(f"\nProgress monitor terminated: no tasks completed in {format_time(elapsed)}") 
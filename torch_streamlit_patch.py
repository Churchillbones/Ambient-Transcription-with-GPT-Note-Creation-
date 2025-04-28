"""
PyTorch-Streamlit Compatibility Patch

This module applies patches and configurations to ensure PyTorch works
correctly with Streamlit, particularly addressing memory issues and
multiprocessing compatibility problems.
"""

import os
import sys
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Apply patches and configurations
def apply_patches():
    """Apply necessary patches for PyTorch-Streamlit compatibility"""
    try:
        # Force single-threaded PyTorch operations
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        # Disable CUDA if not explicitly enabled
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            
        # Configure PyTorch memory allocation
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
            
        # Fix for multiprocessing issues in Windows
        if sys.platform == "win32":
            # Set multiprocessing start method to 'spawn' if possible
            import multiprocessing
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                # Method already set, probably okay
                pass
            
            # Disable forking subprocesses to prevent issues
            os.environ["PYTHONWARNINGS"] = "ignore"
            
        logger.info("PyTorch-Streamlit patch applied successfully")
        return True
    except Exception as e:
        logger.error(f"Error applying PyTorch-Streamlit patch: {e}")
        return False

# Apply patches immediately when module is imported
success = apply_patches()
if not success:
    logger.warning("Failed to apply some PyTorch-Streamlit compatibility patches")
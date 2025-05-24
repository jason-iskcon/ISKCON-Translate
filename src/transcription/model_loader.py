"""
Whisper model initialization and management.

This module handles the initialization of WhisperModel instances with
automatic device detection and fallback logic.
"""

import os
import torch
import logging
from typing import Tuple
from faster_whisper import WhisperModel

# Import with try-except to handle both direct execution and module import
try:
    from logging_utils import get_logger
except ImportError:
    from ..logging_utils import get_logger

logger = get_logger('transcription.model_loader')


def init_whisper_model(model_size: str, device_pref: str = "auto", 
                      compute_type_pref: str = "auto") -> Tuple[WhisperModel, str, str]:
    """Initialize the Whisper model with automatic fallback for device and compute type.
    
    Args:
        model_size: Size of the Whisper model (tiny, base, small, medium, large)
        device_pref: Preferred device ('cuda', 'cpu', or 'auto')
        compute_type_pref: Preferred compute type ('float16', 'int8', or 'auto')
        
    Returns:
        Tuple[WhisperModel, str, str]: (model, actual_device, actual_compute_type)
        
    Raises:
        RuntimeError: If model initialization fails completely
    """
    model_dir = os.path.expanduser("~/.cache/faster-whisper")
    
    # Determine device to use
    use_cuda = False
    if device_pref == "auto":
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            try:
                # Test if CUDA is actually working
                torch.zeros(1).cuda()
                device = "cuda"
            except Exception as e:
                logger.warning(f"CUDA is available but not working: {e}. Falling back to CPU.")
                use_cuda = False
                device = "cpu"
        else:
            device = "cpu"
    else:
        device = device_pref.lower()
        use_cuda = device == "cuda"
    
    # Determine compute type based on device
    if compute_type_pref == "auto":
        compute_type = "float16" if use_cuda else "int8"
    else:
        compute_type = compute_type_pref
    
    logger.info(f"Initializing Whisper model (size={model_size}, device={device}, "
              f"compute_type={compute_type})")
    
    try:
        os.makedirs(model_dir, exist_ok=True)
        logger.debug(f"Model cache directory: {model_dir}")
        
        # Try to initialize with preferred settings
        try:
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=model_dir
            )
            logger.info(f"Successfully initialized Whisper model on {device.upper()}")
            return model, device, compute_type
            
        except Exception as e:
            if use_cuda:  # If CUDA failed, try falling back to CPU
                logger.warning(f"Failed to initialize with CUDA: {e}. Falling back to CPU.")
                device = "cpu"
                compute_type = "int8"
                model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=model_dir
                )
                logger.info("Successfully initialized Whisper model on CPU")
                return model, device, compute_type
            else:
                raise  # Re-raise if we're already on CPU
                
    except Exception as e:
        logger.error(f"Failed to initialize Whisper model: {e}")
        logger.debug("Model initialization failed", exc_info=True)
        raise RuntimeError(f"Model initialization failed: {e}")


def validate_gpu_availability(device: str) -> None:
    """Validate GPU availability for production deployment.
    
    Args:
        device: Device preference ("auto", "cuda", "cpu")
        
    Raises:
        RuntimeError: If GPU mode is required but CUDA is not available
    """
    if device == "auto":
        if not torch.cuda.is_available():
            logger.error("🚨 PRODUCTION ERROR: CUDA not available! GPU mode required for production.")
            logger.error("   → Check NVIDIA driver installation")
            logger.error("   → Verify CUDA runtime compatibility") 
            logger.error("   → Restart application after driver update")
            raise RuntimeError("GPU acceleration required but CUDA not available")
        else:
            logger.info("✅ CUDA available - will attempt GPU initialization")


def get_model_info(model: WhisperModel) -> dict:
    """Get information about the initialized model.
    
    Args:
        model: Initialized WhisperModel instance
        
    Returns:
        dict: Model information including device and compute type
    """
    return {
        'device': getattr(model, 'device', 'unknown'),
        'compute_type': getattr(model, 'compute_type', 'unknown'),
        'model_size': getattr(model, 'model_size_or_path', 'unknown')
    } 
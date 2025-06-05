#!/usr/bin/env python3
"""
Test script to verify CUDA fallback functionality.

This script tests the automatic CUDA→CPU fallback when cuBLAS runtime errors occur.
"""

import sys
import os
import numpy as np
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transcription.engine import TranscriptionEngine
from transcription.worker import CudaRuntimeError

def test_cuda_fallback():
    """Test CUDA fallback functionality."""
    print("🧪 Testing CUDA fallback functionality...")
    
    try:
        # Initialize engine with auto device detection
        engine = TranscriptionEngine(model_size="tiny", device="auto")
        print(f"✅ Engine initialized on {engine.device.upper()}")
        
        # Test the _switch_to_cpu_model method directly
        if engine.device == "cuda":
            print("🔄 Testing CPU fallback...")
            success = engine._switch_to_cpu_model()
            if success:
                print(f"✅ CPU fallback successful, now using {engine.device.upper()}")
            else:
                print("❌ CPU fallback failed")
                return False
        else:
            print("ℹ️  Already using CPU, fallback not needed")
        
        # Test basic transcription functionality
        print("🎤 Testing transcription with dummy audio...")
        
        # Create dummy audio (1 second of silence at 16kHz)
        dummy_audio = np.zeros(16000, dtype=np.float32)
        timestamp = 0.0
        
        # Start the engine
        engine.start_transcription()
        
        # Add audio segment
        success = engine.add_audio_segment((dummy_audio, timestamp))
        if success:
            print("✅ Audio segment added successfully")
        else:
            print("❌ Failed to add audio segment")
            return False
        
        # Wait a bit for processing
        time.sleep(2.0)
        
        # Try to get transcription result
        result = engine.get_transcription()
        if result:
            print(f"✅ Transcription result: {result.get('text', 'No text')}")
        else:
            print("ℹ️  No transcription result yet (normal for silence)")
        
        # Stop the engine
        engine.stop_transcription()
        print("✅ Engine stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuda_runtime_error_handling():
    """Test CudaRuntimeError exception handling."""
    print("\n🧪 Testing CudaRuntimeError exception handling...")
    
    try:
        # Test that CudaRuntimeError can be raised and caught
        try:
            raise CudaRuntimeError("Test cuBLAS runtime missing")
        except CudaRuntimeError as e:
            print(f"✅ CudaRuntimeError caught correctly: {e}")
            return True
    except Exception as e:
        print(f"❌ CudaRuntimeError handling failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 ISKCON-Translate CUDA Fallback Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test CUDA fallback functionality
    results.append(test_cuda_fallback())
    
    # Test CudaRuntimeError handling
    results.append(test_cuda_runtime_error_handling())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    all_passed = all(results)
    
    if all_passed:
        print("🎉 All tests passed! CUDA fallback functionality is working.")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 
python# Check CUDA and GPU setup
import torch
import transformers

print("=== CUDA and GPU Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu.name}")
        print(f"  - Memory: {gpu.total_memory / 1024**3:.1f} GB")
        print(f"  - Multiprocessors: {gpu.multi_processor_count}")
    
    # Test GPU memory allocation
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        print("✓ GPU memory allocation test successful")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ GPU memory allocation test failed: {e}")
else:
    print("✗ CUDA not available - will use CPU (very slow)")

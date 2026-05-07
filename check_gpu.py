import torch

print("=" * 45)
print("  GPU CHECK")
print("=" * 45)

if torch.cuda.is_available():
    gpu     = torch.cuda.get_device_name(0)
    vram    = torch.cuda.get_device_properties(0).total_memory / 1024**3
    cuda_v  = torch.version.cuda
    torch_v = torch.__version__

    print(f"  Status  : READY")
    print(f"  GPU     : {gpu}")
    print(f"  VRAM    : {vram:.1f} GB")
    print(f"  CUDA    : {cuda_v}")
    print(f"  PyTorch : {torch_v}")

    # Quick tensor test on GPU
    x = torch.rand(1000, 1000).cuda()
    y = x @ x
    print(f"  Test    : GPU tensor multiply OK")
    print("=" * 45)
    print("  You are good to run training!")

else:
    print("  Status  : NO GPU FOUND")
    print("  Training will run on CPU (very slow)")

print("=" * 45)

import sys


def run_import_diagnostics():
    print("=== Running diagnostics ===")

    try:
        import torch
        print("[OK] PyTorch imported")

        cuda_available = torch.cuda.is_available()
        print(f"[INFO] CUDA available: {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"[INFO] GPU count: {device_count}")
            print(f"[INFO] GPU name: {device_name}")

            x = torch.tensor([1.0, 2.0]).cuda()
            y = x * 2
            print(f"[OK] GPU tensor computation works: {y}")
        else:
            print("[WARN] CUDA not available - running on CPU")

    except Exception as e:
        print(f"[ERROR] PyTorch test failed: {e}")

    try:
        import transformers
        from transformers import AutoTokenizer

        print("[OK] transformers imported")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer("Test sentence")
        print(f"[OK] Tokenizer works: {tokens['input_ids'][:5]}")

    except Exception as e:
        print(f"[ERROR] transformers test failed: {e}")

    print("=== Diagnostics complete ===\n")

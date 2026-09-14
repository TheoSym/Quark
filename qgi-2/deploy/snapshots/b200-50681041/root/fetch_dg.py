from huggingface_hub import snapshot_download
snapshot_download("nvidia/diffusiongemma-26B-A4B-it-NVFP4", local_dir="/models/diffusiongemma-26B-A4B-it-NVFP4", max_workers=16); print("FETCH_DONE", flush=True)

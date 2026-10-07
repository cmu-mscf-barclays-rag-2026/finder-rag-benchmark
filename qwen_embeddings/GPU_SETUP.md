# GPU environment and reproduction

The local machine has an NVIDIA GeForce RTX 5070 Laptop GPU with approximately 12 GB VRAM. A separate `.venv-qwen-gpu` environment was created; the earlier `.venv-person4` CPU environment was not changed.

Verified: PyTorch `2.7.0+cu128`, CUDA available, GPU matrix multiplication, and both embedding models' execution checks. Transformers is `4.51.3`. This uses the CUDA runtime supplied by the PyTorch wheel, with the existing NVIDIA driver (591.94).

## Fresh Windows environment

Run from your chosen project directory:

```powershell
python -m venv .venv-qwen-gpu
.venv-qwen-gpu/Scripts/python.exe -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
.venv-qwen-gpu/Scripts/python.exe -m pip install -r requirements.txt
.venv-qwen-gpu/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

Use that Python interpreter for the notebook kernel or the command-line runner. The notebook does not automatically change its kernel to the GPU environment. Installing a GPU wheel does not change code already executing in a different Python environment.

The completed experiment records its exact package versions, hardware, and encoding batch size in each `run.json`. All Qwen windows use batch size 4 and MiniLM uses batch size 16 in this run. For the same reproduction, use those values rather than the conservative command-line default of 1.

For a clean rerun, preserve delivered results separately and use an empty `results/dev` and `results/test`; completed runs are otherwise reused. Keep the frozen selection with the run that produced it. Do not use a new test evaluation to choose the input window.

Reference: [PyTorch 2.7 release: CUDA 12.8 and Blackwell support](https://pytorch.org/blog/pytorch-2-7/).

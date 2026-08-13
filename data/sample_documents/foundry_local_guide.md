# Microsoft Foundry Local Developer Guide

## What is Foundry Local?
Microsoft Foundry Local is an end-to-end local AI solution that provides a lightweight runtime and SDK for executing Large Language Models (LLMs) directly on user devices. It enables zero-latency, offline AI applications without sending telemetry or document contents to cloud API endpoints.

## Key Features
- **On-Device Inference**: Runs models locally on Windows, macOS, and Linux without internet connectivity.
- **Hardware Acceleration**: Automatically selects CPU, NPU (DirectML / Neural Processing Unit), or GPU acceleration depending on available hardware.
- **Optimized Catalog**: Access to lightweight models such as Phi-3.5 Mini, Phi-1.5, Qwen, and quantized embedding models.
- **Python & Multi-Language SDK**: Simple programmatic APIs for model loading, text generation, chat completion, and text embeddings.

## Installation & SDK Setup
To install the Python SDK for Microsoft Foundry Local:
```bash
pip install foundry-local-sdk
```

To verify runtime health and load a local model:
```python
from foundry_local import FoundryClient

client = FoundryClient()
model = client.load_model("phi-3.5-mini")
response = model.complete("Explain local AI in one sentence.")
print(response)
```

## Security & Privacy Advantage
Because Foundry Local operates entirely on-device, sensitive personal documents, proprietary manuals, and enterprise notes remain completely private within the user's local file system.

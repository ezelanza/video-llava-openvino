---
license: apache-2.0
base_model: llava-hf/LLaVA-NeXT-Video-7B-hf
tags:
- openvino
- llava
- multimodal
- video
- visual-question-answering
---

# LLaVA-NeXT-Video OpenVINO Model

This is an OpenVINO optimized version of the LLaVA-NeXT-Video-7B-hf model.

## Model Description
- **Base Model**: llava-hf/LLaVA-NeXT-Video-7B-hf
- **Optimization**: Converted to OpenVINO format for efficient inference
- **Size**: ~7B parameters

## Usage

```python
from optimum.intel.openvino import OVModelForVisualCausalLM

model = OVModelForVisualCausalLM.from_pretrained("YOUR_USERNAME/llava-next-video-openvino")
```

## License
This model inherits the license from the original LLaVA-NeXT model.

# Violet: Arabic Image Captioning
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahmood-Anaam/violet/blob/main/notebooks/inference_demo.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahmood-Anaam/violet/blob/main/notebooks/features_extraction_demo.ipynb)

## Overview

Violet is a vision-language model designed to generate high-quality Arabic image captions. Built around a **Gemini Decoder** and a pretrained transformer, Violet bridges the gap between computer vision and natural language processing (NLP) for Arabic. The repository provides a simple, effective, and streamlined pipeline that can handle a variety of image formats and produce descriptive captions in Arabic with minimal effort.

## Key Features:

1. **Arabic Image Captioning**: Generate high-quality captions for images in Arabic.
2. **Visual Feature Extraction**: Extract image features for integration into vision-language models or downstream tasks.
3. **Mixed Input Support**: Handle batches of images in various formats, such as URLs, file paths, NumPy arrays, PyTorch tensors, and PIL Image objects.
4. **Pretrained Model**: Leverages a robust pretrained model, requiring no additional training

## Installation:

#### Option 1: Install via `pip`
```bash
pip install git+https://github.com/Mahmood-Anaam/violet.git
```

#### Option 2: Clone Repository and Install in Editable Mode
```bash
!git clone https://github.com/Mahmood-Anaam/violet.git
%cd violet
!pip install -e .
```

#### Option 3: Use Conda Environment
```bash
conda env create -f environment.yml
conda activate violet

!git clone https://github.com/Mahmood-Anaam/violet.git
%cd violet
!pip install -e .
```

## Quickstart:

### Generate Captions for Images
```python
from violet.pipeline import VioletImageCaptioningPipeline
from violet.configuration import VioletConfig

# Initialize the pipeline
pipeline = VioletImageCaptioningPipeline(VioletConfig)

# Caption a single image
caption = pipeline("http://images.cocodataset.org/val2017/000000039769.jpg")
print(caption)

# Caption a batch of images
images = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "/path/to/local/image.jpg",
    np.random.rand(224, 224, 3),
    torch.randn(3, 224, 224),
    Image.open("/path/to/pil/image.jpg"),
]
captions = pipeline(images)
for caption in captions:
    print(caption)
```

## Additional Capabilities:

### Feature Extraction (Optional)
If needed, extract visual features for further processing:
```python
# Extract features from a single image
features = pipeline.generate_features("http://images.cocodataset.org/val2017/000000039769.jpg")
print(features.shape)

# Extract features for a batch of images
batch_features = pipeline.generate_features(images)
print(batch_features.shape)
```

### Caption Generation from Precomputed Features
Generate captions from extracted visual features:
```python
captions = pipeline.generate_captions_from_features(features)
for caption in captions:
    print(caption)
```

## Example Usage in Google Colab
Interactive Jupyter notebooks are provided to demonstrate Violet's capabilities. You can open these notebooks in Google Colab:

- [Image Captioning Demo](https://github.com/Mahmood-Anaam/violet/blob/main/notebooks/inference_demo.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahmood-Anaam/violet/blob/main/notebooks/inference_demo.ipynb)
- [Feature Extraction Demo](https://github.com/Mahmood-Anaam/violet/blob/main/notebooks/features_extraction_demo.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahmood-Anaam/violet/blob/main/notebooks/features_extraction_demo.ipynb)

from setuptools import setup, find_packages

setup(
    name="violet",
    version="0.1.0",
    description="Violet: A Vision-Language Model for Arabic Image Captioning with Gemini Decoder",
    author="Mahmood Anaam",
    author_email="eng.mahmood.anaam@gmail.com",
    url="https://github.com/Mahmood-Anaam/violet",
    license="MIT",
    packages=find_packages(exclude=["notebooks", "assets", "scripts", "tests"]),
    install_requires = [
        "huggingface-hub==0.27.0",
        "pillow==11.1.0",
        "PyYAML==6.0.2",
        "timm==1.0.12",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "tqdm==4.67.1",
        "transformers==4.47.1",
        "twine==6.0.1",
        "yacs==0.1.8",
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)

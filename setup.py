from setuptools import find_packages, setup

setup(
    name="mfp2eft",
    version="0.1.0",
    description="Parameter- and Energy-Efficient Fine-Tuning",
    license_files=["LICENSE"],
    license="Apache",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9.0",
    install_requires=[
        "numpy>=1.17",
        "packaging>=20.0",
        "psutil",
        "pyyaml",
        "torch>=1.13.0",
        "transformers",
        "tqdm",
        "accelerate>=0.21.0",
        "safetensors",
        "huggingface_hub>=0.20.0",
        "peft>=0.11.1",
    ],
)

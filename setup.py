from setuptools import setup, find_packages

setup(
    name="medical-image-diagnosis-vit",
    version="1.0.0",
    description="Production-quality Medical Image Diagnosis System using Vision Transformers",
    author="Jash",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "albumentations>=1.3.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "medvit-train=scripts.train:main",
            "medvit-eval=scripts.evaluate:main",
            "medvit-infer=scripts.infer:main",
        ]
    },
)

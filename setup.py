import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simple-ai-benchmarking",
    version="0.2.0",
    author="Timo Leitritz",
    author_email="placeholder@example.com",
    description="A package for benchmarking various AI models in a simple way.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TimoIllusion/simple-ai-benchmarking",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy",
        "tqdm",
        "psutil",
        "pandas",
        "tabulate",
    ],
    extras_require={
        "pt": [
            "torch>=1.0.0", 
            "torchvision", 
            "torchaudio", 
            ],
        "tfdml": [
            "tensorflow-cpu==2.10.0", 
            "tensorflow-directml-plugin"
            ],
        "tf": [
            "tensorflow>=2.3.0"
            ], 
        "xlsx":[
            "openpyxl"
        ]
    },
    entry_points={
        "console_scripts": [
            "sai-tf = simple_ai_benchmarking.benchmark:run_tf_benchmarks",
            "sai-pt = simple_ai_benchmarking.benchmark:run_pt_benchmarks"
        ]
    },
)

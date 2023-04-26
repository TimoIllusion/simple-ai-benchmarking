import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simple-ai-benchmarking",
    version="0.1.0",
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
        "tensorflow==2.10.0",
        "tensorflow-addons==0.20.0",
        "tensorflow_datasets==4.8.0",
    ],
    entry_points={
        "console_scripts": [
            "sai_benchmark = simple_ai_benchmarking.benchmark:main"
        ]
    }
)
from setuptools import setup, find_packages

setup(
    name="pioneer-fdm",
    version="0.1.0",
    author="Pioneer Aerospace Research",
    description="High-fidelity, differentiable flight dynamics model for NVIDIA Warp with JSBSim parity.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Pioneer-DRL/pioneer-fdm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "warp-lang>=1.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "jsbsim>=1.2.0",
            "pytest",
        ],
    },
)

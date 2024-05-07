import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="whispersg",
    py_modules=["whispersg"],
    version="1.0.0",
    description="Enhancement of Max Bain's WhiperX for use in a Singaporean context",
    python_requires=">=3.8",
    author="Max Bain, Bhagat Sai",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    entry_points={
        "console_scripts": ["whispersg=demo.transcribe:cli","whispersg_live=demo.live_transcribe:cli"],
    },
    include_package_data=True,
    extras_require={"dev": ["pytest"]},
)

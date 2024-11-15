import setuptools

setuptools.setup(
    name="pyvallib",
    version="0.0.1",
    author="Mike",
    author_email="",
    description="A package for common valuation methodologies in Fair Value and Fair Market Value measurements.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=["numpy>=1.24.2", "pandas>=2.1.3", "scipy>=1.9.3"],
)

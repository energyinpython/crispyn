import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="crispyn",
    version="0.0.2",
    author="Aleksandra Bączkiewicz",
    author_email="aleksandra.baczkiewicz@phd.usz.edu.pl",
    description="CRIteria Significance determining in PYthoN - The Python 3 Library for determining criteria weights for MCDA methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/energyinpython/crispyn",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.4",
	install_requires=['numpy'],
)
from setuptools import setup, find_packages

setup(
    name="Lapyrinth",
    version="1.13",
    packages=find_packages(),
    install_requires=[
        "numpy==2.0.0",
        "Pillow==10.2.0",
    ],
    author="Piétôt",
    author_email="baptiste.blasquez@gmail.com",
    maintainer="Piétôt",
    description="A program capable of creating mazes with many different algorithms and solving them with different pathfinders.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Pietot/Lapyrinth",
    keywords=["maze", "pathfinding", "algorithm", "labyrinth", "pathfinder", "maze generator"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

import setuptools
import pathlib

HERE = pathlib.Path(__file__).parent

long_description = (HERE / "README.md").read_text(encoding="utf-8")

def load_requirements(path="requirements.txt"):
    reqs = []
    for line in (HERE / path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs

setuptools.setup(
    name="pybullet-planning",
    version="0.1.0",
    description="Motion-planning utilities for PyBullet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="yyin34",
    author_email="yyin34@jhu.edu",
    url="https://github.com/yifanyin11/pybullet_planning.git",

    packages=setuptools.find_packages(
        exclude=["archive", "examples", "pybullet_tools.egg-info", "tests*"]
    ),

    python_requires=">=3.7",
    install_requires=load_requirements(),

    include_package_data=True,  # if you have non‑.py files you want included
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
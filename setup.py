from setuptools import find_packages, setup

setup(
    name="simscene-agent",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    extras_require={"demo": ["numpy>=1.24", "Pillow>=9.0"]},
    entry_points={"console_scripts": ["simscene=simscene_agent.cli:main"]},
)

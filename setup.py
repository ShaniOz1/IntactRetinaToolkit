from setuptools import setup, find_packages

setup(
    name='IntactRetinaToolkit',
    version='0.1.0',
    description='Loading, analysis and visualisation of retinal recordings (Intan RHS + MEA EDF)',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pyedflib',
        'pyintan',
        'tqdm',
    ],
)

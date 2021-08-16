"""
Setup file. Install the mlframework with `pip install mlframework`
"""
from setuptools import setup, find_packages

REQUIREMENTS: dict = {
    'core': [
        'tqdm',
        'requests',
        'psutil',
        'npTDMS',
        'h5py',
        'matplotlib',
        'compress_pickle',
        'scipy',
        'pyyaml',
        'pandas',
        'coloredlogs',
        'tsfresh',
        'scikit-learn>=0.24.2',
        'tensorflow>=2.5.0',
    ],
    'test': [
        'pytest',
        'mypy',
        'pylint'
    ],
    'dev': [
    ],
    'doc': [  # sphinx (?)
    ],
}

setup(
    name='mlframework',
    version='0.0.1',
    description='Machine learning framework for a broad usecase',
    author='CERN|TE-MPE-CB',
    url='https://gitlab.cern.ch/machine-protection-ml/mlframework/',
    python_requires='>=3.8, <4',
    setup_requires=['wheel'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=REQUIREMENTS['core'],
    extras_require={
        **REQUIREMENTS,
        'dev': [req
                for extra in ['dev', 'test', 'doc']
                for req in REQUIREMENTS.get(extra, [])],
        'all': [req for reqs in REQUIREMENTS.values() for req in reqs],
    },
    packages=find_packages(),
)

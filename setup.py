from setuptools import setup, find_packages

REQUIREMENTS: dict = {
    'core': [
        'numpy',
        'pandas',
        'datetime',
        'matplotlib',
        "nptdms",
        "h5py",
        "psutil",
        "tqdm",
    ],
    'test': [
        'pytest',
    ],
    'dev': [
        "python-telegram-handler",
    ],
    'doc': [  # sphinx
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
        # The 'dev' extra is the union of 'test' and 'doc', with an option
        # to have explicit development dependencies listed.
        'dev': [req
                for extra in ['dev', 'test', 'doc']
                for req in REQUIREMENTS.get(extra, [])],
        # The 'all' extra is the union of all requirements.
        'all': [req for reqs in REQUIREMENTS.values() for req in reqs],
    },
    packages=find_packages(),
    entry_points={
        'console_scripts': ['mlframework=src:main'],
    },
)

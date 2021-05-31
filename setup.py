"""
setup file
"""
from setuptools import setup, find_packages

REQUIREMENTS: dict = {
    'core': [
        'tqdm >= 4.60.0',
        'python_telegram_handler >= 2.2.1',
        'requests >= 2.25.1',
        'psutil >= 5.8.0',
        'npTDMS >= 1.2.0',
        'h5py >= 3.2.1',
        'matplotlib >= 3.4.1',
        'compress_pickle >= 2.0.1',
        'scipy >= 1.6.3',

    ],
    'test': [
        'pytest >= 6.2.4',
    ],
    'dev': [
        'telegram_handler >= 1.3',
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
        # The 'dev' extra is the union of 'test' and 'doc', with an option
        # to have explicit development dependencies listed.
        'dev': [req
                for extra in ['dev', 'test', 'doc']
                for req in REQUIREMENTS.get(extra, [])],
        # The 'all' extra is the union of all requirements.
        'all': [req for reqs in REQUIREMENTS.values() for req in reqs],
    },
    packages=find_packages(),
)

from setuptools import setup, find_packages

setup(
    name='SurvRNC',
    version='0.1.0',
    description='SurvRNC: Learning Ordered Representations for Survival Prediction using Rank-N-Contrast',
    author='Numan Saeed',
    author_email='numan.saeed@mbzuai.ac.ae',
    url='https://github.com/numanai/SurvRNC',
    packages=find_packages(),
    install_requires=[
        'argparse',
        'numpy',
        'torch',
        'monai',
        'torchvision',
        'pandas',
        'scikit-learn',
        'tqdm',
        'pycox',
        'sklearn-pandas',
        'torchtuples',
        'wandb',
        'umap-learn',
        'lifelines',
        'seaborn',
        'matplotlib',
        'simpleitk',
        'PyYAML'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)

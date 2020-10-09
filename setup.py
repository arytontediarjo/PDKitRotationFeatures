from setuptools import setup

setup(
    name='PDKitRotationFeatures',
    version='0.0.2',
    description='package wrapper for gait feature pipeline',
    author='Aryton Tediarjo and Larsson Omberg',
    author_email='aryton.tediarjo@sagebase.org',
    packages=['PDKitRotationFeatures'],
    install_requires=["numpy",
                      "pandas==1.0.3",
                      "scipy",
                      "pdkit==1.2",
                      "scikit-learn",
                      "tsfresh",
                      "future",
                      "matplotlib",
                      "pandas_validator"]
)

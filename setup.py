import setuptools

with open('README.md', 'r') as infile:
    long_description = infile.read()

setuptools.setup(
    name='simplexgrid',
    version='0.3.3',
    author='Christer Dreierstad',
    author_email='christerdr@outlook.com',
    description='Create grid-like Simplex noise',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chdre/simplexgrid',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'noise_randomized'],
    include_package_data=True,
)

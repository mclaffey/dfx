import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dfx",
    version="0.0.4",
    author="Mike Claffey",
    author_email="mikeclaffey@yahoo.com",
    description="Navigate small data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mclaffey/dfx",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',

    # to include docs
    # include_package_data=True, # not used
    data_files=[('docs', ['docs/*'])],

    # executable
    entry_points={
        'console_scripts': [
            'dfx = dfx.__main__:main',
        ],
    }

)

# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import setuptools, os
from setuptools import find_packages

setuptools.setup(
    name = "mmlspark",
    version = os.environ["MML_VERSION"],
    description = "Microsoft ML for Spark",
    long_description = "The Microsoft ML for Apache Spark package provides a python API to scala.",
    license = "MIT",
    packages = find_packages(),

    # Project's main homepage.
    url = "https://github.com/Azure/mmlspark",
    # Author details
    author = "Microsoft",
    author_email = os.environ["SUPPORT_EMAIL"],

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "Topic :: Software Development :: Datascience Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3"
    ],

    zip_safe = True,

    package_data = {"mmlspark": ["../LICENSE.txt", "../README.txt"]}
)

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

extensions = [
    Extension("nmf_torch.cylib.nnls_bpp_utils", ["ext_modules/nnls_bpp_utils.pyx"],
              extra_compile_args=["-std=c++11"]),
]

setup(
    name="nmf-torch-dl",
    use_scm_version=True,
    description="Fork of nmf-torch (PyTorch NMF) with a DataLoader minibatch mode for memory-efficient GPU usage.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/ymo6/nmf-torch",
    project_urls={
        "Source": "https://github.com/ymo6/nmf-torch",
        "Upstream": "https://github.com/lilab-bcb/nmf-torch",
    },
    author="Yiming Yang, Bo Li",
    author_email="cumulus-support@googlegroups.com",
    maintainer="Alexandra Mo",
    maintainer_email="ymo@stanford.edu",
    license="BSD-3-Clause",
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="matrix factorization",
    packages=find_packages(include=["nmf_torch", "nmf_torch.*"]),
    install_requires=["numpy", "torch>=1.8.0"],
    ext_modules=cythonize(extensions),
    python_requires=">=3.9",
)

import subprocess
subprocess.run("py -3.10 setup.py bdist --cythonize")
subprocess.run("py -3.11 setup.py bdist --cythonize")
subprocess.run("py -3.12 setup.py bdist --cythonize")
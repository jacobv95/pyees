from setuptools import setup
from setuptools_cythonize import get_cmdclass

setup(
    python_requires = '>=3.12',
    name='pyees',
    packages=['pyees'],
    version='2.4.2',
    cmdclass=get_cmdclass(),
    options={
        'build_py':
            {'exclude_cythonize': [
                'pyees.testFit',
                'pyees.testProp',
                'pyees.testPyees',
                'pyees.testSheet',
                'pyees.testSolve',
                'pyees.testUnit',
                'pyees.testVariable',
                'pyees.profilePyees',
                'pyees.profileFit',
                'pyees.profileWindTunnelTest'
                ]}
    },
    license='MIT',
    description='EES but for python. Pyees can be used do perform uncertanty (error) propagation. Furthermore, it can solve nonlinear systems of equations and look up material properties.',
    author='Jacob Vestergaard',
    author_email='jacobvestergaard95@gmail.com',
    url='https://github.com/jacobv95/pyees',
    download_url='https://github.com/jacobv95/pyees/archive/refs/tags/v1.0.tar.gz',
    keywords=['python', 'data processing', 'uncertanty', 'EES'],
    install_requires=[            # I get to this in a second
        'numpy', 'scipy', 'openpyxl', 'xlrd', 'pyfluids', 'xlwt', 'plotly', 'matplotlib', 'python_calamine', 'csv'
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        # Define that your audience are developers
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3',
    ],
)

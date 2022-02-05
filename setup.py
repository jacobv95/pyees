from distutils.core import setup
setup(
    name='pyees',
    packages=['pyees'],
    version='1.2',
    license='MIT',
    description='Python package to replace EES',
    author='Jacob Vestergaard',
    author_email='jacobvestergaard95@gmail.com',
    url='https://github.com/jacobv95/pyees',
    download_url='https://github.com/jacobv95/pyees/archive/refs/tags/v1.tar.gz',
    keywords=['python', 'EES'],
    install_requires=[            # I get to this in a second
        'pandas',
        'scipy',
        'CoolProp',
        'PyPDF2',
        'reportlab'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',  # Specify which pyhton versions that you want to support
    ],
)

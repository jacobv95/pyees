import subprocess
subprocess.run("twine upload --config-file ./.pypirc -r pypi --skip-existing dist/*")
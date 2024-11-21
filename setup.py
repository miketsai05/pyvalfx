from setuptools import setup

if __name__ == "__main__":
    setup()


# python -m build
# twine upload --repository testpypi dist/*
# twine upload dist/*
# twine upload --skip-existing dist/*
# python setup.py clean --all

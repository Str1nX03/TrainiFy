from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]:
    '''
    This gives a list of all the dependencies provided in the file requirements.txt
    '''
    requirements = []
    with open(file_path) as file_obj:

        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        
    return requirements

setup(
    name = "TrainiFy",
    version = "0.0.0.0",
    author = "Str1nX03",
    author_email = "draveen2210@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)
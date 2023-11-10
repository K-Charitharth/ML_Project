from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def package_required(path:str)->List[str]:
    '''
    This method helps in returning all the requirements
    '''
    requirements = []
    with open(path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace('\n','') for i in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='ML_Project',
    version='0.0.1',
    author='Charitharth',
    author_email='charitharthkothakota1@gmail.com',
    packages=find_packages(),
    install_requires=package_required('requirements.txt')
    )    

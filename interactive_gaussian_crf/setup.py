from setuptools import setup

setup(
   name='interactive_gaussian_crf',
   version='1.0',
   description='',
   author='Jannik Irmai',
   author_email='jannik.irmai@tu-dresden.de',
   packages=['gaussian_crf'],
   install_requires=['matplotlib', 'networkx'],  # external packages acting as dependencies
)

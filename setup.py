from setuptools import setup, find_packages

setup(
   name='asteroseismology',
   description='A toolkit for various seismology analysis',
   author='Yaguang Li',
   author_email='yaguangl@hawaii.edu',
   packages=['asteroseismology'],  #same as name
   package_data={
		'asteroseismology' : ['io/*', 'globe/*', 'tools/*'],
   },
   install_requires=['wheel', 'scipy', 'numpy', 'pandas', 'matplotlib'], #external packages as dependencies
)

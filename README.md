This is a repository of all the codes I have written for asteroseismology projects. Using git is better for version control. So I started using them from now! -- Dec 5, 2018

# Introduction
hmmm

# Package dependencies
1. `numpy`, `scipy`, and `matplotlib`. Try to install with Anaconda. To upgrade (there are certain functions in scipy that requires a later version), try `pip install package_name --upgrade`.
2. `emcee` and `corner` while using modefit.py. Try `pip install package_name` to install. The `emcee` used in this package requires version under 3.0 because version 3.0 sets a different way to enable parallel computing while not officialy released yet (Mar 7, 2019). Use `print(emcee.__version__)` to check.

# Installation
1. Git clone to a directory.
2. Insert these lines in your code to import the package.
		
		import sys
		sys.path.append(the_path_where_you_store_the_module)
		import asteroseismology as se
		
3. Use any functions in the package by simply typing `se.the_function_you_want_to_use`.

# General suggestions
If you are working on something about asteroseismology, please create a new branch under this respository. We can share some pretty handful tools, good for everyone!



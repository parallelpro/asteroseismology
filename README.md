# Introduction
This is a repository of essential programs I have written for asteroseismology projects. Using git is better for version control. 

# Package dependencies
`numpy`, `scipy`, and `matplotlib`. Try to install with Anaconda. To upgrade (there are certain functions in scipy that requires a later version), try `pip install package_name --upgrade`.


# Installation
1. Git clone to a local directory.
2. Insert the following lines in your Python program to import the package.
		
		import sys
		sys.path.append(the_path_where_you_store_the_module)
		import asteroseismology as se

Alternatively, set PYTHONPATH in the shell to avoid using `sys.path.append` every time. Here's a bash/zsh example:

		export PYTHONPATH="the_path_where_you_store_the_module"

3. Use any function in the package by simply typing `se.the_function_you_want_to_use`.



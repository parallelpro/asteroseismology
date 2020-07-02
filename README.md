# Introduction
This is a repository of essential codes I have written for asteroseismology projects. Using git is better for version control. 

# Package dependencies
`numpy`, `scipy`, and `matplotlib`. Try to install with Anaconda. To upgrade (there are certain functions in scipy that requires a later version), try `pip install package_name --upgrade`.


# Installation
1. Git clone to a directory.
2. Insert these lines in your code to import the package.
		
		import sys
		sys.path.append(the_path_where_you_store_the_module)
		import asteroseismology as se
Alternatively, add the path of this directory to PYTHONPATH in the shell profile. For a bash/zsh example:

		export PYTHONPATH="the_path_where_you_store_the_module"


3. Use any functions in the package by simply typing `se.the_function_you_want_to_use`.



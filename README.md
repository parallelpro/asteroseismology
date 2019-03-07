This is a repository of all the codes I have written for asteroseismology projects. Using git is better for version control. So I started using them from now! -- 05/12/2018

# Introduction
hmmm

# Package dependencies
1. `numpy`, `scipy`, and `matplotlib`. Try to install with Anaconda. To upgrade (there are certain functions in scipy that requires a later version), try `pip install package_name --upgrade`.
2. `emcee` and `corner` while using modefit.py. Try `pip install package_name` to install.

# Installation
1. Git clone to your directory.
2. To import the package in your code, type
`import sys`
`sys.path.append(the_path_where_you_store_the_module)`
`import asteroseismology as se`.
3. Use any functions in the package by simply typing `se.the_function_you_want_to_use`.

# General suggestions
If you are working on something about asteroseismology, please create a new branch under this respository. We can share some pretty handful tools, good for everyone!



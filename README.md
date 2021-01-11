# Introduction
This is a repository of essential programs I have written for asteroseismology projects. Using git is better for version control. 

# Package dependencies
`numpy`, `scipy`, `matplotlib`, `corner`, et al. 


# How to install
1. Git clone to a local directory.
2. a) Insert the following lines in your Python program to import the package.
		
		import sys
		sys.path.append(the_path_where_you_store_the_module)
		import asteroseismology as se
        
2. b) Alternatively, set PYTHONPATH in the shell to avoid using `sys.path.append` every time. Here's a bash/zsh example:

		export PYTHONPATH="the_path_where_you_store_the_module"

3. Use any function in the package by simply typing `se.the_function_you_want_to_use`.


# What it can do
1. 'fitter/': wrapping routines for various fitting samplers 
2. 'modelling/': stellar grid modelling, using classic or seismic (oscillation frequencies) constraints
3. 'solarlike/': extracting global seismic parameters from the power spectra of a solar-like oscillating star
4. 'tools/': useful functions

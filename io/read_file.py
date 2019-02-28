#!/usr/bin/env/ ipython
# coding: utf-8


import numpy as np

__all__ = ["read_yu_power_spectra"]

def read_yu_power_spectra(filepath: str): 
	'''
	Read .seis power spectrum provided by Yu+2018 and return a dictionary.

	Input:
	filepath: a string.

	Output:
	spec: a dictionary containing all data.

	'''
	if not filepath[-5:] == ".seis":
		raise ValueError("The text data file should have suffix '.seis'.")

	keys, values = [], []
	rfile = open(filepath, "r")
	# row 1 - 12, 4 groups
	for i in range(0,4):
		s1 = rfile.readline().replace("\n", "")[1:].split("|")
		s2 = rfile.readline().replace("\n", "").split(" ")
		s2 = [float(x) for x in s2 if x != ""]
		#if i == 0: 
		#	s1[0] = "numax1"
		#	s1[1] = "dnu1"
		#if i == 3:
		#	s1[1] = "numax3"
		#	s1, s2 = s1[1:], s2[1:]
		if i == 2:
			s1[1] = "numax2"
			s1[3] = "dnu2"
			s1, s2 = s1[1:], s2[1:]
			keys.extend(s1)
			values.extend(s2)
		rfile.readline()

	# row 13 - 21, Harvey models
	rfile.readline() #13
	for i in range(7): #14-20
		rfile.readline()
	rfile.readline() #21

	# row 22 - end, power spectrum
	rfile.readline() #22
	datacube = np.loadtxt(rfile, dtype=float)
	keys.extend(["freq", "psd"])
	values.extend([datacube[:,0], datacube[:,1]])

	# spec dictionary
	spec = {}
	if len(keys) != len(values):
		raise ValueError("len(keys) != len(values)")
	else:
		for i in range(len(keys)):
			spec[keys[i]] = values[i]

	return spec



#!/usr/bin/env/ ipython
# coding: utf-8


import numpy as np
import urllib
import re

__all__ = ["refresh_bibtex_entry"]

def refresh_bibtex_entry(inputfile:str, outputfile:str):
	'''
	Refresh a bibliography file to entries provided by ads service while 
	maintaining the old citekey. Entries are matched by the doi number.

	Input:
	inputfile, outputfile

	Output:
	A file with refreshed entries.

	'''

	# # step 1: extract from a disgusting bibliography. Doi or arxiv.
	f = open(inputfile, "r")
	data = f.read()
	f.close()

	citekey, identifier, doiorbibcode = [], [], []
	data_lines = data.split("\n")
	index_entries = []
	for i, line in enumerate(data_lines):
		if len(line) != 0:
			if line[0]=="@": index_entries.append(i)
	index_entries.append(len(data_lines))

	for i in range(len(index_entries)-1):
		entries = data_lines[index_entries[i]:index_entries[i+1]]
		citekey.append(entries[0].split("{")[1][:-1])
		ifdoi = False
		for j, entry in enumerate(entries): 
			if len(entry) != 0:
				if entry[0:3]=="doi": 
					identifier.append(entry[entry.find("{")+1:entry.find("}")])
					doiorbibcode.append(1)
					ifdoi = True
					# print("Success ",identifier[i])
				if entry[0:6]=="author":
					authorA = entry[entry.find("{")+1:entry.find("{")+2]
				if entry[0:4]=="year":
					year = entry[entry.find("{")+1:entry.find("}")]
		if not ifdoi:
			for j, entry in enumerate(entries): 
				if len(entry) != 0:
					if entry[0:3]=="url": 
						url_split = entry[entry.find("{")+1:entry.find("}")].split("/")
						if url_split[2][0:5] == "adsab" :
							if url_split[-1].find("bibcode=") != -1: 
								s = url_split[-1].split("bibcode=")[1].split("&")[0]
							else:
								s = url_split[-1]
							identifier.append(s)
							doiorbibcode.append(2)
							# print("ads Success ",identifier[i])
						elif url_split[2][0:5] == "arxiv" :
							s = "".join(url_split[-1].split("v")[0].split("."))
							identifier.append(year+"arXiv"+s+authorA)
							doiorbibcode.append(2)
							# print("arxiv Success ",identifier[i])
						else:
							print("Could not find an identifier for "+citekey[i])
							identifier.append(-1)
							doiorbibcode.append(-1)


	# # step 2: send a query
	index = np.array(identifier) == "-1"
	citekey = np.array(citekey)[~index]
	identifier = np.array(identifier)[~index]
	doiorbibcode = np.array(doiorbibcode)[~index]
	concate = []
	for i in range(len(doiorbibcode)):
		if doiorbibcode[i] == 1: concate.append("doi="+identifier[i])
		if doiorbibcode[i] == 2: concate.append("bibcode="+identifier[i])
	f = open(outputfile, "w")
	for i in range(int(len(concate)/100)+1):
		url = "http://adsabs.harvard.edu/cgi-bin/abs_connect?"+"&".join(concate[i*100:(i+1)*100]) #doi="+"10.3847/1538-4365/aaccfb"
		url += "&data_type=BIBTEX&nocookieset=1"	#&db_key=AST
		data = urllib.request.urlopen(url).read().decode()
		f.write(data)
	f.close()


	# # step 3: replace with the original citekey
	f = open(outputfile, "r")
	data = f.read()
	f.close()

	# find new dois and bibcodes from the data file
	new_bibcodes = re.compile("@.*\{\d{4}.*").findall(data)
	# index_entries = []
	# for i in range(len(new_bibcodes)):
	# 	index_entries.append(data.find(new_bibcodes[i]))
	# index_entries.append(len(data))

	print("Find "+str(len(new_bibcodes))+" out of "+str(len(citekey))+" entries in ads database.")

	index_citekey_accessed = []
	# itereate every entry in the new data and replace the citekey
	for i in range(len(new_bibcodes)):
		index_entry_1 = data.find(new_bibcodes[i])
		if i != len(new_bibcodes)-1:
			index_entry_2 = data.find(new_bibcodes[i+1])
		else:
			index_entry_2 = -1
		tdata = data[index_entry_1:index_entry_2]
		theader = new_bibcodes[i]
		tdoi_list = re.compile("doi.*\n").findall(tdata)
		# print(tdata)
		# print(tdoi_list)
		replace_index1 = data.find(theader) + 1 + theader.find("{")
		replace_index2 = data.find(theader) + 1 + theader.find(",")
		if len(tdoi_list) != 0:
			tdoi = tdoi_list[0][7:-3]
			# print(data[replace_index1:replace_index2], tdoi)
			index = np.where(identifier == tdoi)[0][0]
			index_citekey_accessed.append(index)
			tcitekey = citekey[index]
			data = data.replace(data[replace_index1:replace_index2], tcitekey+",")
		else:
			tbibcode = urllib.parse.quote(data[replace_index1:replace_index2])[:-3]
			# print(data[replace_index1:replace_index2], tbibcode, identifier[1])
			index = np.where(identifier == tbibcode)[0][0]
			index_citekey_accessed.append(index)
			tcitekey = citekey[index]
			data = data.replace(data[replace_index1:replace_index2], tcitekey+",")


	
	# print those who didn't get an entry from ads.
	index_citekey_accessed = np.sort(np.array(index_citekey_accessed))
	index_citekey_naccessed = np.setdiff1d(np.arange(len(citekey)), index_citekey_accessed)
	print(citekey[index_citekey_naccessed])
	print(identifier[index_citekey_naccessed])

	# little modification:
	#2018bellinger-phd-inverse-problem
	index = data.find("school = {Max Planck Institute for Solar System")
	data = data.replace(data[index: index+109], "school = {Max Planck Institute for Solar System Research},")

	#2018ting++100000-rc-lamost
	index = data.find("A Large and Pristine Sample of Standard Candles across the Milky Way: ")
	data = data.replace(data[index: index+122], "A Large and Pristine Sample of Standard Candles across the Milky Way: {$\sim$}100,000 Red Clump Stars with 3\% Contamination")


	# # step 4: output
	f = open(outputfile, "w")
	f.write(data)
	f.close()

	return 






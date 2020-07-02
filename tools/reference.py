#!/usr/bin/env/ python
# coding: utf-8


import numpy as np
import urllib
import re
import ads as ads

__all__ = ["refresh_bibtex_entry"]

def refresh_bibtex_entry(inputfile:str, outputfile:str, newbibfile:str, ifadd:bool=True):
	'''
	Refresh a bibliography file to entries provided by ads service while 
	maintaining the old citekey. Entries are matched by the doi number.

	Input:
	inputfile, outputfile

	Output:
	A file with refreshed entries.

	'''

	# # step 1: extract from a disgusting bibliography. Doi or arxiv.
	f = open(inputfile, "r", encoding="utf8")
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
		citekey.append(entries[0].split("{")[1][:-2])
		ifdoi = False
		for j, entry in enumerate(entries): 
			if len(entry) != 0:
				entry=entry.replace(" ","").replace(",","")
				if entry[0:3]=="doi":
					identifier.append(entry[entry.find("{")+1:entry.find("}")])
					doiorbibcode.append(1)
					ifdoi = True
					# print(entry[entry.find("{")+1:entry.find("}")])
					# print("Success ",identifier[i])
				if entry[0:6]=="author":
					authorA = entry[entry.find("{")+1:entry.find("{")+2]
				if entry[0:4]=="year":
					year = entry[entry.find("{")+1:entry.find("}")]
		if not ifdoi:
			# for j, entry in enumerate(entries): 
				# if len(entry) != 0:
				# 	entry=entry.replace(" ","").replace(",","")
				# 	if entry[0:3]=="url": 
				# 		url_split = entry[entry.find("{")+1:entry.find("}")].split("/")
				# 		if url_split[2][0:5] == "adsab" :
				# 			if url_split[-1].find("bibcode=") != -1: 
				# 				s = url_split[-1].split("bibcode=")[1].split("&")[0]
				# 			else:
				# 				s = url_split[-1]
				# 			identifier.append(s)
				# 			doiorbibcode.append(2)
				# 			# print("ads Success ",identifier[i])
				# 		elif url_split[2][0:5] == "arxiv" :
				# 			s = "".join(url_split[-1].split("v")[0].split("."))
				# 			identifier.append(year+"arXiv"+s+authorA)
				# 			doiorbibcode.append(2)
				# 			# print("arxiv Success ",identifier[i])
				# 		else:
				# 			print("Could not find an identifier for "+citekey[i])
			identifier.append("-1")
			doiorbibcode.append("-1")

	# # # step 1.5: remove those already existed in outputfile
	# if ifadd:
	# 	f = open(outputfile, "r")
	# 	data = f.read()
	# 	f.close()

	# 	citekey_o = []
	# 	data_lines = data.split("\n")
	# 	index_entries = []
	# 	for i, line in enumerate(data_lines):
	# 		if len(line) != 0:
	# 			if line[0]=="@": index_entries.append(i)
	# 	index_entries.append(len(data_lines))

	# 	for i in range(len(index_entries)-1):
	# 		entries = data_lines[index_entries[i]:index_entries[i+1]]
	# 		citekey_o.append(entries[0].split("{")[1][:-1])

	# 	index_unique = np.isin(np.array(citekey), np.array(citekey_o))
	# 	citekey, identifier, doiorbibcode = np.array(citekey)[~index_unique], np.array(identifier)[~index_unique], np.array(doiorbibcode)[~index_unique]


	# # step 2: send a query
	index = np.array(identifier) == "-1"
	citekey = np.array(citekey)[~index]
	identifier = np.array([id.lower() for id in identifier])[~index]
	doiorbibcode = np.array(doiorbibcode)[~index]
	# # concate = []
	# # for i in range(len(doiorbibcode)):
	# # 	if doiorbibcode[i] == 1: concate.append("doi="+identifier[i])
	# # 	if doiorbibcode[i] == 2: concate.append("bibcode="+identifier[i])
	# # data = ""
	# # for i in range(int(len(concate)/100)+1):
	# # 	url = "https://ui.adsabs.harvard.edu/search/q="+"&".join(concate[i*100:(i+1)*100]) #doi="+"10.3847/1538-4365/aaccfb"
	# # 	url += "&data_type=BIBTEX&nocookieset=1"	#&db_key=AST
	# # 	data += urllib.request.urlopen(url).read().decode()
	
	# bibcodes = []
	# for i in range(len(identifier)):
	# 	try:
	# 		papers = ads.SearchQuery(doi=identifier[i], fl=['bibcode'])
	# 		bibcode = [article.bibcode for article in papers]
	# 		if len(bibcode) == 1:
	# 			bibcodes.append(bibcode[0])
	# 		else:
	# 			print(identifier[i], bibcode)
	# 	except:
	# 		pass

	# print(bibcodes)

	# data = ads.ExportQuery(bibcodes=bibcodes, format="bibtex").execute()
	# f = open(outputfile, "w")
	# f.write(data)
	# f.close()

	f = open(newbibfile, "r", encoding="utf8")
	data = f.read()
	f.close()

	# # step 3: replace with the original citekey

	# find new dois and bibcodes from the data file
	# new_dois = re.compile("doi = {([a-zA-Z0-9/_.:-]{0,80})},\\n").findall(data)
	new_bibcodes = re.compile("ARTICLE{([a-zA-Z0-9/.\&]{0,80})").findall(data)
	# print(len(new_dois), len(new_bibcodes))

	print("Find "+str(len(new_bibcodes))+" out of "+str(len(citekey))+" entries in ads database.")

	index_citekey_accessed = []
	# itereate every entry in the new data and replace the citekey
	for i in range(len(new_bibcodes)):
		index_entry_1 = data.find("{"+new_bibcodes[i]+",")
		if i != len(new_bibcodes)-1:
			index_entry_2 = data.find(new_bibcodes[i+1])
		else:
			index_entry_2 = -1
		tdata = data[index_entry_1:index_entry_2]
		theader = new_bibcodes[i]
		# tdoi = new_dois[i]
		# print(tdata)
		# print(tdoi_list)
		replace_index1 = data.find("{"+theader+",") + 1 + theader.find("{")
		replace_index2 = data.find("{"+theader+",") + 3 + theader.find("{") + len(theader)
		tdoi_list = re.compile("doi = {([a-zA-Z0-9/_.:-]{0,80})},\\n").findall(tdata)
		# print(data[replace_index1:replace_index2])
		# print(tdoi, tdoi.lower())
		if len(tdoi_list) ==1:
			tdoi = tdoi_list[0]
			# print(tdoi.lower())
			idx = np.where(identifier == tdoi.lower())[0][0]
			index_citekey_accessed.append(idx)
			tcitekey = citekey[idx]
			print(data[replace_index1:replace_index2], tcitekey)
			data = data.replace(data[replace_index1:replace_index2], "{"+tcitekey+",")
		# else:
			# tbibcode = urllib.parse.quote(data[replace_index1:replace_index2])[:-3]
			# # print(data[replace_index1:replace_index2], tbibcode, identifier[1])
			# index = np.where(identifier == tbibcode)[0][0]
			# index_citekey_accessed.append(index)
			# tcitekey = citekey[index]
			# data = data.replace(data[replace_index1:replace_index2], tcitekey+",")


	
	# print those who didn't get an entry from ads.
	# index_citekey_accessed = np.sort(np.array(index_citekey_accessed))
	# index_citekey_naccessed = np.setdiff1d(np.arange(len(citekey)), index_citekey_accessed)
	# print(citekey[index_citekey_naccessed])
	# print(identifier[index_citekey_naccessed])

	# # little modification:
	# #2018bellinger-phd-inverse-problem
	# index = data.find("school = {Max Planck Institute for Solar System")
	# if index!=-1:
	# 	data = data.replace(data[index: index+109], "school = {Max Planck Institute for Solar System Research},")

	# #2018ting++100000-rc-lamost
	# index = data.find("A Large and Pristine Sample of Standard Candles across the Milky Way: ")
	# if index!=-1:
	# 	data = data.replace(data[index: index+122], "A Large and Pristine Sample of Standard Candles across the Milky Way: {$\sim$}100,000 Red Clump Stars with 3\% Contamination")

	# #1980gough-theoretical-remarks-on-stellar-oscillations
	# index = data.find("      doi = {10.1007/3-540-09994-8_27},")
	# if index!=-1:
	# 	data = data.replace(data[index: index+41], "")

	# #2015metcalfe++16-cyg-a-b
	# index = data.find("Asteroseismic Modeling of 16 Cyg A")
	# if index!=-1:
	# 	data = data.replace(data[index: index+78], "Asteroseismic Modeling of 16 Cyg A \& B using the Complete Kepler Data Set")

	# # step 4: output
	if ifadd:
		f = open(outputfile, "a", encoding="utf8")
	else:
		f = open(outputfile, "w", encoding="utf8")
	f.write(data)
	f.close()

	return 






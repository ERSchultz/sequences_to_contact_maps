#Module hic_to_txt
#Created by Aria Coraor

import numpy as np
from subtool import *
import strawC
import straw
import argparse

CHROMS = []

RESOLUTION = 10000
#Autologous chroms
CHROMS += [str(ele) for ele in range(1,23)]

#Sex chrom: X only
CHROMS += "X"

def main():
	""" Read a .hic file and output (non-stochastic) matrices of contacts to
	corresponding outputfiles: <stem>_i_j.txt for chromosomes i and j.
	"""
	print("Parsing Hi-C file...")
	if not args.diagonal:
		for i_c in CHROMS:
			for j_c in CHROMS:
				try:
					result = strawC.straw("NONE", 'NONE',args.file,i_c,j_c,"BP",RESOLUTION)
				except:
					print("Falling back to python")
					result = straw.straw("NONE", args.file,i_c,j_c,"BP",RESOLUTION)

				ofn = inb4(args.file,"_%s_%s.txt" % (i_c,j_c))

				#Result is a set of strings. Parse into outputfiles.
				rows = []
				for i in range(len(result)):
					rows.append("{0}\t{1}\t{2}".format(result[i].binX, result[i].binY, result[i].counts))
				print("Writing contacts for chromosomes: %s x %s" % (i_c,j_c))
				with open(ofn,'w') as f:
					for row in rows:
						f.write(row + "\n")
	else:
		print("Diagonal chromsome matrices selected.")
		for i_c in CHROMS:
			try:
				result = strawC.straw("NONE",  'NONE', args.file,i_c,i_c,"BP",RESOLUTION)
			except:
				print("Falling back to python")
				result = straw.straw("NONE", args.file,i_c,i_c,"BP",RESOLUTION)

			ofn = inb4(args.file,"_%s_%s.txt" % (i_c,i_c))

			#Result is a set of strings. Parse into outputfiles.
			rows = []
			for i in range(len(result)):
				rows.append("{0}\t{1}\t{2}".format(result[i].binX, result[i].binY, result[i].counts))
			print("Writing contacts for chromosomes: %s x %s" % (i_c,i_c))
			with open(ofn,'w') as f:
				for row in rows:
					f.write(row + "\n")

	print("Matrix files written successfully.")

	'''
	result = strawC.strawC('NONE', args.file, '1', '1', 'BP', 1000000)
	rows = []
	for i in range(len(result)):
		rows.append("{0}\t{1}\t{2}".format(result[i].binX, result[i].binY, result[i].counts))
	with open(inb4(args.file,".txt"),'w') as f:
		for row in rows:
			f.write(row + "\n")
	'''





if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f','--file', type=str, default=None,help='Input .hic file')
	parser.add_argument('-d','--diagonal',action="store_true", help = "If True, only save self_chrom-self_chrom contacts.")
	#parser.add_argument('-a','--all',default=False,action='store_const',const=True, help="Calculate helical parameters for all datafiles in NRL range. Output to angles, radii files.")
	args = parser.parse_args()
	main()

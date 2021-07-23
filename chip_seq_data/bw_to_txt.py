#Module bw_to_txt
#Created by Aria Coraor

import numpy as np
# from subtool import *
# import pyBigWig as pbw
import argparse
import hic_to_txt as hic
import os

def main():
	""" Take a BigWig file, and write its output to a matrix.

	"""
	fname = args.file
	if "." in fname:
		base = fname[:fname.rfind(".")]
	else:
		raise ValueError("No dot found in args.file: %s" % args.file)

	psub("mkdir %s" % base)

	#For every chromosome, get stats.

	bw = pbw.open(args.file)

	#Set resolution
	if not args.nucl:
		mode = "mean"
		res = args.res
	else:
		res = 200
		mode = "max"

	print("Loading chromosome data.")
	for i_c in hic.CHROMS:
		bw_name = "chr" + i_c
		length = bw.chroms()[bw_name]

		#Calculate nBins
		nBins = int(int(length) / res)
		#Truncate chrom at end:
		chip_vals = np.array(bw.stats(bw_name, 0, nBins*res,type = mode, nBins = nBins))
		chip_vals = np.where(chip_vals != None, chip_vals, 0.0)

		#chip_vals = np.array(bw.stats(bw_name, 0, nBins*res,type = mode, nBins = nBins))
		pos = np.arange(0,nBins*res,res)

		if chip_vals.shape != pos.shape:
			raise ValueError("Shapes not equivalent: %s vs. %s" % (repr(chip_vals.shape), repr(pos.shape)))
		zipped = np.column_stack((pos,chip_vals))
		#print("Zipped columns: %s" % repr(zipped))
		#print("Saving to ",os.path.join(base,i_c+".txt"))
		#Fix nans to zero
		#print("zipped: ",repr(zipped))
		#print("Last, last element: ",repr(zipped[-1][-1]))
		#print("Type of last, last element: ",repr(type(zipped[-1][-1])))

		#nans = np.argwhere(np.isnan(zipped[:,1])).flatten()
		#zipped[nans,1] = 0.

		np.savetxt(os.path.join(base, i_c + ".txt"),zipped)
		print("Saved data for chromosome %s." % i_c)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f','--file', type=str, default=None, help='Input .hic file')
	parser.add_argument('-n','--nucl',action="store_true", help = "If True, store nucleosome-resolution data.")
	parser.add_argument('-r','--res', type=int, default=hic.RESOLUTION,help='Resolution of map, in bp')
	#parser.add_argument('-a','--all',default=False,action='store_const',const=True, help="Calculate helical parameters for all datafiles in NRL range. Output to angles, radii files.")
	args = parser.parse_args()
	main()

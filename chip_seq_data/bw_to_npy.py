#Module bw_to_txt
#Created by Aria Coraor, modified by Eric Schultz

import os
import os.path as osp

import numpy as np
import pyBigWig as pbw
import argparse

from subtool import *
from aggregate_peaks import get_names
import hic_r_calc as hic


def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir', default=osp.join('chip_seq_data','fold_change_control'), help='directory of chip-seq data')
	parser.add_argument('-n','--nucl',action="store_true", help = "If True, store nucleosome-resolution data.")
	parser.add_argument('-r','--res', type=int, default=25000, help='Resolution of map, in bp')
	args = parser.parse_args()
	return args

def bw_to_npy(fname, args):
	'''Take a BigWig file, and write its output to a matrix.'''
	if "." in fname:
		base = fname[:fname.rfind(".")]
	else:
		raise ValueError("No dot found in fname: {}".format(fname))

	psub("mkdir %s" % base)

	#For every chromosome, get stats.

	bw = pbw.open(fname)

	#Set resolution
	if not args.nucl:
		mode = "mean"
		res = args.res
	else:
		res = 200
		mode = "max"

	print("Loading chromosome data for {}.".format(fname))
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
		zipped = np.column_stack((pos, chip_vals))

		np.save(os.path.join(base, i_c + ".npy"), zipped)
		print("Saved data for chromosome %s." % i_c)

def main():
	args = getArgs()

	files = [osp.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith('bigWig')]
	names = get_names(args.dir, files)
	print(names, len(names))
	for file in files:
		bw_to_npy(file, args)

if __name__ == '__main__':
	main()

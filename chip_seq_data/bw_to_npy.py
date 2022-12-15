#Module bw_to_txt
#Created by Aria Coraor, modified by Eric Schultz

import argparse
import os
import os.path as osp

import numpy as np
import pyBigWig as pbw
from utils import *


def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir', default=osp.join('chip_seq_data','HCT116', 'hg19', 'fold_change_control'),
						help='directory of chip-seq data')
	parser.add_argument('-n','--nucl',action="store_true",
						help="If True, store nucleosome-resolution data.")
	parser.add_argument('-r','--res', type=int, default=50000,
						help='Resolution of map, in bp')
	parser.add_argument('--mode', type=str, default='mean',
						help='Mode for bw.stats')
	args = parser.parse_args()
	return args

def bw_to_npy(fname, args):
	'''Take a BigWig file, and write its output to a matrix.'''
	if "." in fname:
		base = fname[:fname.rfind(".")]
	else:
		raise ValueError("No dot found in fname: {}".format(fname))

	if not osp.exists(base):
		os.mkdir(base, mode = 0o755)

	#For every chromosome, get stats.
	bw = pbw.open(fname)

	#Set resolution
	if not args.nucl:
		mode = args.mode
		res = args.res
	else:
		res = 200
		mode = "max"

	print("Loading chromosome data for {}.".format(fname))
	for chr in CHROMS:
		bw_name = "chr" + chr
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

		np.save(os.path.join(base, f"{chr}.npy"), zipped)
		print("Saved data for chromosome %s." % chr)

def main():
	args = getArgs()

	files = [osp.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith('bigWig')]
	names = get_names(args.dir, files)
	print(names, len(names))
	for file in files:
		bw_to_npy(file, args)

if __name__ == '__main__':
	main()

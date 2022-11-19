#Module hic_r_calc
#Created by Aria Coraor

import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import argparse
import copy
import csv
import itertools
import json
from time import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subtool import *
from utils import CHROMS, save_chip_for_CHROMHMM


def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d','--dir', type=str,
						default=osp.join('chip_seq_data', 'HCT116', 'hg19', 'fold_change_control'),
						help="Input Chip-Seq master directory")
	parser.add_argument('--res', type=int, default=50000, help='resolution')
	parser.add_argument('--cell_line', default='HCT116', help='cell line')

	args = parser.parse_args()
	return args

@timeit
def threshold_chip(chips, cfl_chips, names, args):
	"""Convert fold-over-control chipseq into
	binary yes/no vector of whether or not mark is present.
	Determines thresholds for every mark which ensures the entire genome has the
	total fraction as defined in args.dir/frac.json.
	Parameters:
		chips: *list of lists of 2d np.array*
				for all eles: first index is chromsome, second index is mark.
				For each np.array, [:,0] is the list of indices, [:,1] is the
				list of fold-over-control magnitudes. Every one is a chipseq
				track.
		cfl_chips: *2d np.array*
				First axis is mark, second axis is bp. cfl_chips[i,j,0] is
				bp, cfl_chips[i,j,1] is value.
	Returns:
		thresh_chip: *2d np.array*
				[:,0] is the list of indices, [:,1] is a binary 1,0
				for whether or not chip is present.
		threshes: *list of floats*
				The final thresholds used.
	Raises:
		ValueError, if thresh is not None.
	"""
	# get frac data
	with open(osp.join(osp.split(args.dir)[0], "frac.json"), 'r') as f:
		frac_dict = json.load(f)

	#Target fractions
	fracs = []
	for name in names:
		fracs.append(frac_dict[name])
	fracs = (np.array(fracs)).flatten()

	print("Input resolution: %s" % repr(args.res))

	# Raise error if cfl_chips resolution is different from args.res
	for i, mark in enumerate(cfl_chips):
		res = mark[1,0] - mark[0,0]
		if res != args.res:
			raise ValueError(f"Resolution of mark '{names[i]}', is {res} not {args.res}")

	threshes = []
	#frac-th percentile over the control.
	for i, mark in enumerate(cfl_chips):
		frac = fracs[i]
		thresh = np.quantile(cfl_chips[i][:,1].ravel(), 1-frac)
		threshes.append(thresh)
		print(f"Frac,thresh for mark {names[i]}: {frac}, {thresh}")
	threshes = np.array(threshes)

	print(f"Fraction-based thresholds: \n{threshes}\n for marks:\n {names[:-1]}\n")

	final_chips = copy.deepcopy(chips)
	#Lengths of each chromosome in bp indices
	lengths = [len(chrom[0]) for chrom in final_chips]

	for j, chrom in enumerate(final_chips):
		#Calculate indices from chromosome beginning
		start_length = int(np.sum(lengths[:j]))
		end_length = int(start_length + lengths[j])
		for i, mark in enumerate(chrom):

			#Get fold-over-control value
			foc = (cfl_chips[i])[start_length:end_length]

			# if FOC > threshold, set mark to 1
			mark[:,1] = np.where(foc[:,1] > threshes[i], np.ones(mark[:,1].shape),
					np.zeros(mark[:,1].shape))

			#If FOC is NAN, set mark to 0
			mark[:,1] = np.where(np.isfinite(foc[:,1]), mark[:,1],
					np.zeros(mark[:,1].shape))

	#Save binarization for each mark at filename "<chrom>_<name>_<coverage>_seq.npy"
	odir = osp.join(args.dir, 'processed')
	if not osp.exists(odir):
		os.mkdir(odir, mode = 0o755)

	for i, chrom in enumerate(final_chips):
		for j, mark in enumerate(chrom):
			# What proportion of each chromosome is covered by each mark?
			coverage = np.round(np.average(mark[:,1]), 5)
			print(f"{names[j]} coverage on chrom {CHROMS[i]}: {coverage}")
			ofile = f"chr{CHROMS[i]}_{names[j]}_{coverage}.npy"
			np.save(osp.join(odir, ofile), mark.astype(np.uint32)) # 32 bit int should be fine
		print('')

	return final_chips, threshes

@timeit
def load_chipseq(args):
	"""Load all the chipseq tracks in args.dir into numpy arrays.
	returns:
		chrom_chips: *3d list*
				first dimension is chromosome, second dimension is mark.
				chrom_chips[i][j] is a 2d np.array of the response for chromsome i
				and mark j. ele[:,0] is the bp, ele[:,1] is the chip value.
		names: list containing name of epigenetic mark
	"""
	# load metadata
	meta_data = pd.read_csv(osp.join(args.dir, 'metadata.tsv'), sep = '\t')

	# get marks
	marks = [osp.join(args.dir, ele) for ele in os.listdir(args.dir)]
	marks = [ele for ele in marks if osp.isdir(ele) and osp.split(ele)[1] in set(meta_data['File accession'])]

	#Find names
	names = []
	del_list = []
	for i, mark in enumerate(marks):
		try:
			mark = osp.split(mark)[1]
			df = meta_data[meta_data['File accession'] == mark]
			name = df['Experiment target'].item().split('-')[0]
			if name.startswith("H2"):
				print(f"H2 modifications not accepted: {mark}")
				del_list.append(i)
			elif name.startswith("H4K20"):
				print(f"H4K20 modifications not accepted: {mark}")
				del_list.append(i)
			else:
				names.append(name)
		except Exception as e:
			print(e)
			print(f"Had issues processing dirname {mark}")
			del_list.append(i)
	for ind in reversed(del_list):
		print("Popping item: ", marks[ind])
		marks.pop(ind)

	#Sort names immediately:
	inds = np.argsort(names)
	names = [names[i] for i in inds]
	marks = [marks[i] for i in inds]

	print(f"Names: {names}\n")
	#print("marks: %s" % repr(marks))


	chrom_chips = [] #Accumulator: N_chroms x N_chroms x N_marks

	loading_chroms = copy.copy(CHROMS)
	#For every mark, load each chrom

	for i, base in enumerate(loading_chroms):
		base_list = []
		print("Loading chrom ", base)
		for mark in marks:
			print("Loading mark ",mark)
			track = np.load(osp.join(mark, f"{base}.npy"), allow_pickle = True)
			base_list.append(track)
		chrom_chips.append(base_list)

	return chrom_chips, names

def main():
	'''Perform hic_r_calc. Need to run bw_to_np before hand.'''
	args = getArgs()

	#Load Chip vectors
	print("Loading Chip vectors.")
	chips, names = load_chipseq(args)
	print("Chip vectors loaded.")

	# Create chrom_flattened chips along bp
	chrom_flat_chips = [] # Cross over all chroms, break down only by mark
	for j_mark in range(len(chips[0])):
		l = [chrom[j_mark] for chrom in chips]
		l = np.concatenate(l)
		chrom_flat_chips.append(l)
	chrom_flat_chips = np.array(chrom_flat_chips).astype(np.float64)
	#chrom_flat_chips = np.concatenate(chrom_flat_chips)
	print(f"Chromosome flattened chip has shape: {chrom_flat_chips.shape}")


	#Calculate maxEnt thresholding
	print("Thresholding chip tracks.")
	final_chips, threshes = threshold_chip(chips, chrom_flat_chips, names, args)
	print("Chip tracks thresholded.")

	save_chip_for_CHROMHMM(final_chips, names, args)

if __name__ == '__main__':
	main()

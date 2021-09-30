#Module hic_r_calc
#Created by Aria Coraor

import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np
import argparse
import pandas as pd
import itertools
import copy
import json
from time import time
import csv

from subtool import *

#Autologous chroms
CHROMS = [str(ele) for ele in range(1,23)]
#Sex chrom: X only
CHROMS.append("X")

def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c','--chip', type=str, default=osp.join('chip_seq_data','fold_change_control'), help="Input Chip-Seq master directory")
	parser.add_argument('--res', type=int, default=25000, help='resolution')

	args = parser.parse_args()
	return args

def main():
	'''Perform hic_r_calc.'''
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
	print("Chromosome flattened chip has shape: %s" % repr(chrom_flat_chips.shape))


	#Calculate maxEnt thresholding
	print("Thresholding chip tracks.")
	final_chips, threshes = threshold_chip(chips, chrom_flat_chips, names, args)
	print("Chip tracks thresholded.")

	save_chip_for_CHROMHMM(final_chips, names, args)

def save_chip_for_CHROMHMM(chips, names, args):
	for i, chrom in enumerate(chips):
		ofile = osp.join(args.chip, 'processed', 'chr{}_binary.txt'.format(CHROMS[i]))
		combined_marks = np.zeros((len(chrom[0]), len(chrom)))
		for j, mark in enumerate(chrom):
			combined_marks[:, j] = mark[:, 1]
			with open(ofile, 'w', newline = '') as f:
				wr = csv.writer(f, delimiter = '\t')
				wr.writerow(["HTC116", "chr{}".format(CHROMS[i])])
				wr.writerow(names)
				wr.writerows(combined_marks)

@timeit
def threshold_chip(chips, cfl_chips, names, args):
	"""Convert fold-over-control chipseq into
	binary yes/no vector of whether or not mark is present.
	Determines thresholds for every mark which ensures the entire genome has the
	total fraction as defined in args.chip/frac.json.

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
	with open(osp.join(osp.split(args.chip)[0], "frac.json"), 'r') as f:
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
			raise ValueError("Resolution of mark '{}', is {} not {}".format(names[i], res, args.res))

	threshes = []
	#frac-th percentile over the control.
	for i, mark in enumerate(cfl_chips):
		frac = fracs[i]
		thresh = np.quantile(cfl_chips[i][:,1].ravel(), 1-frac)
		threshes.append(thresh)
		print("Frac,thresh for mark {}: {}, {}".format(names[i], repr(frac), repr(thresh)))
	threshes = np.array(threshes)

	print("Fraction-based thresholds: \n{}\n for marks:\n {}\n".format(repr(threshes), repr(names[:-1])))

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
	odir = osp.join(args.chip, 'processed')
	if not osp.exists(odir):
		os.mkdir(odir, mode = 0o755)

	for i, chrom in enumerate(final_chips):
		for j, mark in enumerate(chrom):
			# What proportion of each chromosome is covered by each mark?
			coverage = np.round(np.average(mark[:,1]), 5)
			print("{} coverage on chrom {}: {}".format(names[j],
				CHROMS[i], coverage))
			ofile = "{}_{}_{}_seq.npy".format(CHROMS[i], names[j], coverage)
			np.save(osp.join(odir, ofile), mark.astype(np.uint32)) # 32 bit int should be fine
		print('')

	return final_chips, threshes

@timeit
def load_chipseq(args):
	"""Load all the chipseq tracks in args.chip into numpy arrays.

	returns:
		chrom_chips: *3d list*
				first dimension is chromosome, second dimension is mark.
				chrom_chips[i][j] is a 2d np.array of the response for chromsome i
				and mark j. ele[:,0] is the bp, ele[:,1] is the chip value.
		names: list containing name of epigenetic mark
	"""
	# load metadata
	meta_data = pd.read_csv(osp.join(args.chip, 'metadata.tsv'), sep = '\t')

	# get marks
	marks = [osp.join(args.chip,ele) for ele in os.listdir(args.chip)]
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
				print("H2 modifications not accepted: {}".format(mark))
				del_list.append(i)
			elif name.startswith("H4K20"):
				print("H4K20 modifications not accepted: {}".format(mark))
				del_list.append(i)
			else:
				names.append(name)
		except Exception as e:
			print(e)
			print("Had issues processing dirname %s" % mark)
			del_list.append(i)
	for ind in reversed(del_list):
		print("Popping item: ", marks[ind])
		marks.pop(ind)

	#Sort names immediately:
	inds = np.argsort(names)
	names = [names[i] for i in inds]
	marks = [marks[i] for i in inds]

	print("Names: %s\n" % repr(names))
	#print("marks: %s" % repr(marks))


	chrom_chips = [] #Accumulator: N_chroms x N_chroms x N_marks

	loading_chroms = copy.copy(CHROMS)
	#For every mark, load each chrom

	for i, base in enumerate(loading_chroms):
		base_list = []
		print("Loading chrom ", base)
		for mark in marks:
			print("Loading mark ",mark)
			track = np.load(osp.join(mark, "{}.npy".format(base)), allow_pickle = True)
			base_list.append(track)
		chrom_chips.append(base_list)

	return chrom_chips, names

if __name__ == '__main__':
	main()

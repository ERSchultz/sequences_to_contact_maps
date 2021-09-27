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
from time import time
from joblib import Parallel, delayed
from sklearn import mixture as mix

from subtool import *

#CHIP = "/project2/depablo/coraor/chipseq"
#HIC = "/project2/depablo/coraor/hic/HSA"

RES = 10000
chroms = []
#Autologous chroms
chroms += [str(ele) for ele in range(1,23)]
#Sex chrom: X only
chroms += "X"

#INPUT_RES = 200
INPUT_RES = 10000 # Resolution of the input channel
#INPUT_RES = 20000

def main():
	""" Perform hic_r_calc
	"""
	global names
	#Load Chip vectors
	print("Loading Chip vectors.")
	chips, names = load_chipseq()
	print("Chip vectors loaded.")

	#Create chrom_flattened chips along bp
	chrom_flat_chips = [] # Cross over all chroms, break down only by mark
	for j_mark in range(len(chips[0])):
		l = [chrom[j_mark] for chrom in chips]
		l = np.concatenate(l)
		chrom_flat_chips.append(l)
	chrom_flat_chips = np.array(chrom_flat_chips)
	#chrom_flat_chips = np.concatenate(chrom_flat_chips)
	print("Chromosome flattened chip has shape: %s" % repr(chrom_flat_chips.shape))


	#Calculate maxEnt thresholding
	print("Thresholding chip tracks.")
	flat_chips, threshes = threshold_chip_ref(chips, chrom_flat_chips)
	print("Chip tracks thresholded.")


def threshold_chip_ref(chips, cfl_chips):
	"""Convert absolute magnitude fold-over-control chipseq into
	binary yes/no vector of whether or not mark is present.
	Determines thresholds for every mark which ensures the entire genome has the
	total fraction as defined in args.chip/<mark>_frac.dat.

	Average input over 10kbp has coverage about 90% of all chromosomes; presuming
	other regions are poorly sampled and discarding them.

	Pre-section the region of chromatin to be normalized by the average "input"
	mark over 10kbp. Work on fold-over this control to prevent singularities.

	Actual production quality.

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
	if args.debug:
		#Work only on chromosome 1
		chips = [chips[0]]

	#Target fractions
	fracs = []
	for name in names[:-1]:
		fracs.append(np.loadtxt(osp.join(args.chip,"{}_frac.dat".format(name))))
	fracs = (np.array(fracs)).flatten()


	#Raise error if cfl_chips resolution is different from INPUT_RES
	inp = cfl_chips[-1]
	inp_res = int(inp[1,0] - inp[0,0])
	print("Input resolution: %s" % repr(inp_res))
	if inp_res != INPUT_RES:
		raise ValueError(
			"Input track at %s must have resolution %s, instead has %s" % (
				args.chip, repr(INPUT_RES), repr(inp_res)))


	#INPUT_RES/res must be an integer. if not, raise ValueError
	for i, mark in enumerate(cfl_chips):
		res = mark[1,0] - mark[0,0]
		if INPUT_RES % res != 0:
			raise ValueError("Resolution of mark '%s', %s,  cannot evenly divide INPUT_RES" +
				" of %s" % (names[i],repr(res),repr(INPUT_RES)))

	#Calculate fold-over-control
	for i, mark in enumerate(cfl_chips[:-1]):
		res = mark[1,0] - mark[0,0]
		repeats = INPUT_RES // res

		#Create repeats of control array values
		rep_ctrl = np.repeat(inp[:,1],repeats)

		#Calculate fold over control
		if(len(rep_ctrl) < len(mark[:,1])):
			diff = np.array([0.0]*(len(mark[:,1])-len(rep_ctrl)))
			rep_ctrl = np.concatenate((rep_ctrl,diff))
		print("Shapes to broadcast:")
		print("Mark[:,1] shape: ",mark[:,1].shape)
		print("rep_ctrl shape: ",rep_ctrl.shape)
		if len(rep_ctrl) > len(mark[:,1]):
			rep_ctrl = rep_ctrl[:len(mark[:,1])]
		mark[:,1] = np.where(rep_ctrl > 0.0, mark[:,1]/rep_ctrl,np.zeros(mark[:,1].shape))

	threshes = []
	#frac-th percentile over the control.
	for i,mark in enumerate(cfl_chips[:-1]):
		frac = fracs[i]
		thresh = np.quantile(cfl_chips[i][:,1].ravel(),1-frac)
		threshes.append(thresh)
		print("Frac,thresh for mark %s: %s,%s" % (names[i],repr(frac),repr(thresh)))
	threshes = np.array(threshes)


	#For every mark, real mark name is between 3rd and fourth underscore
	if args.debug:
		print("Creating box and whisker plots...")
		plt.clf()
		mark_dict = dict()
		print("cfl_chips shape: %s" % repr(cfl_chips.shape))
		print("cfl first element shape: %s" % repr(cfl_chips[0].shape))
		for i,mark in enumerate([ele[:,1] for ele in cfl_chips[:-1]]):
			mark_dict[names[i]] = mark.ravel()

		fig,ax = plt.subplots()
		ax.boxplot(mark_dict.values())
		ax.set_xticklabels(mark_dict.keys())
		plt.yscale("log")
		plt.title("Chip-Seq Value Frequencies")
		plt.savefig(osp.join(args.chip,"chip_bwp.png"),dpi=600)


		#For each mark, plot a histogram
		for i, mark in enumerate([ele[:,1] for ele in cfl_chips[:-1]]):
			plt.clf()
			plt.hist(mark.ravel()[np.argwhere(mark.ravel() < 5)], bins = 100, edgecolor='black',linewidth=2)
			plt.xlabel(names[i] + " Value")
			plt.xlim(0,2)
			plt.ylabel("Number of occurences")
			plt.yscale("log")
			plt.title(names[i] + " Value Distribution")
			plt.savefig(osp.join(args.chip,names[i]+"_dist.png"),dpi=600)

		print("Plotting done")

	print("Fraction-based thresholds: %s\n for marks %s" % (repr(threshes),
		repr(names[:-1])))

	#Use the ctrl threshold to construct final chips:

	final_chips = copy.deepcopy(chips)
	#Lengths of each chromosome in bp indices
	lengths = [len(chrom[0]) for chrom in final_chips]
	#If > thresh, mark = 1, else 0

	for j,chrom in enumerate(final_chips):
		#Calculate indices from chromosome beginning
		start_length = int(np.sum(lengths[:j]))
		end_length = int(start_length + lengths[j])
		for i, mark in enumerate(chrom[:-1]):

			#Get fold-over-input value
			foi = (cfl_chips[i])[start_length:end_length]

			mark[:,1] = np.where(foi[:,1] > threshes[i],np.ones(mark[:,1].shape),
					np.zeros(mark[:,1].shape))
			#If FOC is NAN, set mark to 0
			mark[:,1] = np.where(np.isfinite(foi[:,1]),mark[:,1],
					np.zeros(mark[:,1].shape))

	#print("Thresholded chip: %s" % repr(final_chips))
	#What proportion of each is above the 95th percentile?
	for i,c in enumerate(final_chips):
		for j,n in enumerate(c[:-1]):
			print("%s coverage on chrom %s: %s" % (names[j],
				chroms[i],repr(np.average(n[:,1]))))

	#Save binarization for each mark at filename "<chrom>_<name>_thresh_<val>_seq.txt"
	for i,chrom in enumerate(final_chips):
		for j,mark in enumerate(chrom[:-1]):
			fn = "%s_%s_thresh_%s_seq.txt" % (chroms[i],names[j],repr(threshes[j]))
			np.savetxt(fn,mark, fmt = '%i')



	return final_chips, threshes


@timeit
def _max_ent(chips,sorteds,As):
	"""Calculate the total entropy over set S of a classification
	<threshold A.

	Parameters:
		chips: *list of np.array*
				Elements are raw chipseq vectors
		sorteds: *list of np.array*
				Results of np.argsort for each of chip.
		As: *lists floats*
				List of thresholds for each chip

	"""
	start = time()
	#Calculate the number in each class
	c = np.ones((len(chips[0]),len(chips)),dtype = int)
	for i, A in enumerate(As):
		#print("chips[i,:,1]: %s" % repr(chips[i,:,1]))
		#print("A: %s" % repr(A))
		#print("sorteds[i]: %s" % repr(sorteds[i]))
		ind = np.searchsorted(chips[i,:,1],A,sorter = sorteds[i])
		#For every index below the threshold, set c to 0
		c[(sorteds[i][:ind]).flatten(),i] = 0
	#Determine the number in each class
	lst = np.array(list(itertools.product([0,1],repeat=chips.shape[0])))

	#Counts for every class
	#print("Length of lst: %s" % repr(lst.shape))
	#print("Length of chips: %s" % repr(chips.shape))
	#print("Counts shape: %s" % repr(counts.shape))

	#print("Sorting bp chunks into classes:")
	#print("ls shape: %s" % repr(lst.shape))

	counts = np.zeros(2**chips.shape[0])
	#print("Counting alternative way.")
	start = time()
	for i, ls in enumerate(lst):
		#Use np to count number of occurences
		counts[i] += len(np.where((c == tuple(ls)).all(axis=1))[0])
	#print("Alt counting took %s seconds" % repr(round(time()-start,3)))


	#Now, calculate total class entropy
	probs = np.array(counts,dtype = float)/len(chips[0])

	#To remove log issue, mask all zero entries
	#print("Probs: %s" % repr(probs))
	#print("Probs shape: %s" % repr(probs.shape))
	probs = probs[np.argwhere(probs > 0)]
	#print("Masked probs: %s" % repr(probs))

	ents = -probs*np.log(probs)

	#Mask any infs: these have probability zero.
	#valid_inds = np.argwhere(np.isfinite(ents))

	total_ent = np.sum(ents)
	#print("Single max_ent evaluation took %s seconds." % repr(round(time()-start,3)))
	return total_ent

@timeit
def load_chipseq():
	"""Load all the chipseq tracks in args.chip into numpy arrays.

	returns:
		chrom_chips: *3d list*
				first dimension is chromosome, second dimension is mark.
				chrom_chips[i][j] is a 2d np.array of the response for chromsome i
				and mark j. ele[:,0] is the bp, ele[:,1] is the chip value.
		names: list containing name of epigenetic mark
	"""
	marks = [osp.join(args.chip,ele) for ele in os.listdir(args.chip)]
	marks = [ele for ele in marks if osp.isdir(ele)]

	# load metadata
	meta_data = pd.read_csv(osp.join(args.chip, 'metadata.tsv'), sep = '\t')

	#Find names
	names = []
	del_list = []
	for i, mark in enumerate(marks):
		try:
			mark = osp.split(mark)[1]
			df = meta_data[meta_data['File accession'] == mark]
			name = df['Experiment target'].item().split('-')[0]
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

	print("Names: %s" % repr(names))
	#print("marks: %s" % repr(marks))



	chrom_chips = [] #Accumulator: N_chroms x N_chroms x N_marks

	loading_chroms = copy.copy(chroms)
	#For every mark, load each chrom
	if args.debug:
		loading_chroms = ["2"]
		chroms[0] = "2" #TEMP!!


	for i,base in enumerate(loading_chroms):
		base_list = []
		print("Loading chrom ",base)
		for mark in marks:
			print("Loading mark ",mark)
			track = np.load(osp.join(mark,"%s.npy" % base))
			base_list.append(track)
		chrom_chips.append(base_list)

	return chrom_chips, names

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c','--chip', type=str, default=osp.join('chip_seq_data','bedFiles'), help="Input Chip-Seq master directory")
	parser.add_argument('-d','--debug',action="store_true", help="Only run on chrom 1, debug")
	#Note: Chip-seq master directory must contain as many marks as possible
	#To enable meaningful maxEnt thresholding

	args = parser.parse_args()
	main()

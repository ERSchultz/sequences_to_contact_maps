#Module hic_r_calc
#Created by Aria Coraor

import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import hic_to_txt as hic_m
import itertools
import copy
from subtool import *
from time import time
from joblib import Parallel, delayed
from sklearn import mixture as mix

#CHIP = "/project2/depablo/coraor/chipseq"
#HIC = "/project2/depablo/coraor/hic/HSA"
chroms = hic_m.CHROMS
ENT_REPS = 3 # 3 for debug, 99 for production
RES = hic_m.RESOLUTION
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
	flat_chips,threshes = threshold_chip_ref(chips, chrom_flat_chips)
	print("Chip tracks thresholded.")

	print("Loading hi-c matrix.")
	start = time()
	hics,diags, fnames = load_hic()

	print("Hi-c matrix loaded: %s" % repr(hics))



	print("Shape of (first) hic matrix: %s" % repr(hics[0].shape))
	print("Shape of all Chip vectors: %s" % repr([flat_chips[0][0].shape]))

	print("Shifting chip vector to find maximum fit with HiC eigenvectors...")
	shifted_chips = shift_chip(flat_chips,diags)

	#Calculate R_i(x,y) for all x,y
	print("Calculating Mole Ratios (R_i(x,y)).")
	R_ixy = calc_r(shifted_chips,hics)
	print("Mole Ratios calculated: ")

	#Plot results
	print("Plotting results.")

	print("Mole ratios plotted and saved at %s" % repr(args.hic))


def threshold_chip_ref(chips,cfl_chips):
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

	#chip_marks = [] # Cross over all chroms, break down only by mark
	#for j in range(len(chips[0])):
	#	l = [chrom[j] for chrom in chips]
	#	l = np.concatenate(l)
	#	chip_marks.append(l)

	#Target fractions
	fracs = []
	for name in names[:-1]:
		fracs.append(np.loadtxt(os.path.join(args.chip,"%s_frac.dat" % name)))
	fracs = (np.array(fracs)).flatten()

	#Now work on chip_marks exclusively
	#chip_marks = np.array(chip_marks)
	#print("Array-form chips: %s" % repr(chip_marks))
	#print("Array-form chips shape: %s" % repr(chip_marks.shape))
	#Chip_marks: mark x index x [bp, val]

	#Raise error if cfl_chips resolution is different from INPUT_RES

	inp = cfl_chips[-1]
	inp_res = int(inp[1,0] - inp[0,0])
	print("Input resolution: %s" % repr(inp_res))
	if inp_res != INPUT_RES:
		raise ValueError(
			"Input track at %s must have resolution %s, instead has %s" % (
				args.chip,repr(INPUT_RES),repr(inp_res)))


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
	"""
	mark_sorteds = [np.argsort(ele[:,1]) for ele in chip_marks]
	med_ent = _max_ent(chip_marks,mark_sorteds,meds)
	mean_ent = _max_ent(chip_marks,mark_sorteds,means)
	ctrl_ent = _max_ent(chip_marks,mark_sorteds,[ctrl_95]*len(means))
	#Compare to maximum theoretical entropy
	p_theo = 1./(2**(chip_marks.shape[0]))
	print("p_theo: %s" % repr(p_theo))
	max_theoretical_ent = (- np.log(p_theo))
	print("Entropy from median: %s" % repr(med_ent))
	print("Entropy from means: %s" % repr(mean_ent))
	print("Entropy from control: %s" % repr(ctrl_ent))

	print("Maximum theoretical entropy: %s" % repr(max_theoretical_ent))
	#quit()
	"""
	""" #Old style, true max ent
	min_maxes = []
	for mark in chip_marks[:,:,1]:
		min_maxes.append(np.quantile(mark.ravel(),[0.01,0.99]))
	trial_thresholds = np.array([np.linspace(ele[0],ele[1],ENT_REPS) for ele in min_maxes]		)

	mark_sorteds = [np.argsort(ele[:,1]) for ele in chip_marks]
	#entropies = _ents(chips,sorteds,As)
	#thresholds =  trial_thresholds[np.argmax(entropies)]
	print("Calculating entropy for all thresholds.")
	all_threshes = itertools.product(*trial_thresholds)
	listified = list(all_threshes)
	print("Shape of thresholds: %s" % repr(listified))
	print("Length: %s" % repr(len(listified)))
	quit()
	#Calculate all ents in parallel manner
	all_ents = np.array(Parallel(n_jobs=-1)(delayed(_max_ent)(
		chip_marks,mark_sorteds,ele) for ele in all_threshes))
	#all_ents = np.array([_max_ent(chip_marks,mark_sorteds,ele) for ele in all_threshes])
	final_threshes = all_threshes[np.argmax(all_threshes)]
	"""


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
		plt.savefig(os.path.join(args.chip,"chip_bwp.png"),dpi=600)


		#For each mark, plot a histogram
		for i, mark in enumerate([ele[:,1] for ele in cfl_chips[:-1]]):
			plt.clf()
			plt.hist(mark.ravel()[np.argwhere(mark.ravel() < 5)], bins = 100, edgecolor='black',linewidth=2)
			plt.xlabel(names[i] + " Value")
			plt.xlim(0,2)
			plt.ylabel("Number of occurences")
			plt.yscale("log")
			plt.title(names[i] + " Value Distribution")
			plt.savefig(os.path.join(args.chip,names[i]+"_dist.png"),dpi=600)

		print("Plotting done")

	#print("Calculating gaussian mixture models")
	#clf = mix.GaussianMixture(n_components=2, covariance_type='full')
	#clf.fit(X_train)

	#print("MaxEnt Thresholds: %s" % repr(final_threshes))

	#print("Median Thresholds: %s" % repr(meds))
	#print("Mean thresholds: %s" % repr(means))
	#print("Control threshold: %s" % repr(ctrl_95))
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


def threshold_chip(chips, thresh=None):
	"""DEPRECATED
	Convert absolute magnitude fold-over-control chipseq into
	binary yes/no vector of whether or not mark is present.
	Determine a threshold based on the maximum entropy principle for multiple marks.
	That is, for N marks, there are 2^N possible classes for any region. Find
	thresholds for every mark which maximizes the classification entropy
	over all 2^N possible classes.

	Parameters:
		chips: *list of lists of 2d np.array*
				for all eles: first index is chromsome, second index is mark.
				For each np.array, [:,0] is the list of indices, [:,1] is the
				list of fold-over-control magnitudes. Every one is a chipseq
				track.
		thresh: *float or None*
				If float, use this value as the threshold for all marks. If None,
				determine threshold by the maximum entropy principle.

	Returns:
		thresh_chip: *2d np.array*
				[:,0] is the list of indices, [:,1] is a binary 1,0
				for whether or not chip is present.
		threshes: *list of floats*
				The final thresholds used.

	Raises:
		ValueError, if thresh is not None.
	"""
	##DEPRECATED
	if args.debug:
		chips = [chips[0]]
	#print("Chips: %s" % repr(chips))
	#print("Length of chips: %s" % repr(len(chips)))
	#print("Chips first element: %s" % repr(chips[0]))
	chip_marks = [] # Cross over all chroms, break down only by mark
	for j in range(len(chips[0])):
		l = [chrom[j] for chrom in chips]
		l = np.concatenate(l)
		chip_marks.append(l)

	#Now work on chip_marks exclusively
	chip_marks = np.array(chip_marks)
	#print("Array-form chips: %s" % repr(chip_marks))
	#print("Array-form chips shape: %s" % repr(chip_marks.shape))
	#Chip_marks: mark x index x [bp, val]

	if thresh is not None:
		raise ValueError("Manual threshold not implemented.")
		#ones = np.argwhere(chip_marks[:,:,1] > thresh)
		#binned = np.zeros(chips.shape[1])
		#for ind in ones:
		#	binned[ones] = 1
	else:
		#Variation on Chen et al 2011, "A varying threshold..."

		#Determine all possible thresholds for all N values.
		#Use: 99-point-tile between mark N's 1st percentile and N's 99th percentile.
		#Evaluate entropy 99*N^2 different times, select best.
		#print("chips: %s" % repr(chips))


		#chip_vals = [ele[]]
		#sorted_vals = [np.sort(ele[:,1]) for ele in chip_marks] #Sorted by mark
		#sorted_val = np.sort(chips[:,1])

		#Calculate the geometric median of the dataset
		#For all, mask the zeros
		meds = []
		means = []
		for mark in chip_marks[:,:,1]:
			meds.append(np.median(mark.ravel()))
			means.append(np.mean(mark.ravel()))

		#95th percentile of the control. Corresponds to 0.05 FDR.
		ctrl_95 = np.quantile(chip_marks[-1,:,1].ravel(),0.95)

		mark_sorteds = [np.argsort(ele[:,1]) for ele in chip_marks]
		med_ent = _max_ent(chip_marks,mark_sorteds,meds)
		mean_ent = _max_ent(chip_marks,mark_sorteds,means)
		ctrl_ent = _max_ent(chip_marks,mark_sorteds,[ctrl_95]*len(means))
		#Compare to maximum theoretical entropy
		p_theo = 1./(2**(chip_marks.shape[0]))
		print("p_theo: %s" % repr(p_theo))
		max_theoretical_ent = (- np.log(p_theo))
		print("Entropy from median: %s" % repr(med_ent))
		print("Entropy from means: %s" % repr(mean_ent))
		print("Entropy from control: %s" % repr(ctrl_ent))

		print("Maximum theoretical entropy: %s" % repr(max_theoretical_ent))
		#quit()

		""" #Old style, true max ent
		min_maxes = []
		for mark in chip_marks[:,:,1]:
			min_maxes.append(np.quantile(mark.ravel(),[0.01,0.99]))
		trial_thresholds = np.array([np.linspace(ele[0],ele[1],ENT_REPS) for ele in min_maxes]		)

		mark_sorteds = [np.argsort(ele[:,1]) for ele in chip_marks]
		#entropies = _ents(chips,sorteds,As)
		#thresholds =  trial_thresholds[np.argmax(entropies)]
		print("Calculating entropy for all thresholds.")
		all_threshes = itertools.product(*trial_thresholds)
		listified = list(all_threshes)
		print("Shape of thresholds: %s" % repr(listified))
		print("Length: %s" % repr(len(listified)))
		quit()
		#Calculate all ents in parallel manner
		all_ents = np.array(Parallel(n_jobs=-1)(delayed(_max_ent)(
			chip_marks,mark_sorteds,ele) for ele in all_threshes))
		#all_ents = np.array([_max_ent(chip_marks,mark_sorteds,ele) for ele in all_threshes])
		final_threshes = all_threshes[np.argmax(all_threshes)]
		"""


	#For every mark, real mark name is between 3rd and fourth underscore
	if args.debug:
		print("Creating box and whisker plots...")
		plt.clf()
		mark_dict = dict()
		for i,mark in enumerate(chip_marks[:,:,1]):
			mark_dict[names[i]] = mark.ravel()

		fig,ax = plt.subplots()
		ax.boxplot(mark_dict.values())
		ax.set_xticklabels(mark_dict.keys())
		plt.yscale("log")
		plt.title("Chip-Seq Value Frequencies")
		plt.savefig(os.path.join(args.chip,"chip_bwp.png"),dpi=600)


		#For each mark, plot a histogram
		for i, mark in enumerate(chip_marks[:,:,1]):
			plt.clf()
			plt.hist(mark.ravel()[np.argwhere(mark.ravel() < 5)], bins = 100, edgecolor='black',linewidth=2)
			plt.xlabel(names[i] + " Value")
			plt.xlim(0,2)
			plt.ylabel("Number of occurences")
			plt.yscale("log")
			plt.title(names[i] + " Value Distribution")
			plt.savefig(os.path.join(args.chip,names[i]+"_dist.png"),dpi=600)

		print("Plotting done")

	#print("Calculating gaussian mixture models")
	#clf = mix.GaussianMixture(n_components=2, covariance_type='full')
	#clf.fit(X_train)

	#print("MaxEnt Thresholds: %s" % repr(final_threshes))

	#print("Median Thresholds: %s" % repr(meds))
	#print("Mean thresholds: %s" % repr(means))
	print("Control threshold: %s" % repr(ctrl_95))

	#Use the ctrl threshold to construct final chips:

	final_chips = copy.deepcopy(chips)

	#If > thresh, mark = 1, else 0

	for chrom in final_chips:
		for mark in chrom:
			mark[:,1] = np.where(mark[:,1] > ctrl_95,np.ones(mark[:,1].shape),
					np.zeros(mark[:,1].shape))

	#print("Thresholded chip: %s" % repr(final_chips))
	#What proportion of each is above the 95th percentile?
	for i,c in enumerate(final_chips):
		for j,n in enumerate(c):
			print("%s coverage on chrom %s: %s" % (names[j],
				chroms[i],repr(np.average(n[:,1]))))

	#Save binarization for each mark at filename "<chrom>_<name>_thresh_<val>_seq.txt"
	for i,chrom in enumerate(final_chips):
		for j,mark in enumerate(chrom):
			fn = "%s_%s_thresh_%s_seq.txt" % (chroms[i],names[j],repr(ctrl_95))
			np.savetxt(fn,mark)



	return final_chips, ctrl_95

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
def shift_chip(flat_chips,diags):
	"""For uneven but complementary chipseq tracks and Hi-C: Find linear shift of
	Chip-Seq which best matches Hi-C Eigenvectors, and truncate length of ChipSeq
	with this shift to as accurately match the data as possible.

	Parameters:
			flat_chips: *list of lists of np.array*
					Output from threshold_chip()
			diags: *list of np.matrix*
					All diagonal / self-chrom-selfchrom hic matrices.

	Returns:
			shifted_chips: *list of np.array*
					Chipseq-vectors, shifted to maximally match the Hi-C data.

	"""
	all_shifteds = []
	for i in range(len(flat_chips)):
		all_shifteds.append(_shift_chrom(flat_chips[i],diags[i]))

	return all_shifteds

def _shift_chrom(chips, hic):

	"""Calculate the optimally shifted chips for a single chromosome."""
	diff = -(hic.shape[0] - chips[0].shape[0])

	print("Truncating chips instead.")
	return _trunc_chrom(chips,hic)

	print("Length difference for chromosomes: %s" % repr(diff))


	#Calculate eigendecomposition of every matrix
	#eigs = []
	#for m in diags:
	#	eigs.append(np.linalg.eig(m))

	#Calculate the "angle" in the direction of the matrix magnitude.
	#Maximize \sum_i ||Axi|| for shifts in i

	#calculate the translated array for all possible translations prior to
	#truncation
	chips = np.array(chips)
	#chips is now 2d array: first index is mark, second index is
	displ_arrays = np.array([chips[:,i:-diff+i,1] for i in range(diff)])

	#displ_arrays: first index is displacement, second index is mark, third
	#index is bp

	sum_norm = [] #List of sum of chipseq norms for each displ

	#Calculate A*xi for each displ_array
	for i, v in enumerate(displ_arrays):
		axis = [np.matmul(hic,chip) for chip in v]
		norm = np.sum([np.linalg.norm(ax) for ax in axis])
		sum_norm.append(norm)

	#Highest norm == best shift
	sum_norm = np.array(sum_norm)
	pct_norm = 100.*(sum_norm - np.average(sum_norm))/np.average(sum_norm)
	print("Sum norm (percentages): %s" % repr(pct_norm))
	best_displ = np.argmax(sum_norm).flatten()
	print("Best alignment found with a shift of %s" % repr(best_displ))


	#Use best alignment to construct new chipseq
	return chips[:,best_displ[0]:-diff + best_displ[0],:]
	'''
	for i, chrom in enumerate(flat_chips):
		diff_array = []
		for j, mark in enumerate(chrom):
			#Calculate
			pass




	pass
	'''

def _trunc_chrom(chips,hic):
	"""Rescale chips and then perform right-truncation or zero-addition to match
	HIC size."""
	chip_res = 200
	hic_res = hic_m.RESOLUTION
	factor = hic_res//chip_res

	print("Chips type: ",type(chips))
	if type(chips) == np.array:
		print("Chips shape: ",chips.shape)

	hic_len = hic.shape[0]
	#Have to reduce by factor.
	new_chips = []

	for chip in chips:
		print("Chip shape: ",chip.shape)
		print("Hic shape: ",hic.shape)
		print("Factor: ",factor)

		print("Chip / factor: ", chip.shape[0]/factor)

		actual_data = chip[:,1]
		run_aves = [np.average(actual_data[i*factor:(i+1)*factor]) for i in range(chip.shape[0]//factor)]
		run_aves = np.array(run_aves)
		if len(run_aves) < hic_len:
			run_aves = np.concatenate((run_aves,np.zeros(hic_len - len(run_aves))))
		inds = np.arange(0,hic_res*hic_len,hic_res)
		new_data = np.column_stack((inds,run_aves))
		print("New data shape: ",new_data.shape)
		new_chips.append(new_data)
	return new_chips




def calc_r(chips, diags,hic_fn = None, c_names = None):
	"""Calculate the R ratios of chipseq tracks over the given
	hic contact matrix.

	Parameters:
		chips: *list of lists of 2d np.array*
			Output of shift_chip(). First index is chromosome, second
			index is mark. All vectors have identical length to the corresponding
			diagonal hic matrix, and is shifted for maximum norm.
		diags: *np.matrix*
			List of chromosome-diagonal hic matrices
	"""
	if hic_fn is None:
		hic_fn = args.hic

	if c_names is None:
		c_names = names

	#Calculate all Axi
	for i in range(len(chroms)):
		print("Chip vector: %s" % chips[i][0])
		axis = [np.matmul(diags[i],chips[i][j][:,1]) for j in range(len(chips[0]))]

		R_matrix = np.zeros((len(chips[0]),len(chips[0])))
		for mark_i in range(len(chips[0])):
			#Calculate denoms
			denom = np.dot(axis[mark_i],chips[i][mark_i][:,1])
			if denom == 0:
				print("Error! No self-self contacts detected in Hi-C map!")
				print("Correcting to 1/10th of the minimum off-diag value:")

				nums = np.array([np.dot(axis[mark_i],chips[i][mark_j][:,1]) for mark_j in
						range(len(chips[0]))])
				zeros = np.argwhere(nums == 0.0)
				mask = [ind for ind in range(len(nums)) if ind not in zeros]


				denom = 1./10. * np.min(nums[mask])


			for mark_j in range(len(chips[0])):
				num = np.dot(axis[mark_i],chips[i][mark_j][:,1])
				R_matrix[mark_i,mark_j] = num/denom
				print("For mark contact %s vs %s:" % (repr(mark_i),repr(mark_j)))
				print("num: %s" % repr(num))
				print("denom: %s" % repr(denom))
				print("")
			#denoms = [np.dot(axis[j],chips[i][j]) for j in range(len(chips[0]))]
		print("R Matrix: %s" % repr(R_matrix))
		#Plot contact matrix for chromosome i
		plt.clf()
		color = cm.viridis
		sm = cm.ScalarMappable(matplotlib.colors.Normalize(0,2),color)
		plt.matshow(R_matrix,norm=sm.norm)
		plt.colorbar(sm)
		fname = os.path.join(hic_fn,"R_matrix_%s_%s.png" % (repr(i+1),repr(i+1)))
		plt.savefig(fname,dpi=600)
		print("Saved hic R matrix plot at %s" % fname)
		#Save actual matrix
		fname = os.path.join(hic_fn,"R_mat_vals_%s_%s.txt" % (repr(i+1),repr(i+1)))
		np.savetxt(fname,R_matrix)
		print("Saved hic R matrix values at %s" % fname)

	#Save names
	with open("mark_names.txt",'w') as f:
		for name in c_names:
			f.write(name+"\n")

	#Calculate R for each chromosome
	return R_matrix


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
	marks = [os.path.join(args.chip,ele) for ele in os.listdir(args.chip)]
	marks = [ele for ele in marks if os.path.isdir(ele)]

	#Find names
	names = []
	del_list = []
	for i, mark in enumerate(marks):
		m = mark[mark.rfind("/")+1:]
		if "Ctrl" in mark:
			names.append(m[18:m.find("_",18)])
		elif "Heat_Shock" in mark:
			names.append(m[24:m.find("_",24)])
		elif len(m) <= 8:
			names.append(m)
		else:
			print("Had issues processing dirname %s" % mark)
			del_list.append(i)
			#raise ValueError("Cannot process dirname %s" % mark)
	for ind in reversed(del_list):
		print("Popping item: ",marks[ind])
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
			track = np.loadtxt(os.path.join(mark,"%s.txt" % base))
			base_list.append(track)
		chrom_chips.append(base_list)

	return chrom_chips, names

@timeit
def load_hic():
	"""Load the sparse hic matrices located at args.hic into numpy matrices.

	Returns:
		matrices: *list of np.matrix*
				A flat list of matrices. Contains the series of contacts over
				chroms i and j.
		diags: *list of np.matrix*
				A flat list of only the diagonal matrices, i.e. CA_i_i.txt
		fnames: *list of str*
				The filenames loaded into each matrix.
	"""

	load_chroms = copy.copy(chroms)
	#For every mark, load each chrom
	if args.debug:
		load_chroms = ["1"]
	dats = []

	#Calculate base name
	base = os.path.abspath(args.hic)
	base = base[base.rfind("/")+1:]
	fnames = []
	diag_dats = []
	for i,i_c in enumerate(load_chroms):
		for j,j_c in enumerate(load_chroms[i:]):
			fname = "%s_%s_%s.txt" % (base,i_c,j_c)
			dats.append(np.loadtxt(os.path.join(args.hic,fname)))
			fnames.append(fname)
			if i_c == j_c:
				diag_dats.append(len(dats)-1)

	#Determine dimensions of each hic matrix
	hic_shapes = []
	for i,arr in enumerate(dats):
		shape = [int(arr[:,0].max()),int(arr[:,1].max())]
		if i in diag_dats:
			val = max(shape)
			shape = [val,val]
		#Division by RES should be exact
		remainders = (shape[0] % RES) + (shape[1] % RES)
		if remainders != 0:
			raise ValueError("Error: inexactness in Hi-C matrix dimensions %s"
					% repr(shape))

		shape = [shape[0]/RES, shape[1]/RES]
		hic_shapes.append(np.array(shape,dtype = int))
	hic_shapes = np.array(hic_shapes,dtype = int)


	print("Shape of all hic matrices: %s" % repr(hic_shapes))
	print("dtype of hic_shapes: %s" % repr(hic_shapes.dtype))

	#Create empty 2d arrays for each matrix.
	matrices = [np.zeros(shape) for shape in hic_shapes]
	#print("First matrix: %s" % repr(matrices[0]))
	#print("First matrix shape: %s" % repr(matrices[0].shape))

	for i,d in enumerate(dats):
		rows = np.array((d[:,0] - RES)/RES,dtype = int)
		cols = np.array((d[:,1] - RES)/RES,dtype = int)
		zipped = np.column_stack((rows,cols))
		#Calculate flattened indices for np.put
		flat_inds = (matrices[i].shape[1] * rows + cols).flatten()
		#print("Setting all matrix values at once.")
		#print("Zipped indices: %s" % repr(zipped))
		#print("Zipped shape: %s" % repr(zipped.shape))
		#print("Values: %s" % repr(d[:,2]))
		#print("Values shape: %s" % repr(d[:,2].shape))
		np.put(matrices[i],flat_inds,d[:,2].ravel())
		#matrices[i][zipped] = d[:,2]

	'''
	for row in dat1:
		i = int(round((row[0]-RES)/RES))
		j = int(round((row[1]-RES)/RES))
		#print("i, j: %s, %s" % (repr(i),repr(j)))
		try:
			matrix[i,j] = row[2]
			matrix[j,i] = row[2]
		except:
			pass
			#print("Error: %s or %s not within matrix bounds." % (repr(i),repr(j)))
	'''

	#Convert to real matrices
	matrices = [np.matrix(m) for m in matrices]
	#Diagonalize all matrices
	for i in range(len(matrices)):
		#Add the transpose of the matrix and subtract the diagonal
		matrices[i] = matrices[i] + matrices[i].T - np.diag(np.diag(matrices[i]))

	diags = [matrices[i] for i in diag_dats]
	return matrices,diags, fnames

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--hic', type=str, help='Input hic directory (txt files)')
	parser.add_argument('-c','--chip', type=str, help="Input Chip-Seq master directory")
	parser.add_argument('-d','--debug',action="store_true", help="Only run on chrom 1, debug")
	#Note: Chip-seq master directory must contain as many marks as possible
	#To enable meaningful maxEnt thresholding

	#parser.add_argument('-a','--all',default=False,action='store_const',const=True, help="Calculate helical parameters for all datafiles in NRL range. Output to angles, radii files.")
	args = parser.parse_args()
	main()

from subprocess import Popen, PIPE, STDOUT
from time import time

def psub(cmd):
	"""Submit cmd to the shell."""
	return Popen(cmd,shell=True,stdout=PIPE,stderr=STDOUT).stdout.read().decode('utf-8')

def inb4(fname,s):
	"""Inserts s before the ".", if it exists, and excises what currently is
	before the dot. Otherwise, inserts s at the end.
	Returns a relative path.
	"""
	if "." in fname:
		base = fname[:fname.rfind(".")]
	else:
		base = fname
	return base + s

def timeit(f):

	def timed(*args, **kw):

		ts = time()
		result = f(*args, **kw)
		te = time()
		if te-ts > 0.0001:
			print(('func:%r  took: %2.4f sec\n' % \
			  (f.__name__, te-ts)))
		else:
			print(('func:%r  took: %2.4f ms\n' % \
			  (f.__name__, (te-ts)*1000)))
		return result

	return timed

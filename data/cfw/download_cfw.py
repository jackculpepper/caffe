"""
download_cfw.py
Download the cfw dataset from the URLs listed in info.txt of each celeb
"""

import urllib2
import os
import random
from itertools import islice
import time
import socket
import httplib

from IPython.parallel import Client
rc = Client()
dview = rc[:]

CFW_DIR = './thumbnails_features_deduped_publish/'
CFW_DST_DIR = './cfw_full/'

TIMEOUT = 0.25
socket.setdefaulttimeout(TIMEOUT)

def celeb_download(celeb):
	import urllib2
	import os
	import random
	from itertools import islice
	import time
	import socket
	import httplib

	CFW_DIR = './thumbnails_features_deduped_publish/'
	CFW_DST_DIR = './cfw_full/'
	
	TIMEOUT = 0.25
	socket.setdefaulttimeout(TIMEOUT)

	if not os.path.exists(CFW_DST_DIR + celeb):
		os.makedirs(CFW_DST_DIR + celeb)
	
	info = open(CFW_DIR + celeb + '/info.txt', 'r')
	
	i = 0
	num = int(info.readline().split('\t')[0])
	print "Celeb: " + celeb
	while True:
		images = list(islice(info, num+1))
		print num
		# If there is another batch of images to process
		if len(images) == num+1:
			num = int(images[-1].split('\t')[0])
			images = images[:-1]
		elif len(images) == num:
			pass
			#we've reached the last batch of images
		elif not images:
			break
	
		#if not os.path.exists(CFW_DST_DIR + celeb + '/' + str(i)):
			#os.makedirs(CFW_DST_DIR + celeb + '/' + str(i))

		for image in images:
			try:
				print "Getting " + image
				req = urllib2.Request(image)
				response = urllib2.urlopen(req, timeout=TIMEOUT).read()
				img_dir = CFW_DST_DIR + celeb + '/' + str(i) + \
					os.path.splitext(image)[1][:4]
				f = open(img_dir, 'wb')
				f.write(response)
				f.close()
			except  (urllib2.URLError, IOError, httplib.HTTPException):
				pass
			except  socket.timeout:
				print "Timed out"
				pass
			except  socket.error:
				print "Socket error"
				pass
			else:
				if os.stat(img_dir).st_size > 5120:
					break
				else:
					print "Too small"
					os.remove(img_dir)
					pass
			
		i += 1
CELEB_LIST = os.listdir(CFW_DIR)[192:1360] # Blake Lively to Sarah Mclachlan [incliuded]
#CELEB_LIST.sort()
dview.map_sync(celeb_download, CELEB_LIST)


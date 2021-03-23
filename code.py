#!/usr/bin/python3

from sys import argv
file_name = str(argv[1])
fh = open(file_name)

for line in fh:
	line = line.strip()
	# print(line)
	if line.startswith('the file is:'):
		lin = line.split('=>')
		img = lin[1].strip()
		camera = lin[0].split(':')[1].strip()

		print(camera)
		print(img)

	elif line.startswith('The number of objects is/are ==>'):
		lin = line.split('==>')
		minLabels = lin[1].strip()

		print(minLabels)

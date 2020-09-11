import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np


graph_csv =  glob.glob('results_1000/*.csv')
for items in graph_csv:
	input_file = open(items,'r')
	readed = input_file.read().split('\n')
	x = [] 
	y = []
	for elemnt in readed:
		if(elemnt):
			xx,yy = elemnt.split()
			x.append(float(xx))
			y.append(float(yy))
	plt.plot(x,y)
	print (x)


plt.legend(graph_csv, loc='upper left')
plt.ylabel('Loss')
plt.xlabel('Time (seconds)')
plt.title('Comparision_1000_Iteration')
plt.savefig('GraphA.png')
plt.show()
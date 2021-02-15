from django.shortcuts import render
import matplotlib.pyplot as plt 
import io 
import urllib, base64
from Graph.Qlearning import Deltas

def Graph(request):
	L = Deltas()
	plt.xlabel("iterations")
	
	plt.ylabel("Q-value at each iteration")
	plt.plot(L)
	fig = plt.gcf()
	buf = io.BytesIO()
	fig.savefig(buf,format = 'png')
	buf.seek(0)
	string = base64.b64encode(buf.read())
	uri = urllib.parse.quote(string)
	return render(request,'Graph/graph.html',{'data':uri})



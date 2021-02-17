from django.shortcuts import render
import matplotlib.pyplot as plt 
import io 
import urllib, base64
from Graph.Qlearning import Deltas

def Graph(request):
	deltas, policy, update_counts, V = Deltas()
	plt.xlabel("iterations")
	
	plt.ylabel("Q-value at each iteration")
	plt.plot(deltas)
	fig = plt.gcf()
	buf = io.BytesIO()
	fig.savefig(buf,format = 'png')
	buf.seek(0)
	string = base64.b64encode(buf.read())
	uri = urllib.parse.quote(string)
	pn = len(policy)
	un = len(update_counts)
	vn = len(V)

	return render(request,'Graph/graph.html',{'data':uri, 
		'policy':policy,
		'pn': range(pn),
		'update_counts': update_counts,
		'un': range(un), 
		'V':V,
		'vn':range(vn)
		})



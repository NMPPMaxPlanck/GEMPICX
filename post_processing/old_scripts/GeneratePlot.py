import matplotlib
import matplotlib.pylab as plt

import numpy as np
import math

########### sample data ####################

ref_filename = "/home/irga/Documents/Projects/gempic/simulations/PIC/reference_results/Weibel_01.txt"

n_lines = sum(1 for line in open(ref_filename))

t = np.empty(n_lines-1)
B3 = np.empty(n_lines-1)

file = open(ref_filename, 'r')

line_num = 0
for line in file.readlines():
    if line_num > 0 :
        vals = line.rstrip().split(' ') #using rstrip to remove the \n
        t[line_num-1] = vals[0]
        B3[line_num-1] = vals[6]
    line_num = line_num + 1


########### plotting ####################

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 13}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(t,B3,t,7e-8*np.exp(0.02784*2*t), linewidth=0.5)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\frac{1}{2} || B_z ||$')
axes = plt.gca()
axes.set_xlim([0,500])
axes.set_ylim([1e-9,1e-3])
plt.tight_layout()

########### save data to csv to plot in tikz ####################
# we save a maximum of 10,000 steps so that tikz does not crash
max_steps = 1000.0
ratio = math.ceil(float(n_lines)/max_steps)

import csv
f = open('Documents/Projects/gempic_amrex/notes/amrex_documentation/amrex_course/PlotData/Weibel.csv', 'w')

with f:

    writer = csv.writer(f)
    writer.writerow(["t", "B3"])
    
    file = open(ref_filename, 'r')

    line_num = 0
    for line in file.readlines():
        if line_num > 0:
            if (line_num % ratio) == 0:
                vals = line.rstrip().split(' ') #using rstrip to remove the \n
                writer.writerow([float(vals[0]), float(vals[6])])
        line_num = line_num + 1
    
        
f.close()

########### tikz code for plotting ####################
"""
\usepackage{pgfplots}
\pgfplotsset{compat=newest}

\begin{tikzpicture}
\begin{axis}[
	ymode=log,
	xlabel=$t$,
	ylabel=$\frac{1}{2} ||B_z||$,
	title={Norm of $B_z$ field over time (Weibel)},
	grid=both,
	minor grid style={gray!25},
	major grid style={gray!25},
	width=0.75\linewidth,
	no marks]
\addplot[line width=1pt,solid,color=blue] %
	table[x=t,y=B3,col sep=comma]{PlotData/Weibel.csv};
\addlegendentry{Norm};
\end{axis}
\end{tikzpicture}
"""

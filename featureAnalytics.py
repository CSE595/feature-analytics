from matplotlib import pyplot as plt
#from ui import generateUI
import os, sys
import pandas as pd
import numpy as np
import Tkinter as Tk
from Tkinter import *
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from PIL import ImageTk, Image

FEATUREOPTIONS = [
    "Area",
    "Perimeter",
    "Roundness",
    "Convex Area",
    "Solidity",
    "Major Axis",
    "Minor Axis",
    "Orientation",
    "Eccentricity",
    "Shape Index"
]

TENCOLORS = [
	'#191970',
	'#adff2f',
	'#daa520',
	'#cd5c5c',
	'#ffa500',
	'#8a2be2',
	'#00ffff',
	'#228b22',
	'#eedd82',
	'#b22222'
]

TENCOLORSHEX = [
	(25,25,112),
	(173,255,47),
	(218,165,32),
	(205,92,92),
	(255,165,0),
	(138,43,226),
	(0,255,255),
	(34,139,34),
	(238,221,130),
	(178,34,34)
]

CLUSTEROPTIONS = [
    "K-Means",
    "DBSCAN",
]

CLUSTERCOUNTOPTIONS = [
	"1",
	"2",
	"3",
	"4",
	"5",
	"6",
	"7",
	"8",
	"9",
	"10"
]

ctrList = list()
master = Tk()


masterColor = '#F8F8F8'
master.title("Data Analytics")
master.geometry("850x680")
#master.configure(bg = masterColor)
master.tk_setPalette(background = '#FFFFFF', foreground = 'black', activeBackground = 'black', activeForeground = '#FFFFFF')
variablef1 = StringVar(master)
variablef2 = StringVar(master)
variableClusterCount = StringVar(master)
variableCluster = StringVar(master)
variableEpsilon = StringVar()

varCheckArea = IntVar()
varCheckPer = IntVar()
varCheckRound = IntVar()
varCheckConvex = IntVar()
varCheckSol = IntVar()
varCheckMaAxis = IntVar()
varCheckMiAxis = IntVar()
varCheckOr = IntVar()
varCheckEcc = IntVar()
varCheckSIndex = IntVar()
varCheckColor = IntVar()

variableMinPoints = StringVar()

variableLessArea = DoubleVar()
variableMoreArea = DoubleVar()
variableLessPer = DoubleVar()
variableMorePer = DoubleVar()
variableLessRound = DoubleVar()
variableMoreRound = DoubleVar()
variableLessConvex = DoubleVar()
variableMoreConvex = DoubleVar()
variableLessSol = DoubleVar()
variableMoreSol = DoubleVar()
variableLessMaAxis = DoubleVar()
variableMoreMaAxis = DoubleVar()
variableLessMiAxis = DoubleVar()
variableMoreMiAxis = DoubleVar()
variableLessOr = DoubleVar()
variableMoreOr = DoubleVar()
variableLessEcc = DoubleVar()
variableMoreEcc = DoubleVar()
variableLessSIndex = DoubleVar()
variableMoreSIndex = DoubleVar()


featDimList = list()
temp = list()
tempArea = list()
tempPerimeter = list()
tempRoundness = list()
tempConvex = list()
tempSolidity = list()
tempMaAxis = list()
tempMiAxis = list()
tempOrientation = list()
tempEcc = list()
tempShapeIndex = list()
tempShadowRed = list()
tempShadowGreen = list()
tempShadowBlue = list()


fig = plt.figure()
fig2 = plt.figure()
canvas = FigureCanvasTkAgg(fig, master)
canvas2 = FigureCanvasTkAgg(fig2, master)
toolbar = NavigationToolbar2TkAgg( canvas, master )
toolbar.place(x=450, y=640, width = 440, height = 40)

selectFeatures_frame = Frame(width=420, height=440)

img = Image.open('cancerImage.jpg').resize((600, 600), Image.ANTIALIAS)
imgCropped = img.crop((0, 0, 350, 350))
imgToShow = ImageTk.PhotoImage(imgCropped)
labelFeature2 = Label(master, image=imgToShow)
labelFeature2.image = imgToShow
labelSaveFile = Label(master, text = "Image saved as maskCluster.jpg")

imgFeat = cv2.imread('cancerImage.jpg', cv2.IMREAD_COLOR)
mask = cv2.imread('black.png',0)

def showOrHide(*args):
	if variableCluster.get() == "K-Means":
		print "This is K-Means"
		MinpointsEntry.config(state = 'disabled')
		epsilonentry.config(state = 'disabled')
		optionClusterCount.config(state = 'normal')

	else:
		print "This is DBSCAN"
		MinpointsEntry.config(state = 'normal')
		epsilonentry.config(state = 'normal')
		optionClusterCount.config(state = 'disabled')

def generateUI():
	
	
	labelSave = Label(master, text="All features generated and saved to output.txt")
	labelSave.place(x=5, y=10)	

	labelCluster = Label(master, text="Please select the clustering method")
	labelCluster.place(x=5, y=30)

	variableCluster.set(CLUSTEROPTIONS[0]) # default value

	optionCluster = apply(OptionMenu, (master, variableCluster) + tuple(CLUSTEROPTIONS))
	optionCluster.place(x=5, y=55, width = 100)

	labelClusterCount = Label(master, text="Number of clusters")
	labelClusterCount.place(x=110, y=60)

	variableClusterCount.set(CLUSTERCOUNTOPTIONS[0]) # default value
	variableCluster.trace("w", showOrHide)

	optionClusterCount.place(x=240, y=55, width = 100)

	labelEpsilon = Label(master, text="Epsilon")
	labelEpsilon.place(x=350, y=60)

	epsilonentry.place(x=400, y=60, width = 50)
	variableEpsilon.set("25")
	epsilonentry.config(state = 'disabled')

	labelMinpoints = Label(master, text="Minimum Points")
	labelMinpoints.place(x=460, y=60)

	MinpointsEntry.place(x=565, y=60, width = 50)
	variableMinPoints.set("2")
	MinpointsEntry.config(state = 'disabled')

	labelFeature = Label(master, text="Select features to show")
	labelFeature.place(x=450, y=140)
	
	variablef1.set(FEATUREOPTIONS[0]) # default value
	
	variablef2.set(FEATUREOPTIONS[1]) # default value

	optionf1 = apply(OptionMenu, (master, variablef1) + tuple(FEATUREOPTIONS))
	optionf1.place(x=450, y=160, width = 150)

	optionf2 = apply(OptionMenu, (master, variablef2) + tuple(FEATUREOPTIONS))
	optionf2.place(x=610, y=160, width = 150)

	generateButton = Button(master, text="Generate the cluster", command = generateCluster)
	generateButton.place(x=5, y=170, width = 150)

	labelFeature2.place(x=945, y=230)
	
	labelHeader1 = Label(selectFeatures_frame, text="Select feature", font = "Verdana 10 bold")
	labelHeader2 = Label(selectFeatures_frame, text="<", font = "Verdana 10 bold")
	labelHeader3 = Label(selectFeatures_frame, text=">", font = "Verdana 10 bold")
	
	CArea = Checkbutton(selectFeatures_frame, text="Area", onvalue=True, variable=varCheckArea)
	CPer = Checkbutton(selectFeatures_frame, text="Perimeter", onvalue=True, variable=varCheckPer)
	CRound = Checkbutton(selectFeatures_frame, text="Roundess", onvalue=True, variable=varCheckRound)
	CConvex = Checkbutton(selectFeatures_frame, text="Convex Area", onvalue=True, variable=varCheckConvex)
	CSol = Checkbutton(selectFeatures_frame, text="Solidity", onvalue=True, variable=varCheckSol)
	CMaAxis = Checkbutton(selectFeatures_frame, text="Major Axis", onvalue=True, variable=varCheckMaAxis)
	CMiAxis = Checkbutton(selectFeatures_frame, text="Minor Axis", onvalue=True, variable=varCheckMiAxis)
	COr = Checkbutton(selectFeatures_frame, text="Orientation", onvalue=True, variable=varCheckOr)
	CEcc = Checkbutton(selectFeatures_frame, text="Eccentricity", onvalue=True, variable=varCheckEcc)
	CSIndex = Checkbutton(selectFeatures_frame, text="Shape Index", onvalue=True, variable=varCheckSIndex)
	CShadow = Checkbutton(selectFeatures_frame, text="Nucleus Shadow Color", onvalue=True, variable=varCheckColor)
	
	MinLessArea = Entry(selectFeatures_frame, textvariable = variableLessArea, width = 15)
	MinMoreArea = Entry(selectFeatures_frame, textvariable = variableMoreArea, width = 15)
	MinLessPer = Entry(selectFeatures_frame, textvariable = variableLessPer, width = 15)
	MinMorePer = Entry(selectFeatures_frame, textvariable = variableMorePer, width = 15)
	MinLessRound = Entry(selectFeatures_frame, textvariable = variableLessRound, width = 15)
	MinMoreRound = Entry(selectFeatures_frame, textvariable = variableMoreRound, width = 15)
	MinLessConvex = Entry(selectFeatures_frame, textvariable = variableLessConvex, width = 15)
	MinMoreConvex = Entry(selectFeatures_frame, textvariable = variableMoreConvex, width = 15)
	MinLessSol = Entry(selectFeatures_frame, textvariable = variableLessSol, width = 15)
	MinMoreSol = Entry(selectFeatures_frame, textvariable = variableMoreSol, width = 15)
	MinLessMaAxis = Entry(selectFeatures_frame, textvariable = variableLessMaAxis, width = 15)
	MinMoreMaAxis = Entry(selectFeatures_frame, textvariable = variableMoreMaAxis, width = 15)
	MinLessMiAxis = Entry(selectFeatures_frame, textvariable = variableLessMiAxis, width = 15)
	MinMoreMiAxis = Entry(selectFeatures_frame, textvariable = variableMoreMiAxis, width = 15)
	MinLessOr = Entry(selectFeatures_frame, textvariable = variableLessOr, width = 15)
	MinMoreOr = Entry(selectFeatures_frame, textvariable = variableMoreOr, width = 15)
	MinLessEcc = Entry(selectFeatures_frame, textvariable = variableLessEcc, width = 15)
	MinMoreEcc = Entry(selectFeatures_frame, textvariable = variableMoreEcc, width = 15)
	MinLessSIndex = Entry(selectFeatures_frame, textvariable = variableLessSIndex, width = 15)
	MinMoreSIndex = Entry(selectFeatures_frame, textvariable = variableMoreSIndex, width = 15)
	
	selectFeatures_frame.grid(column=0, row=0, columnspan=10, rowspan=2)
	labelHeader1.grid(column=0, row=1, padx=2, pady=2)
	labelHeader2.grid(column=1, row=1, padx=2, pady=2)
	labelHeader3.grid(column=2, row=1, padx=2, pady=2)
	
	CArea.grid(column=0, row=2, padx=2, pady=2, sticky="W") 
	CPer.grid(column=0, row=3, padx=2, pady=2, sticky="W") 
	CRound.grid(column=0, row=4, padx=2, pady=2, sticky="W") 
	CConvex.grid(column=0, row=5, padx=2, pady=2, sticky="W") 
	CSol.grid(column=0, row=6, padx=2, pady=2, sticky="W") 
	CMaAxis.grid(column=0, row=7, padx=2, pady=2, sticky="W") 
	CMiAxis.grid(column=0, row=8, padx=2, pady=2, sticky="W") 
	COr.grid(column=0, row=9, padx=2, pady=2, sticky="W") 
	CEcc.grid(column=0, row=10, padx=2, pady=2, sticky="W") 
	CSIndex.grid(column=0, row=11, padx=2, pady=2, sticky="W")
	CShadow.grid(column=0, row=12, padx=2, pady=2, sticky="W")
	
	MinLessArea.grid(column=1, row=2, padx=2, pady=2, sticky="W")
	MinMoreArea.grid(column=2, row=2, padx=2, pady=2, sticky="W") 
	MinLessPer.grid(column=1, row=3, padx=2, pady=2, sticky="W") 
	MinMorePer.grid(column=2, row=3, padx=2, pady=2, sticky="W") 
	MinLessRound.grid(column=1, row=4, padx=2, pady=2, sticky="W") 
	MinMoreRound.grid(column=2, row=4, padx=2, pady=2, sticky="W") 
	MinLessConvex.grid(column=1, row=5, padx=2, pady=2, sticky="W") 
	MinMoreConvex.grid(column=2, row=5, padx=2, pady=2, sticky="W") 
	MinLessSol.grid(column=1, row=6, padx=2, pady=2, sticky="W") 
	MinMoreSol.grid(column=2, row=6, padx=2, pady=2, sticky="W") 
	MinLessMaAxis.grid(column=1, row=7, padx=2, pady=2, sticky="W") 
	MinMoreMaAxis.grid(column=2, row=7, padx=2, pady=2, sticky="W") 
	MinLessMiAxis.grid(column=1, row=8, padx=2, pady=2, sticky="W") 
	MinMoreMiAxis.grid(column=2, row=8, padx=2, pady=2, sticky="W") 
	MinLessOr.grid(column=1, row=9, padx=2, pady=2, sticky="W") 
	MinMoreOr.grid(column=2, row=9, padx=2, pady=2, sticky="W") 
	MinLessEcc.grid(column=1, row=10, padx=2, pady=2, sticky="W") 
	MinMoreEcc.grid(column=2, row=10, padx=2, pady=2, sticky="W") 
	MinLessSIndex.grid(column=1, row=11, padx=2, pady=2, sticky="W") 
	MinMoreSIndex.grid(column=2, row=11, padx=2, pady=2, sticky="W") 

	selectFeatures_frame.place(x=5, y=230)
	w, h = master.winfo_screenwidth(), master.winfo_screenheight()
	print 'Screen - ', w, h
	master.geometry("%dx%d+0+0" % (w, h))
	master.mainloop()
	
def generateKMeans(featDimList, featDimListAllCond, ctrListAllCond, clusterCount):
	#featDimList = np.hstack((featDimList,tempPerimeter))
     
     featDimList = np.array(featDimList, dtype='f')
     #Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

     # Set flags (Just to avoid line break in the code)
     flags = cv2.KMEANS_RANDOM_CENTERS

     # Apply KMeans
     #compactness,labels,centers = cv2.kmeans(z,2,criteria,10,flags)
     ret,label,center = cv2.kmeans(featDimList,clusterCount,criteria,5,flags)
     A = [[0 for x in range(4000)] for x in range(4000)] 
     for i in range(0, clusterCount):
		A[i] = featDimListAllCond[label.ravel()==i]
	#Plot the data
	#rand_colours = [random.choice(colour) for i in range(clusterCount)]
     fig.clf()
     ax = fig.add_subplot(111)

     clusterPointsNP = np.array(ctrListAllCond)
     cluster = cv2.imread('black.png',cv2.IMREAD_COLOR)
	 
     for i in range(0, clusterCount):
     	ax.scatter(A[i][:,FEATUREOPTIONS.index(variablef1.get())],A[i][:,FEATUREOPTIONS.index(variablef2.get())], c = TENCOLORS[i])
     	temp = clusterPointsNP[np.where(label==i)[0]]
     	cv2.drawContours(cluster,temp, -1, TENCOLORSHEX[i], -1)
     #ax.scatter(center[:,1],center[:,2],s = 80,c = 'y', marker = 's')
     ax.set_title(variableCluster.get() + " clustering for " + variablef1.get() + " and " + variablef2.get(), fontsize = 12)
     ax.set_xlabel(variablef1.get(), fontsize = 12),ax.set_ylabel(variablef2.get(), fontsize = 12)
     canvas.show()
     canvas.get_tk_widget().pack()#place(x=5, y=410, width = 400, height = 20)
     toolbar.update()
     toolbar.place(x=450, y=640, width = 440, height = 40)
     canvas._tkcanvas.place(x=450, y=210, width = 460, height = 430)
     cv2.imwrite('maskCluster.jpg',cluster)
	 
def generateDBSCAN(featDimList, featDimListAllCond, ctrListAllCond):
	
	##############################################################################
	# Compute DBSCAN
	db = DBSCAN(eps=float(variableEpsilon.get()), min_samples=float(variableMinPoints.get())).fit(featDimList)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	clusterPointsNP = np.array(ctrListAllCond)
	cluster = cv2.imread('black.png',cv2.IMREAD_COLOR)
	temp = clusterPointsNP[np.where(labels==-1)[0]]
	cv2.drawContours(cluster,temp, -1, (0, 0, 255), -1)
	temp = clusterPointsNP[np.where(labels!=-1)[0]]
	cv2.drawContours(cluster,temp, -1, (255, 255, 255), -1)
	cv2.imwrite('maskCluster.jpg',cluster)
	##############################################################################
	# Plot result
	import matplotlib.pyplot as plt
	fig.clf()
	bx = fig.add_subplot(111)
	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Black used for noise.
			col = 'k'

		class_member_mask = (labels == k)

		xy = featDimListAllCond[class_member_mask & core_samples_mask]
		bx.plot(xy[:, FEATUREOPTIONS.index(variablef1.get())], xy[:, FEATUREOPTIONS.index(variablef2.get())], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=14)

		xy = featDimListAllCond[class_member_mask & ~core_samples_mask]
		bx.plot(xy[:, FEATUREOPTIONS.index(variablef1.get())], xy[:, FEATUREOPTIONS.index(variablef2.get())], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=6)

	bx.set_title(variableCluster.get() + " clustering for " + variablef1.get() + " and " + variablef2.get(), fontsize = 12)
	bx.set_xlabel(variablef1.get(), fontsize = 12),bx.set_ylabel(variablef2.get(), fontsize = 12)
	canvas.show()
	canvas.get_tk_widget().pack()#place(x=5, y=410, width = 400, height = 20)

	toolbar.update()
	toolbar.place(x=450, y=640, width = 440, height = 40)
	canvas._tkcanvas.place(x=450, y=210, width = 460, height = 430)


def generateCluster():
     
     clusterCount = int(variableClusterCount.get())
     featDimList = list()
     featDimList = tempArea  
     featDimListAllCond = list()
     ctrListAllCond = list()
     ctrListAllCond = np.array(ctrList)
     i = IntVar()
     i = 0
     if varCheckArea.get() == 1:
		featDimList = np.hstack((featDimList,tempArea))
		areaLoc = i
		i = i + 1
     if varCheckPer.get() == 1:
		featDimList = np.hstack((featDimList,tempPerimeter))
		perLoc = i
		i = i + 1
     if varCheckRound.get() == 1:
		featDimList = np.hstack((featDimList,tempRoundness))
		roundLoc = i
		i = i + 1
     if varCheckConvex.get() == 1:
		featDimList = np.hstack((featDimList,tempConvex))
		convexLoc = i
		i = i + 1
     if varCheckSol.get() == 1:
		featDimList = np.hstack((featDimList,tempSolidity))
		solLoc = i
		i = i + 1
     if varCheckMaAxis.get() == 1:
		featDimList = np.hstack((featDimList,tempMaAxis))
		maAxisLoc = i
		i = i + 1
     if varCheckMiAxis.get() == 1:
		featDimList = np.hstack((featDimList,tempMiAxis))
		miAxisLoc = i
		i = i + 1
     if varCheckOr.get() == 1:
		featDimList = np.hstack((featDimList,tempOrientation))
		oriLoc = i
		i = i + 1
     if varCheckEcc.get() == 1:
		featDimList = np.hstack((featDimList,tempEcc))
		eccLoc = i
		i = i + 1
     if varCheckSIndex.get() == 1:
		featDimList = np.hstack((featDimList,tempShapeIndex))
		sIndexLoc = i
		i = i + 1
     if varCheckColor.get() == 1:
		featDimList = np.hstack((featDimList,tempShadowRed))
		featDimList = np.hstack((featDimList,tempShadowGreen))
		featDimList = np.hstack((featDimList,tempShadowBlue))
     featDimList = featDimList[:, 1:]
     featDimListAllCond = featDimListAll
     if varCheckArea.get() == 1:
		featDimList = featDimList[featDimList[:,areaLoc] < variableLessArea.get()]
		featDimList = featDimList[featDimList[:,areaLoc] > variableMoreArea.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,0] < variableLessArea.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,0] < variableLessArea.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,0] > variableMoreArea.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,0] > variableMoreArea.get()]
     if varCheckPer.get() == 1:
		featDimList = featDimList[featDimList[:,perLoc] < variableLessPer.get()]
		featDimList = featDimList[featDimList[:,perLoc] > variableMorePer.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,1] < variableLessPer.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,1] < variableLessPer.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,1] > variableMorePer.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,1] > variableMorePer.get()]
     if varCheckRound.get() == 1:
		featDimList = featDimList[featDimList[:,roundLoc] < variableLessRound.get()]
		featDimList = featDimList[featDimList[:,roundLoc] > variableMoreRound.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,2] < variableLessRound.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,2] < variableLessRound.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,2] > variableMoreRound.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,2] > variableMoreRound.get()]
     if varCheckConvex.get() == 1:
		featDimList = featDimList[featDimList[:,convexLoc] < variableLessConvex.get()]
		featDimList = featDimList[featDimList[:,convexLoc] > variableMoreConvex.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,3] < variableLessConvex.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,3] < variableLessConvex.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,3] > variableMoreConvex.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,3] > variableMoreConvex.get()]
     if varCheckSol.get() == 1:
		featDimList = featDimList[featDimList[:,solLoc] < variableLessSol.get()]
		featDimList = featDimList[featDimList[:,solLoc] > variableMoreSol.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,4] < variableLessSol.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,4] < variableLessSol.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,4] > variableMoreSol.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,4] > variableMoreSol.get()]
     if varCheckMaAxis.get() == 1:
		featDimList = featDimList[featDimList[:,maAxisLoc] < variableLessMaAxis.get()]
		featDimList = featDimList[featDimList[:,maAxisLoc] > variableMoreMaAxis.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,5] < variableLessMaAxis.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,5] < variableLessMaAxis.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,5] > variableMoreMaAxis.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,5] > variableMoreMaAxis.get()]
     if varCheckMiAxis.get() == 1:
		featDimList = featDimList[featDimList[:,miAxisLoc] < variableLessMiAxis.get()]
		featDimList = featDimList[featDimList[:,miAxisLoc] > variableMoreMiAxis.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,6] < variableLessMiAxis.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,6] < variableLessMiAxis.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,6] > variableMoreMiAxis.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,6] > variableMoreMiAxis.get()]
     if varCheckOr.get() == 1:
		featDimList = featDimList[featDimList[:,oriLoc] < variableLessOr.get()]
		featDimList = featDimList[featDimList[:,oriLoc] > variableMoreOr.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,7] < variableLessOr.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,7] < variableLessOr.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,7] > variableMoreOr.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,7] > variableMoreOr.get()]
     if varCheckEcc.get() == 1:
		featDimList = featDimList[featDimList[:,eccLoc] < variableLessEcc.get()]
		featDimList = featDimList[featDimList[:,eccLoc] > variableMoreEcc.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,8] < variableLessEcc.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,8] < variableLessEcc.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,8] > variableMoreEcc.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,8] > variableMoreEcc.get()]
     if varCheckSIndex.get() == 1:
		featDimList = featDimList[featDimList[:,sIndexLoc] < variableLessSIndex.get()]
		featDimList = featDimList[featDimList[:,sIndexLoc] > variableMoreSIndex.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,9] < variableLessSIndex.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,9] < variableLessSIndex.get()]
		ctrListAllCond = ctrListAllCond[featDimListAllCond[:,9] > variableMoreSIndex.get()]
		featDimListAllCond = featDimListAllCond[featDimListAllCond[:,9] > variableMoreSIndex.get()]
     featDimList = np.array(featDimList, dtype='f')
     if variableCluster.get() == "K-Means":
		 generateKMeans(featDimList, featDimListAllCond, ctrListAllCond, clusterCount)
     else:
		 generateDBSCAN(featDimList, featDimListAllCond, ctrListAllCond)
     img = Image.open('maskCluster.jpg').resize((600, 600), Image.ANTIALIAS)
     imgCropped = img.crop((0, 0, 350, 350))
     imgToShow = ImageTk.PhotoImage(imgCropped)
     labelFeature2 = Label(master, image=imgToShow)
     labelFeature2.image = imgToShow
     labelFeature2.place(x=945, y=230)
     labelSaveFile.place(x=945, y=210)


def getFeature(x):
     if x == 1:
         feature = 'Area'
     elif x == 2:
         feature = 'Perimeter'
     elif x == 3:
         feature = 'Roundness'
     elif x == 4:
         feature = 'Convex Area'
     elif x == 5:
         feature = 'Solidity'
     elif x == 6:
         feature = 'Major Axis'
     elif x == 7:
         feature = 'Minor Axis'
     elif x == 8:
         feature = 'Orientation'
     elif x == 9:
         feature = 'Eccentricity'
     return feature


print "Please wait while we generate the feature set..."
# File to load all the 
states = pd.read_csv('path-image-100.seg.000000.000000.txt', sep ='\t')
f = os.open('output.txt', os.O_WRONLY)

Boundaries = 'Boundaries: x1,y1;x2,y2;...'

y = states[Boundaries].tolist()
thing = ''

#get all the
points = list()
for i in range(0,len(y)):
    pair = y[i].split(';')
    if thing in pair: pair.remove(thing)
    p = list()
    for j in range(0,len(pair)):
    	temp = pair[j].split(',')
    	temp = [int(float(k)) for k in temp]
    	p.append(temp)
    points.append(p)

ctrList = list()

os.write(f, 'Polygon\tArea\tPerimeter\tRoundness\tConvexity\tConvex Area\tSolidity\tCenter\tMajor Axis\tMinor Axis\tOrientation\tEccentricity\tShape Index\tShadowRed\tShadowBlue\tShadowGreen\n')

for i in range(0, len(points)):
     ctr = np.array(points[i])
     ctrList.append(ctr)
     area = cv2.contourArea(ctr)
     perimeter = cv2.arcLength(ctr,True)
     roundness = (4*3.14*area)/(perimeter*perimeter)
     isContour = cv2.isContourConvex(ctr)
     convex_hull = cv2.convexHull(ctr)
     convex_area = cv2.contourArea(convex_hull)
     solidity = area/convex_area
     (center,axis,angle) = cv2.fitEllipse(ctr)
     majorAxis = max(axis)
     minorAxis = min(axis)
     eccentricity=np.sqrt(1.0-(minorAxis/majorAxis)**2)
     edgeLength = 0
     lengthp = len(ctr)	
     btemp= [[0 for x in range(0,2)] for y in range(0,2*(len(ctr)-1))]
     #new= []
     #for j in range(0,2*len(ctr))
     cx=center[0]
     cy=center[1]
     #print cx
     #print cy
     for j in range(0, len(ctr)-1):
        if ctr[j][0]>=cx:
			btemp[j][0]=ctr[j][0]+1
        elif ctr[j][0]<cx: 
			btemp[j][0]=ctr[j][0]-1
        if ctr[j][1]>=cy:
			btemp[j][1]=ctr[j][1]+1
        if ctr[j][1]<cy:
			btemp[j][1]=ctr[j][1]-1
     for j in range(len(ctr)-1,0,-1):
		btemp[(len(ctr)-2)+j][0]=ctr[j][0]
		btemp[(len(ctr)-2)+j][1]=ctr[j][1]
     btemp=np.array(btemp)
     cv2.drawContours(mask,[btemp],0,(255, 0, 0),-1)
     meancolor=cv2.mean(imgFeat,mask = mask)
	 #APPEND AS THREE VALUED ARRAY RATHER THAN SINGLE ELEMENT
     #print 'Color - ', meancolor[0]
     shadowRed = meancolor[0]
     shadowGreen = meancolor[1]
     shadowBlue = meancolor[2]
     for j in range(0, len(ctr)-1):
        k = j + 1
        length = np.sqrt(((ctr[k][1] - ctr[j][1]) ** 2) + ((ctr[k][0] - ctr[j][0]) ** 2))
        edgeLength += length
     shapeIndex = (edgeLength)/(4*(np.sqrt(area)))
     os.write(f, str(i+1) + '\t' + str(area))
     os.write(f, '\t' + str(perimeter) + '\t' + str(roundness) + '\t' + str(isContour) + '\t' + str(convex_area) + '\t' + str(solidity))
     os.write(f, '\t' + str(center) + '\t' + str(majorAxis) + '\t' + str(minorAxis) + '\t' + str(angle) + '\t' + str(eccentricity) + '\t' + str(shapeIndex) + '\t' + str(shadowRed) + '\t' + str(shadowGreen) + '\t' + str(shadowBlue) +'\n')

os.fsync(f)
os.close
print 'File write done'

states = pd.read_csv('output.txt', sep ='\t')

for i in range(0, len(states["Area"].tolist())):
	tempArea.append([states["Area"].tolist()[i]])
	tempPerimeter.append([states["Perimeter"].tolist()[i]])
	tempRoundness.append([states["Roundness"].tolist()[i]])
	tempConvex.append([states["Convex Area"].tolist()[i]])
	tempSolidity.append([states["Solidity"].tolist()[i]])
	tempMaAxis.append([states["Major Axis"].tolist()[i]])
	tempMiAxis.append([states["Minor Axis"].tolist()[i]])
	tempOrientation.append([states["Orientation"].tolist()[i]])
	tempEcc.append([states["Eccentricity"].tolist()[i]])
	tempShapeIndex.append([states["Shape Index"].tolist()[i]])
	tempShadowRed.append([states["ShadowRed"].tolist()[i]])
	tempShadowGreen.append([states["ShadowGreen"].tolist()[i]])
	tempShadowBlue.append([states["ShadowBlue"].tolist()[i]])
featDimListAll = np.hstack((tempArea, tempPerimeter, tempRoundness, tempConvex, tempSolidity, tempMaAxis, tempMiAxis, tempOrientation, tempEcc, tempShapeIndex, tempShadowRed, tempShadowGreen, tempShadowBlue))
print featDimListAll

MinpointsEntry = Entry(master, textvariable = variableMinPoints)
epsilonentry = Entry(master, textvariable = variableEpsilon)
optionClusterCount = apply(OptionMenu, (master, variableClusterCount) + tuple(CLUSTERCOUNTOPTIONS))

generateUI()

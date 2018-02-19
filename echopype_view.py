"""
Classes for visualizing echo data
"""

import matplotlib.colors as colors


# Colormap: multi-frequency availability from Jech & Michaels 2006
MF_COLORS = np.array([[0,0,0],\
                      [86,25,148],\
                      [28,33,179],\
                      [0,207,239],\
                      [41,171,71],\
                      [51,204,51],\
                      [255,239,0],\
                      [255,51,0]])/255.
MF_CMAP_TMP = colors.ListedColormap(MF_COLORS)
MF_CMAP = colors.BoundaryNorm(range(MF_COLORS.shape[0]+1),MF_CMAP_TMP.N)


# Colormap: standard EK60
EK60_COLORS = np.array([[255, 255, 255],\
                          [159, 159, 159],\
                          [ 95,  95,  95],\
                          [  0,   0, 255],\
                          [  0,   0, 127],\
                          [  0, 191,   0],\
                          [  0, 127,   0],\
                          [255, 255,   0],\
                          [255, 127,   0],\
                          [255,   0, 191],\
                          [255,   0,   0],\
                          [166,  83,  60],\
                          [120,  60,  40]])/255.
EK60_CMAP_TH = [-80,-30]
EK60_CMAP_TMP = colors.ListedColormap(EK60_COLORS)
EK60_CMAP_BOUNDS = np.linspace(EK60_CMAP_TH[0],EK60_CMAP_TH[1],EK60_CMAP_TMP.N+1)
EK60_CMAP = colors.BoundaryNorm(EK60_CMAP_BOUNDS,EK60_CMAP_TMP.N)

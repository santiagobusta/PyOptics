"""
test.py

Python script for testing PyOptics

Using milimeters as unit distance
"""

import sys
sys.path.insert( 1 , '..\\')

import PyOptics as opt

WL = 0.000532 # Testing wavelenght
pp = 0.08 # Testing pixel pitch

#%% Interference

U1 = opt.beams.Gaussian( 1 , 5 , WL )
U2 = opt.beams.LaguerreGaussian( 1 , 5 , WL , 2 , 1 )

f = opt.__tools__.zeros( (501,501) ) # same as numpy.zeros
g = opt.__tools__.copy(f)

f = U1.profile( 100 , f , pp )
g = U2.profile( 0 , f , pp )

h = f + g

opt.improc.ExportImage( abs(g)**2 , '.\\images\\LaguerreProfile.png')
opt.improc.ExportImage( abs(h)**2 , '.\\images\\LaguerreGaussianInterferogram.png')
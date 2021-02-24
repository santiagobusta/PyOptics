"""
fourier.py

Fourier optics module
Contains:
    ~ Propagators:
        ~ FreeSpace
        ~ FFT2
        ~ IFFT2
    ~ Elements:
        ~ ThinLens
        ~ Pinhole

"""

import __tools__ as tl
from proc2d import FFT2, IFFT2

#%% Propagators

def FreeSpace( f , z , wl , dx  = 1.):
    """
    Propagates input 2D array in free space using the Fresnel diffraction integral via convolution.
    Voelz D. Computational Fourier Optics: A MATLAB Tutorial. 2011.
    
    ========Input=========
    
    f : Input complex amplitude profile (2D complex array)
    z : Propagation distance
    wl : Central wavelenght of the field
    dx : Pixel pitch (default value is unit of measurement)
    
    ========Output========
    
    h : Propagated complex amplitude profile (2D real or complex array)
    
    """
    
    n , m = f.shape
    Lx = dx*n ; Ly = dx*m # Square pixels
    zmax = dx*Lx/ wl
    F = FFT2(f)
    
    if( abs(z) > zmax ): # Propagating using Impulse Response Function
        x = tl.arange( -Lx/2 , Lx/2 , dx )
        y = tl.arange( -Ly/2 , Ly/2 , dx )
        X , Y = tl.meshgrid( x , y )
        g = tl.exp( 1j * tl.pi / ( wl * z ) * ( X**2 + Y**2 ) ) / ( 1j * wl * z )
        G = FFT2(g) * dx * dx
        
    else: # Propagating using Optical Transfer Function, Fraunhofer Approximations
        fx = tl.arange( -1/( 2 * dx ) , 1/( 2 * dx ) , 1/Lx )
        fy = tl.arange( -1/( 2 * dx ) , 1/( 2 * dx ) , 1/Ly )
        FX , FY = tl.meshgrid( fx , fy )
        G = tl.fftshift(tl.exp( ( -1j * wl * tl.pi * z ) * ( FX**2 + FY**2 ) ))
        
    h = IFFT2( F * G )
    
    return h

#%% Elements
    
def ThinLens( F , WL , f , dx = 1.):
    """
    Simulates the trasmittance function of a thin lens with focal lenght f for a wave with central wavelenght WL.
    
    ========Input=========
    
    F : Focal lenght of thin lens
    WL : Central wavelenght
    f :  2D array defining the output array shape
    dx : Pixel pitch (default value is unit of measurement)
    
    ========Raises=========
    
    TypeError: If f isn't a 2D array
    
    ========Output========
    
    t : Thin lens complex transmittance
    
    """
    
    if( len(tl.shape(f)) != 2 ):
        raise TypeError('f must be a 2D array')
    
    n , m = tl.shape( f )
    x = tl.arange( -m*dx/2 , m*dx/2 , dx ); y = tl.arange( -n*dx/2 , n*dx/2 , dx )
    X , Y = tl.meshgrid( x , y )
    t = tl.exp(-1j*(tl.pi/(WL*F))*(X**2+Y**2))
    
    return t

def Pinhole( R , f , dx = 1.):
    """
    Simulates the trasmittance function of a pinhole with radius R.
    
    ========Input=========
    
    R : Radius of pinhole
    f :  2D array defining the output array shape
    dx : Pixel pitch (default value is unit of measurement)
    
    ========Raises=========
    
    TypeError: If f isn't a 2D array
    
    ========Output========
    
    t : Pinhole complex transmittance
    
    """
    
    if( len(tl.shape(f)) != 2 ):
        raise TypeError('f must be a 2D array')
    
    n , m = tl.shape( f )
    x = tl.arange( -m*dx/2 , m*dx/2 , dx ); y = tl.arange( -n*dx/2 , n*dx/2 , dx )
    X , Y = tl.meshgrid( x , y )
    t = (X**2 + Y**2 <= R**2)*1.
    
    return t
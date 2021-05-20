"""
fourier.py

Fourier optics module
Contains:
    ~ Propagators:
        ~ AngularTF*
        ~ AngularIR*
        ~ FresnelTF
        ~ FresnelIR
        ~ FresnelCSC
        ~ Fresnel2S*
        ~ Fraunhofer*
    ~ Elements:
        ~ Tilt*
        ~ Pupil
        ~ Lens
        ~ LSL* (light sword lens)
        ~ CosGrating*
        ~ RecGrating*

"""

import __tools__ as tl
from proc2d import FFT2, IFFT2, GetRBW

#%% Propagators

def FresnelTF( f , z , wl , dx ):
    """
    ~ Propagates f in free space sampling the Transfer Function
    Voelz D. Computational Fourier Optics: A MATLAB Tutorial. 2011.
    
    ========Input=========
    
    f :     Input complex amplitude profile (2D complex array)
    z :     Propagation distance
    wl :    Central wavelenght of the field
    dx :    Sampling interval (default value is unit of measurement)
    SBC :   Use the source bandwidth criterion for large propagation regime
    
    ========Output========
    
    h :     Propagated complex amplitude profile (2D complex array)
    """
    
    n , m = f.shape
    Lx = dx*n ; Ly = dx*m 
    F = FFT2(f)
    fx = tl.arange( -1/( 2 * dx ) , 1/( 2 * dx ) , 1/Lx )
    fy = tl.arange( -1/( 2 * dx ) , 1/( 2 * dx ) , 1/Ly )
    FX , FY = tl.meshgrid( fx , fy )
    G = tl.exp( ( -1j * wl * tl.pi * z ) * ( FX**2 + FY**2 ) )
    h = IFFT2( F * G )
    
    return h
    
def FresnelIR( f , z , wl , dx ):
    """
    ~ Propagates f in free space sampling the Impulse Response
    Voelz D. Computational Fourier Optics: A MATLAB Tutorial. 2011.
    
    ========Input=========
    
    f :     Input complex amplitude profile (2D complex array)
    z :     Propagation distance
    wl :    Central wavelenght of the field
    dx :    Sampling interval (default value is unit of measurement)
    
    ========Output========
    
    h :     Propagated complex amplitude profile (2D complex array)
    
    """
    
    n , m = f.shape
    Lx = dx*n ; Ly = dx*m
    F = FFT2(f)
    x = tl.arange( -Lx/2 , Lx/2 , dx )
    y = tl.arange( -Ly/2 , Ly/2 , dx )
    X , Y = tl.meshgrid( x , y )
    g = tl.exp( 1j * tl.pi / ( wl * z ) * ( X**2 + Y**2 ) ) / ( 1j * wl * z )
    G = FFT2(g) * dx * dx
    h = IFFT2( F * G )
    
    return h

def FresnelCS( f , z , wl , dx  = 1., SBC = False):
    """
    ~ Propagates input 2D array in free space using the critical sampling criterion
    ~ Samples either the Impulse Response or the Transfer Function given the critic sampling criterion
    Voelz D. Computational Fourier Optics: A MATLAB Tutorial. 2011.
    
    ========Input=========
    
    f :     Input complex amplitude profile (2D complex array)
    z :     Propagation distance
    wl :    Central wavelenght of the field
    dx :    Sampling interval (default value is unit of measurement)
    SBC :   Use the source bandwidth criterion for large propagation regime
    
    ========Output========
    
    h :     Propagated complex amplitude profile (2D complex array)
    
    """
    
    n, m = tl.shape(f) # Array dimensions
    Lx = dx*m ; Ly = dx*n # Source plane side lenghts
    zc = dx*Lx/ wl # Critic sampling propagation distance
    F = FFT2(f)
    
    if(SBC): # Check the source bandwidth criterion
        B = GetRBW( f , dx )
        SBC = B <= min([Lx,Ly])/(wl*z)
    
    if( abs(z) > zc and not SBC ): # Propagating using Impulse Response
        x = tl.arange( -Lx/2. , Lx/2. , dx )
        y = tl.arange( -Ly/2. , Ly/2. , dx )
        X , Y = tl.meshgrid( x , y )
        g = tl.exp( 1j * tl.pi / ( wl * z ) * ( X**2 + Y**2 ) ) / ( 1j * wl * z )
        G = FFT2(g) * dx * dx
    
    if( abs(z) <= zc ): # Propagating using Transfer Function
        F = 0.5/dx # Nyquist frequency
        fx = tl.arange( -F , F , 1./Lx )
        fy = tl.arange( -F , F , 1./Ly )
        FX , FY = tl.meshgrid( fx , fy )
        G = tl.exp( ( -1j * wl * tl.pi * z ) * ( FX**2 + FY**2 ) )
    

        
    h = IFFT2( F * G )
    
    return h

def Fresnel2S( f , z , wl , L1 , L2):
    """
    ~ Propagates input 2D array in free space using the two-step Fresnel propagator
    Voelz D. Computational Fourier Optics: A MATLAB Tutorial. 2011.
    
    ========Input=========
    
    f :     Input complex amplitude profile (2D complex array)
    z :     Propagation distance
    wl :    Central wavelenght of the field
    L1 :    Input sample plane side lenght
    L2 :    Output sample plane side lenght
    
    ========Output========
    
    h :     Propagated complex amplitude profile (2D complex array)
    
    """
    
    if( tl.ndim(f) != 2 ):
        raise TypeError("Input array must be a 2D square array")
    
    m , n = tl.shape(f)
    
    if( m != n ):
        raise ValueError("Input array must be a 2D square array")
    
    k = 2*tl.pi/wl
    
    # Source plane
    dx1 = L1/m
    x1 = tl.arange(-L1/2.,L1/2.,dx1)
    X , Y = tl.meshgrid(x1,x1)
    F = f*tl.exp(1j*k/(2.*z*L1)*(L1-L2)*(X**2+Y**2))
    F = tl.fft.fft2(tl.fft.fftshift(F))
    
    # Dummy plane
    fx1 = tl.arange(-1./(2*dx1) , 1./(2*dx1) , 1./L1)
    fx1 = tl.fft.fftshift(fx1)
    FX1 , FY1 = tl.meshgrid(fx1,fx1)
    G = F*tl.exp(-1j*tl.pi*wl*z*L1/L2*(FX1**2+FY1**2))
    g = tl.fft.ifftshift(tl.fft.ifft2(G))
    
    # Observation plane
    dx2 = L2/m
    x2 = tl.arange(-L2/2.,L2/2.,dx2)
    X , Y = tl.meshgrid( x2 , x2 )
    g = (L2/L1)*g*tl.exp(-1j*k/(2.*z*L2)*(L1-L2)*(X**2+Y**2))
    g = g*(dx1/dx2)**2
    return g

#%% Elements
    
def Pupil( R , shape , dx = 1.):
    """
    ~ Simulates the trasmittance function of a pinhole with radius R
    ~ This  is essentially a circ function
    
    ========Input=========
    
    R :         Radius of pinhole
    shape :     Array shape
    dx :        Sampling interval (default value is unit of measurement)
    
    ========Raises=========
    
    TypeError:  If f isn't a 2D array
    
    ========Output========
    
    t :     Pinhole transmittance
    
    """
    
    if( len(shape) != 2 ):
        raise TypeError('f must be a 2D array')
    
    n , m = shape
    x = tl.arange( -m*dx/2 , m*dx/2 , dx ); y = tl.arange( -n*dx/2 , n*dx/2 , dx )
    X , Y = tl.meshgrid( x , y )
    t = (X**2 + Y**2 <= R**2)*1.
    
    return t
 
def Lens( F , R , wl , shape , dx = 1.):
    """
    ~ Samples the trasmittance function of a thin lens with focal lenght f for a wave with central wavelenght wl with sampling interval dx
    
    ========Input=========
    
    F :         Focal lenght of thin lens
    R :         Radius of the lens
    wl :        Central wavelenght
    shape :     Array shape
    dx :        Sampling interval (default value is unit of measurement)
    
    ========Raises=========
    
    TypeError :     If f isn't a 2D array
    
    ========Output========
    
    t :     Thin lens complex transmittance
    
    """
    
    if( len(shape) != 2 ):
        raise TypeError('Shape must be 2D')
    
    n , m = shape
    x = tl.arange( -m*dx/2 , m*dx/2 , dx ); y = tl.arange( -n*dx/2 , n*dx/2 , dx )
    X , Y = tl.meshgrid( x , y )
    t = tl.exp(-1j*(tl.pi/(wl*F))*(X**2+Y**2))
    t *= Pupil( R , shape , dx )
    
    return t
"""
beams.py

Beam optics module
Contains:
    \ Beams:
        ~ Plane
        ~ Spherical
        ~ Bessel
        ~ Gaussian
        ~ HermiteGaussian
        ~ LaguerreGaussian
"""
import __tools__ as tl

#%% Beams

class Plane:
    
    def __init__( self , AO , WLO , kxO = 0 , kyO = 0 ): # These parameters define the units of measurement
        """
        Plane wave with wavevector (2*pi/WLO)*[sqrt(kxO),sqrt(kyO),sqrt(1 - abs(kxO) - abs(kyO))]
        Default values for a plane wave propagating in the z axis.
        If |kxO| + |kyO| > 1 the wave is evanescent.
        """
        self.A = complex(AO) # Amplitude
        self.WL = float(WLO) # Central wavelenght
        self.k = 2*tl.pi/WLO # Central wavenumber
        self.kx = float(kxO)*self.k # x wavevector component
        self.ky = float(kyO)*self.k # y wavevector component
        self.kz = tl.sqrt(self.k**2 - self.kx**2 - self.ky**2) # z wavevector component
        
    def value( self , x , y , z ):
        """
        Value of complex amplitude at coordinates (x,y,z).
        (x,y,z) is a cartesian coordinate frame.
        The wavevector defines the propagation axis.
        
        ========Input=========
        
        x : Horizontal cartesian coordinate
        y : Vertical cartesian coordinate
        z : Default axis cartesian coordinate
    
        ========Output========
        
        u : Complex amplitude of beam at coordinates (x,y,z)
        
        """
        
        u = self.A * tl.exp(-1j*(self.kx*x + self.ky*y + self.kz*z))
        
        return u
    
    def profile( self , z , f , dx = 1. ):
        """
        Beam's complex amplitude at an axial distance z.
        Plane lies perpendicular to z axis and has the same shape as f.
        
        ========Input=========
    
        z : Axial distance
        f : 2D Array defining the output array shape
        dx : Pixel pitch (default value is unit of measurement)
        
        ========Raises========
        
        TypeError: If f isn't a 2D array
    
        ========Output========
    
        g : 2D complex array (complex amplitude at an axial distance z from waist centered at z axis)
        
        """
        
        if( len(tl.shape(f)) != 2 ):
            raise TypeError('f must be a 2D array')
        
        n , m = tl.shape( f )
        x = tl.arange( -m*dx/2 , m*dx/2 , dx ); y = tl.arange( -n*dx/2 , n*dx/2 , dx )
        X , Y = tl.meshgrid( x , y )
        g = self.value( X , Y , z )
        
        return g
    
class Spherical:
    
    def __init__( self , AO , WLO ): # These parameters define the units of measurement
        """
        Spherical wave with central wavelenght WLO and amplitude AO.
        """
        self.A = complex(AO) # Amplitude
        self.WL = float(WLO) # Central wavelenght
        self.k = 2*tl.pi/WLO # Central wave number
        
    def value( self , r ):
        """
        Value of complex amplitude at a distance r from wave source.
        
        ========Input=========
        
        r : Radial coordinate centered at wave source
    
        ========Output========
        
        u : Complex amplitude of beam at coordinate r
        
        """
        
        u = tl.zeros_like(r, dtype='complex')
        u[ r == 0. ] = tl.inf
        u[ r != 0. ] = self.A * tl.exp(-1j*self.k*r[ r != 0. ]) / r[ r != 0. ]
        
        return u
    
    def profile( self , z , f , dx = 1. ):
        """
        Beam's complex amplitude at an axial distance z.
        Plane lies perpendicular to z axis and has the same shape as f.
        
        ========Input=========
    
        z : Axial distance
        f : 2D Array defining the output array shape
        dx : Pixel pitch (default value is unit of measurement)
        
        ========Raises========
        
        TypeError: If f isn't a 2D array
    
        ========Output========
    
        g : 2D complex array (complex amplitude at an axial distance z from waist centered at z axis)
        
        """
        
        if( len(tl.shape(f)) != 2 ):
            raise TypeError('f must be a 2D array')
        
        n , m = tl.shape( f )
        x = tl.arange( -m*dx/2 , m*dx/2 , dx ); y = tl.arange( -n*dx/2 , n*dx/2 , dx )
        X , Y = tl.meshgrid( x , y )
        R = tl.sqrt(X**2 + Y**2 + z**2)
        g = self.value( R )
        
        return g

class Bessel:
    
    def __init__( self , AO , WLO , bO , m ): # These parameters define the units of measurement
        self.A = complex(AO) # Amplitude
        self.WL = float(WLO) # Central wavelength
        self.k = 2*tl.pi/WLO # Central wave number
        self.b = bO # Axial wave number
        self.kT = tl.sqrt(self.k**2 - self.b**2) # Transverse wave number
        self.m = int(m) # Spin eigenvalue
        
    def value( self , r , p , z ):
        """
        Value of complex amplitude at coordinates (r,p,z).
        (r,p,z) is a cylindrical coordinate frame centered at the waist.
        z lies along the propagation axis.
        
        ========Input=========
        
        r : Radial distance from z axis
        p : Azimuth angle
        z : Axial distance from waist
    
        ========Output========
        
        u : Complex amplitude of beam at coordinates (r,z)
        
        """
        
        env = self.A*tl.jv( self.m , self.kT*r )*tl.exp(1j*self.m*p)
        u = env * tl.exp( -1j*self.b*z )
        
        return u
    
    def profile( self , z , f , dx = 1. ):
        """
        Beam's complex amplitude at an axial distance z from waist.
        Plane lies perpendicular to z axis and has the same shape as f.
        
        ========Input=========
    
        z : Axial distance from waist
        f : 2D Array defining the output array shape
        dx : Pixel pitch (default value is unit of measurement)
        
        ========Raises========
        
        TypeError: If f isn't a 2D array
    
        ========Output========
    
        g : 2D complex array (complex amplitude at an axial distance z from waist centered at z axis)
        
        """
        
        if( len(tl.shape(f)) != 2 ):
            raise TypeError('f must be a 2D array')
        
        n , m = tl.shape( f )
        x = tl.arange( -m*dx/2 , m*dx/2 , dx ); y = tl.arange( -n*dx/2 , n*dx/2 , dx )
        X , Y = tl.meshgrid( x , y )
        R = tl.sqrt(X**2 + Y**2)
        P = tl.arctan(Y/X)
        g = self.value( R , P , z )
        
        return g

class Gaussian:
    
    def __init__( self , AO , WO , WLO ): # These parameters define the units of measurement
        self.A = complex(AO) # Amplitude
        self.W = float(WO) # Waist radius (containing approximately 86 percent of total power)
        self.WL = float(WLO) # Central wavelength
        self.k = 2*tl.pi/WLO # Central wave number
        self.zO = tl.pi*WO**2/WLO # Rayleight range
        self.angle = 2*WLO/(tl.pi*WO) # Divergence angle
        self.peak = abs(AO)**2 # Peak intensity
        self.power = 0.5*self.peak*tl.pi*WO**2 # Power carried by the beam
        
    def q( self , z ): # q-parameter
        return z + 1j*self.zO
        
    def width( self , z ): # Width at axial distance z from waist
        return self.W * tl.sqrt( 1 + ( z / self.zO )**2 )
    
    def curv_radius( self , z ):
        """
        Wavefront radius of curvature at axial distance z from waist.
        
        ========Input=========
        
        z : Axial distance from waist
    
        ========Output========
        
        R : Wavefront radius of curvature at axial distance z from waist.
        
        """
        
        R = tl.zeros_like(z)
        R[ z == 0. ] = tl.inf
        R[ z != 0. ] = z[ z != 0. ] * ( 1 + ( self.zO / z[ z != 0. ] )**2 )
            
        return R
    
    def gouy_phase( self , z ): # Phase retardation due to Gouy effect
        return tl.arctan( z / self.zO )
    
    def value( self , r , z ):
        """
        Value of complex amplitude at coordinates (r,z).
        (r,z) is a cylindrical coordinate frame centered at the waist.
        z lies along the propagation axis.
        
        ========Input=========
        
        r : Radial distance from z axis
        z : Axial distance from waist
    
        ========Output========
        
        u : Complex amplitude of beam at coordinates (r,z)
        
        """
        
        q = self.q(z)
        env = 1j * self.zO * self.A * tl.exp( -0.5j*self.k*r**2/q ) / q
        u = env * tl.exp( -1j*self.k*z )
        
        return u
    
    def profile( self , z , f , dx = 1. ):
        """
        Beam's complex amplitude at an axial distance z from waist.
        Plane lies perpendicular to z axis and has the same shape as f.
        
        ========Input=========
    
        z : Axial distance from waist
        f : 2D Array defining the output array shape
        dx : Pixel pitch (default value is unit of measurement)
        
        ========Raises========
    
        ========Output========
    
        g : 2D complex array (complex amplitude at an axial distance z from waist centered at z axis)
        
        """
        
        n , m = tl.shape( f )
        x = tl.arange( -m*dx/2 , m*dx/2 , dx ); y = tl.arange( -n*dx/2 , n*dx/2 , dx )
        X , Y = tl.meshgrid( x , y )
        R = tl.sqrt( X**2 + Y**2 )
        g = self.value( R , z )
        return g
        
class HermiteGaussian:
    
    def __init__( self , AO , WO , WLO , l , m ): # These parameters define the units of measurement
        self.A = complex(AO) # Amplitude
        self.W = float(WO) # Waist radius
        self.WL = float(WLO) # Central wavelength
        self.k = 2*tl.pi/WLO # Central wave number
        self.zO = tl.pi*WO**2/WLO # Rayleight range
        self.l = int(l) # Horizontal eigenvalue
        self.m = int(m) # Vertical eigenvalue
    
    def q( self , z ): # q-parameter
        return z + 1j*self.zO
        
    def width( self , z ): # Width at axial distance z from waist
        return self.W * tl.sqrt( 1 + ( z / self.zO )**2 )
    
    def curv_radius( self , z ):
        """
        Wavefront radius of curvature at axial distance z from waist.
        
        ========Input=========
        
        z : Axial distance from waist
    
        ========Output========
        
        R : Wavefront radius of curvature at axial distance z from waist.
        
        """
        
        R = tl.zeros_like(z)
        R[ z == 0. ] = tl.inf
        R[ z != 0. ] = z[ z != 0. ] * ( 1 + ( self.zO / z[ z != 0. ] )**2 )
            
        return R
    
    def gouy_phase( self , z ): # Phase retardation due to Gouy effect
        return tl.arctan( z / self.zO )
    
    def value( self , x , y , z ):
        """
        Value of complex amplitude at coordinates (x,y,z).
        (x,y,z) is a cartesian coordinate frame centered at the waist.
        z lies along the propagation axis.
        
        ========Input=========
        
        x : Radial horizontal distance from z axis
        y : Radial vertical distance from z axis
        z : Axial distance from waist
    
        ========Output========
        
        u : Complex amplitude of beam at coordinates (x,y,z)
        
        """
        
        q = self.q(z)
        W = self.width(z)
        env = 1j * self.zO * self.A * tl.exp( -0.5j*self.k*(x**2+y**2)/q ) / q
        env *= tl.eval_hermite( self.l , tl.sqrt(2)*x/W )*tl.eval_hermite( self.m , tl.sqrt(2)*y/W )*tl.exp(1j*(self.l+self.m)*self.gouy_phase(z))
        u = env * tl.exp( -1j*self.k*z )
        
        return u
    
    def profile( self , z , f , dx = 1. ):
        """
        Beam's complex amplitude at an axial distance z from waist.
        Plane lies perpendicular to z axis and has the same shape as f.
        
        ========Input=========
    
        z : Axial distance from waist
        f : 2D Array defining the output array shape
        dx : Pixel pitch (default value is unit of measurement)
        
        ========Raises========
    
        ========Output========
    
        g : 2D complex array (complex amplitude at an axial distance z from waist centered at z axis)
        
        """
        
        n , m = tl.shape( f )
        x = tl.arange( -m*dx/2 , m*dx/2 , dx ); y = tl.arange( -n*dx/2 , n*dx/2 , dx )
        X , Y = tl.meshgrid( x , y )
        g = self.value( X , Y , z )
        return g
    
class LaguerreGaussian:
    
    def __init__( self , AO , WO , WLO , l , m ): # These parameters define the units of measurement
        self.A = complex(AO) # Amplitude
        self.W = float(WO) # Waist radius
        self.WL = float(WLO) # Central wavelength
        self.k = 2*tl.pi/WLO # Central wave number
        self.zO = tl.pi*WO**2/WLO # Rayleight range
        self.l = int(l) # Orbital angular momentum eigenvalue
        self.m = int(m) # Spin eigenvalue
    
    def q( self , z ): # q-parameter
        return z + 1j*self.zO
        
    def width( self , z ): # Width at axial distance z from waist
        return self.W * tl.sqrt( 1 + ( z / self.zO )**2 )
    
    def curv_radius( self , z ):
        """
        Wavefront radius of curvature at axial distance z from waist.
        
        ========Input=========
        
        z : Axial distance from waist
    
        ========Output========
        
        R : Wavefront radius of curvature at axial distance z from waist.
        
        """
        
        R = tl.zeros_like(z)
        R[ z == 0. ] = tl.inf
        R[ z != 0. ] = z[ z != 0. ] * ( 1 + ( self.zO / z[ z != 0. ] )**2 )
            
        return R
    
    def gouy_phase( self , z ): # Phase retardation due to Gouy effect
        return tl.arctan( z / self.zO )
    
    def value( self , r , p , z ):
        """
        Value of complex amplitude at coordinates (r,p,z).
        (r,p,z) is a cylindrical coordinate frame centered at the waist.
        z lies along the propagation axis.
        
        ========Input=========
        
        r : Radial distance from z axis
        p : Azimuth angle
        z : Axial distance from waist
    
        ========Output========
        
        u : Complex amplitude of beam at coordinates (r,p,z)
        
        """
        
        q = self.q(z)
        W = self.width(z)
        env = 1j * self.zO * self.A * tl.exp( -0.5j*self.k*r**2/q ) / q
        env *= (r/W)**self.l*tl.eval_genlaguerre( self.m , self.l , 2*(r/W)**2 )*tl.exp(1j*(self.l + 2*self.m)*self.gouy_phase(z)-1j*self.l*p )
        u = env * tl.exp( -1j*self.k*z )
        
        return u
    
    def profile( self , z , f , dx = 1. ):
        """
        Beam's complex amplitude at an axial distance z from waist.
        Plane lies perpendicular to z axis and has the same shape as f.
        
        ========Input=========
        
        z : Axial distance from waist
        f : 2D Array defining the output array shape
        dx : Pixel pitch (default value is unit of measurement)
        
        ========Raises========
        
        TypeError: If f isn't a 2D array
    
        ========Output========
    
        g : 2D complex array (complex amplitude at an axial distance z from waist centered at z axis)
        
        """
        
        if( len(tl.shape(f)) != 2 ):
            raise TypeError('f must be a 2D array')
        
        n , m = tl.shape( f )
        x = tl.arange( -m*dx/2 , m*dx/2 , dx ); y = tl.arange( -n*dx/2 , n*dx/2 , dx )
        X , Y = tl.meshgrid( x , y )
        R = tl.sqrt(X**2 + Y**2)
        P = tl.arctan(Y/X)
        g = self.value( R , P , z )
        
        return g
    
class BesselGaussian:
    """
    Gori F, Guattari G, Padovani C. BESSEL-GAUSS BEAMS. Opt Comm 1987;64:491-5.
    """
    
    def __init__( self , AO , WO , WLO, bO ): # These parameters define the units of measurement
        self.A = complex(AO) # Amplitude
        self.W = float(WO) # Waist radius
        self.WL = float(WLO) # Central wavelength
        self.k = 2*tl.pi/WLO # Central wave number
        self.b = bO # Axial wave number
        self.kT = tl.sqrt(self.k**2 - self.b**2) # Transverse wave number
        self.zO = tl.pi*WO**2/WLO # Rayleight range
    
    def q( self , z ): # q-parameter
        return z + 1j*self.zO
        
    def width( self , z ): # Width at axial distance z from waist
        return self.W * tl.sqrt( 1 + ( z / self.zO )**2 )
    
    def curv_radius( self , z ):
        """
        Wavefront radius of curvature at axial distance z from waist.
        
        ========Input=========
        
        z : Axial distance from waist
    
        ========Output========
        
        R : Wavefront radius of curvature at axial distance z from waist.
        
        """
        
        R = tl.zeros_like(z)
        R[ z == 0. ] = tl.inf
        R[ z != 0. ] = z[ z != 0. ] * ( 1 + ( self.zO / z[ z != 0. ] )**2 )
            
        return R
    
    def gouy_phase( self , z ): # Phase retardation due to Gouy effect
        return tl.arctan( z / self.zO )
    
    def value( self , r , z ):
        """
        Value of complex amplitude at coordinates (r,z).
        (r,z) is a cylindrical coordinate frame centered at the waist.
        z lies along the propagation axis.
        
        ========Input=========
        
        r : Radial distance from z axis
        z : Axial distance from waist
    
        ========Output========
        
        u : Complex amplitude of beam at coordinates (r,z)
        
        """
        
        q = self.q(z)
        env = 1j * self.zO * self.A * tl.exp( -0.5j*self.k*(r**2 + (self.b*z/self.k)**2 )/q ) / q
        env *= tl.jv( 0 , self.b*r/(1. + 1j*z/self.zO) )
        u = env * tl.exp( -1j*(self.k - self.b**2/(2.*self.k) )*z )
        
        return u
    
    def profile( self , z , f , dx = 1. ):
        """
        Beam's complex amplitude at an axial distance z from waist.
        Plane lies perpendicular to z axis and has the same shape as f.
        
        ========Input=========
    
        z : Axial distance from waist
        f : 2D Array defining the output array shape
        dx : Pixel pitch (default value is unit of measurement)
        
        ========Raises========
        
        TypeError: If f isn't a 2D array
        
        ========Output========
    
        g : 2D complex array (complex amplitude at an axial distance z from waist centered at z axis)
        
        """
        
        if( len(tl.shape(f)) != 2 ):
            raise TypeError('f must be a 2D array')
        
        n , m = tl.shape( f )
        x = tl.arange( -m*dx/2 , m*dx/2 , dx ); y = tl.arange( -n*dx/2 , n*dx/2 , dx )
        X , Y = tl.meshgrid( x , y )
        R = tl.sqrt(X**2 + Y**2)
        g = self.value( R , z )
        
        return g
    
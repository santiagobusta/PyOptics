"""
proc2d.py

2D signal processing module
Contains:
    ~ Input/Output:
        ~ ExportImage
        ~ ImportImage
    ~ 2D Signals:
        ~ Rect
        ~ Ellipse
        ~ Gauss
    ~ Manipulation:
        ~ Trim
        ~ Embed
    ~ Normalization:
        ~ LinearNormal
        ~ QuadNormal
        ~ SigmoidNormal
    ~ Metrics:
        ~ CC
        ~ PSNR
        ~ Entropy
    ~ Binarization:
        ~ ErrDiff
        ~ QZP
    ~ Bidimensional Fourier Analysis:
        ~ FFT2
        ~ IFFT2
        ~ LowPass
        ~ HighPass
        ~ BandPass
        ~ BandStop
"""

import __tools__ as tl

#%% Input/Output

def ExportImage( f , path , normalize = True ):
    """
    Exports real 2D input array f as an 8-bit image.
    
    ========Input=========
    
    f : Real 2D input array with [0,1] dynamic range
    path : Image save path
    normalize : Normalize input array before saving (default value for linear normalization to [0,1])
    
    ========Raises========
    
    TypeError : If f isn't a real 2D array
    
    """
    if( (tl.isreal(f).all == False) or (len(tl.shape(f)) != 2) ):
        raise TypeError('Input array must be a real 2D array')
        
    if( normalize ):
        f = LinearNormal(f)
        
    f = tl.Image.fromarray(f*255)
    f = f.convert('L')
    f.save(path)
    
def ImportImage( path , normalize = True ):
    """
    Imports image as a real 2D array
    
    ========Input=========

    path : Image load path
    normalize : Normalize output array (default value for linear normalization to [0,1])
    
    ========Raises========
    
    OSError: If path doesn't exist
    
    ========Output=========
    
    f : 2D array containing the image information
    
    """
    
    if( tl.isfile(path) == False ):
        raise OSError('File not found')
    
    f = tl.Image.open( path ).convert( 'L' )
    f = tl.array(f , dtype='float')
    
    if( normalize ):
        f = LinearNormal(f)
        
    return f

#%% 2D Signals
    
def Rect( w , h , shape , x = 0 , y = 0 , angle = 0 , use_pxc = False ):
    """
    2D rectangular signal.
    (x,y) coordinates are given in a centered cartesian frame.
    angle is measured counterclockwise in radians.

    ========Input=========
    
    w :         Width
    h :         Height
    shape :     Output array shape
    x :         Horizontal center of rectangle (Default for horizontally centered)
    y :         Vertical center of rectangle (Default for vertically centered)
    angle :     Angle defining the width direction (Default for horizontal direction)
    use_pxc :   Use pixel as unit of measurement (Default for shape as unit of measurement)
    
    ========Output========
    
    f : 2D float array

    """
    
    x += 0.5 ; y += 0.5
    
    if( use_pxc == False ):
        w *= shape[1] ; h *= shape[0]
        x *= shape[1] ; y *= shape[0]
    
    w = w/2 ; h = h/2
    X = range(shape[1]) ; Y = range(shape[0])
    X, Y = tl.meshgrid(X , Y)
    
    f = ((X-x)*tl.cos(angle) - (Y-y)*tl.sin(angle) > - w)*((X-x)*tl.cos(angle) - (Y-y)*tl.sin(angle) <= w)*((Y-y)*tl.cos(angle) + (X-x)*tl.sin(angle) > - h)*((Y-y)*tl.cos(angle) + (X-x)*tl.sin(angle) <= h)*1
    
    return f

def Ellipse( a , b , shape , x = 0 , y = 0 , angle = 0 , use_pxc = False ):
    """
    2D ellipse signal.
    (x,y) coordinates are given in a centered cartesian frame.
    angle is measured counterclockwise in radians.

    ========Input=========
    
    a :         Semi-major axis
    b :         Semi-minor axis
    shape :     Output array shape
    x :         Horizontal center of ellipse (Default for horizontally centered)
    y :         Vertical center of ellipse (Default for vertically centered)
    angle :     Angle defining the major axis direction (Default for horizontal direction)
    use_pxc :   Use pixel as unit of measurement (Default for shape as unit of measurement)
    
    ========Output========
    
    f : 2D float array

    """
    
    x += 0.5 ; y += 0.5
    
    if( use_pxc == False ):
        a *= shape[1] ; b *= shape[0]
        x *= shape[1] ; y *= shape[0]

    X = range(shape[1]) ; Y = range(shape[0])
    X, Y = tl.meshgrid(X , Y)
    
    f = ((((X-x)*tl.cos(angle) - (Y-y)*tl.sin(angle))/a)**2 + (((Y-y)*tl.cos(angle) + (X-x)*tl.sin(angle))/b)**2 < 1)*1
    
    return f

def Gauss( sx , sy , shape , x = 0 , y = 0 , angle = 0 , use_pxc = False ):
    """
    2D Gaussian distribution signal.
    (x,y) coordinates are given in a centered cartesian frame.

    ========Input=========
    
    sx :        Gaussian RMS width
    sy :        Gaussian RMS height
    shape:      Output array shape
    x :         Horizontal center of ellipse (Default for horizontally centered)
    y :         Vertical center of ellipse (Default for vertically centered)
    angle :     Angle defining the width direction (Default for horizontal direction)
    use_pxc :   Use pixel as unit of measurement (Default for shape as unit of measurement)
    
    ========Output========
    
    f : 2D float array

    """
    
    x += 0.5 ; y += 0.5
    
    if( use_pxc == False ):
        sx *= shape[1] ; sy *= shape[0]
        x *= shape[1] ; y *= shape[0]

    X = range(shape[1]) ; Y = range(shape[0])
    X, Y = tl.meshgrid(X , Y)
    
    f = tl.exp( - 0.5 * ( ( (X-x)*tl.cos(angle) - (Y-y)*tl.sin(angle) ) / sx ) ** 2 - 0.5 * ( ( (Y-y)*tl.cos(angle) + (X-x)*tl.sin(angle) ) / sy ) ** 2 )
    return f

#%% Manipulation
 
def Trim( f , shape , x , y ):
    """
    Trims a 2D input array f. The trim is centered at (x,y) with shape (shape).
    (x,y) coordinates are given in a centered cartesian frame.

    ========Input=========
    
    f : 2D input array from which the trim is taken
    g : 2D array defining the trim shape
    x : x cartesian coordinate of the trimming center measured from input array's center
    y : y cartesian coordinate of the trimming center measured from input array's center
    
    ========Raises========
    
    ValueError : If the trimming area is outside f
    TypeError : If either f or g isn't a 2D array
    
    ========Output========
    
    h : Trimmed 2D array

    """
    
    if( len(tl.shape(f)) != 2 or len(shape) != 2 ):
        raise TypeError('Input and embedded arrays must be 2D')
        
    (m,n) = tl.shape(f)
    (mm,nn) = shape
    
    if( mm>m or nn>n ):
        raise ValueError('Trimmed array must be smaller than input array')
        
    h = f[tl.int32(tl.floor((m-mm)*0.5)+y):tl.int32(tl.floor((m+mm)*0.5)+y),tl.int32(tl.floor((n-nn)*0.5)+x):tl.int32(tl.floor((m+nn)*0.5)+x)]
    
    return h

def Embed( f , g , x , y ):
    """
    Embeds g into f centered at (x,y).
    (x,y) coordinates are given in a cartesian frame centered in f.

    ========Input=========
    
    f : 2D input array
    g : 2D array to be embedded
    x : x cartesian coordinate of the embedding center measured from input array's center
    y : y cartesian coordinate of the embedding center measured from input array's center
    
    ========Raises========
    
    ValueError : If the embedded array is outside f
    ValueError : If either f or g isn't a 2D array
    
    ========Output========
    
    h : 2D input array f with g embedded at (x,y)

    """
    
    if( (tl.ndim(f) , tl.ndim(g)) != (2 , 2) ):
        raise ValueError('Input and embedded arrays must be 2D')
    
    (m,n) = tl.shape(f)
    (mm,nn) = tl.shape(g)
    
    if( mm>m or nn>n ):
        raise ValueError('Embedded array must be smaller than input array')
    
    h = tl.copy(f)
    h[tl.int32(tl.floor((m-mm)*0.5)+y):tl.int32(tl.floor((m+mm)*0.5)+y),tl.int32(tl.floor((n-nn)*0.5)+x):tl.int32(tl.floor((m+nn)*0.5)+x)]=g
    
    return h

def Resize( f , shape ):
    """
    Resizes input array into input shape

    ========Input=========
    
    f : Input 2D array
    shape : New shape
    
    ========Raises========
    
    ValueError : If the trimming area is outside f
    TypeError : If either f or g isn't a 2D array
    
    ========Output========
    
    g : Resized f 2D array

    """
    
    f = LinearNormal(f)
    f = tl.Image.fromarray(f*255)
    g = f.resize(shape)
    g = tl.array( g , dtype=float)
    g = LinearNormal(g)

    return g

#%% Normalization

def LinearNormal( f , new_min = 0. , new_max = 1. ):
    """
    Linearly normalizes input array to interval [new_min,new_max].
    If the input array is complex, only affects its norm.
    Default values for [0,1] normalization.
    
    ========Input=========
    
    f : Input array
    new_min : Minimum value in normalization interval (default 0)
    new_max : Maximum value in normalization interval (default 1)
    
    ========Output========
    
    g : Normalized input array
    
    """

    if tl.iscomplex(f).any():
        fnorm = tl.abss(f)
        if( (fnorm == tl.mean(fnorm)).all() ):
            g = 1.
        else:
            g = ( fnorm - tl.minn(fnorm) )/( tl.maxx(fnorm) - tl.minn(fnorm))*(new_max-new_min) + new_min
        g *= tl.exp(1j*tl.angle(f))
        
    else:
        if( (f == tl.mean(f)).all() ):
            g = tl.ones(tl.shape(f))
        else:
            g = ( f - tl.minn(f) )/( tl.maxx(f) - tl.minn(f) )*(new_max-new_min) + new_min
        
    return g

def QuadNormal( f , grid = None ):
    """
    Normalizes input array as in Hilbert spaces.
    f and grid must have the same shape.

    ========Input=========
    
    f :     Input 2D array
    grid :  Function domain 2D array (default for square grid of unit squares)
    
    ========Output========
    
    g : Normalized input array
    
    """
    
    shape = tl.shape(f)
    
    if( grid == None ):
        x = range(shape[1]) ; y = range(shape[0])
    elif( tl.shape(grid) == tl.shape(f) ):
        x = grid[0] ; y = grid[:,0]
    else:
        raise ValueError("f and grid must have the same shape")
    
    N2 = tl.simps(tl.simps( tl.abss(f)**2 , y  ) , x )
    g = f/tl.sqrt(N2)
    
    return g

def SigmoidNormal( f , new_min = 0. , new_max = 1. ):
    """
    Normalizes input array to interval [new_min,new_max] using a sigmoid function.
    If the input array is complex, only affects its norm.
    Default values for [0,1] normalization.
    
    ========Input=========
    
    f : Input array
    new_min : Minimum value in normalization interval (default 0)
    new_max : Maximum value in normalization interval (default 1)
    
    ========Output========
    
    g : Normalized input array
    
    """
    
    if tl.iscomplex(f).any():
        g = 1/(1 + tl.exp(-(abs(f) - tl.mean(abs(f)))/(tl.maxx(abs(f)) - tl.minn(abs(f)))) )*(new_max-new_min) + new_min
        g *= tl.exp(1j*tl.angle(f))
        
    else:
        g = 1/(1 + tl.exp(-(f - tl.mean(f))/(tl.maxx(f) - tl.minn(f))) )*(new_max-new_min) + new_min
        
    return g

#%% Metrics

def CC( f , g ):
    """
    Calculates correlation coefficient between two arrays.
    Works as corr2 MATLAB function.
    
    ========input=========
    
    f : Input array 1
    g : Input array 2
    
    ========output========
    
    cc : Correlation coefficient
    
    """

    f = tl.array(f)
    g = tl.array(g)
    
    cr = tl.summ( abs( (f - tl.mean(f)) * (g - tl.mean(g)) )  )
    cc = cr / tl.sqrt( ( tl.summ( abs(f - tl.mean(f))**2 ) ) * ( tl.summ( abs(g - tl.mean(g))**2 ) ) )
    return cc

def PSNR( f , g ):
    """
    Calculates peak signal-to-noise ratio of an array and its noisy approximation
    
    ========input=========
    
    f : Input noise-free array
    g : Input noisy approximation array
    
    ========output========
    
    psnr : PSNR
    
    """
    
    f = tl.array(f)
    g = tl.array(g)
    
    if( tl.shape(f) != tl.shape(g) ):
        raise TypeError('f and g must be arrays of equal shape')
        
    if( (f==g).all() ):
        psnr = tl.inf
    else:
        mse = tl.summ( abs(f - g)**2 )/tl.prod(tl.shape(f))
        psnr = 10*tl.log10( tl.maxx(f)**2 / mse)
    
    return psnr

def Entropy(f , nbits = 0):
    """
    Calculates entropy of array histogram
    
    ========input=========
    
    f : Input array
    nbits : Calculate for a given number of bits
    
    ========output========
    
    S : Shannon Entropy
    
    """

    f = tl.reshape(f, (1,tl.prod(tl.shape(f))) )[0]
    
    if(nbits!=0):
        f = LinearNormal(f)
        MAX = 2.**nbits - 1
        f = tl.floor(f*MAX)
    
    pdf = tl.Series(f).value_counts().values
    
    pdf = pdf/tl.summ(pdf*1.)
    
    S = -tl.summ(pdf*tl.log2(pdf))
    
    return S

#%% Binarization

def ErrDiffBin( f , T = 0.5 , b = [[1/16., 5/16., 3/16.],[7/16., 0., 0.],[0., 0., 0.]] ):
    """
    Binarizes input 2D array using the Error-Diffusion algorithm.
    Barnard E. Optimal error diffusion for computer-generated holograms. J Opt Soc Am A 1988;5:1803-17.
    
    ========Input=========
    
    f : Real input 2D array 
    T : Binarization threshold (default value for input array with [0,1] dynamic range)
    b : 3x3 2D real array with the diffusion coefficients (default given by paper; see paper for details)
    
    ========Raises========
    
    TypeError : If b isn't a 3x3 2D real array
    ValueError : If b doesn't meet the convergence conditions (see paper for details)
    
    ========Output========
    
    h[1:-1,1:-1] : Binarized input array

    """
    
    b = tl.array(b)
    
    if( tl.shape(b) != (3,3) or tl.isreal(b).all() == False ):
        raise TypeError('b has to be a 3x3 2D array')
    
    if( tl.summ(abs(b)) > 1 ):
        raise ValueError('The diffusion coefficients don\'t meet the convergence conditions')
        
    
    n , m = f.shape
    g = tl.zeros((n + 2 , m + 2))
    h = tl.zeros((n + 2 , m + 2))
    s = tl.zeros((n + 2 , m + 2))
    
    g[1:-1 , 1:-1] = f
    
    for i in range(1 , n - 1):
        for j in range( 1 , m-1):
            
            g[i,j] += tl.summ( b * s[i-1 : i+2 , j-1 : j+2] )
            h[i,j] = (g[i,j] > T)*1.
            s[i,j] = g[i,j] - h[i,j]
            
    return h[1:-1,1:-1]

def AdaptativeThresholding( f , s , T = 1 ):
    """
    Binarizes input 2D array using the adaptative thresholding algorithm via integral images.
    
    ========Input=========
    
    f : Real input 2D array
    s : Sub-array size in pixels for mean calculation
    T : Percentage to be taken from mean for thresholding
    
   ========Output========
    
    g : Binarized input array

    """

    n, m = f.shape
    I = tl.zeros((n,m),dtype='float')
    g = tl.zeros((n,m),dtype='float')
    
    for j in range(m):
        S = 0
        for i in range(n):
            S += f[i,j]
            I[i,j] = S
            if j > 0:
                I[i,j] += I[i, j-1]
                
    for j in range(m):
        x1 = int(tl.floor(j - s/2));
        if( x1 < 0 ): x1 = 0;
        x2 = int(tl.floor(j + s/2));
        if( x2 > m-1 ): x2 = m-1;
        for i in range(n):
            y1 = int(tl.floor(i - s/2));
            if( y1 < 0 ): y1 = 0;
            y2 = int(tl.floor(i + s/2));
            if( y2 > n-1 ): y2 = n-1;
            
            A = (x2-x1)*(y2-y1)
            S = I[y2,x2] - I[y1-1,x2] - I[y2,x1-1] + I[y1-1,x1-1]
            if( f[i,j]*A > S*T ):
                g[i,j] = 1
    
    return g
                

def QZP( f , Z = 2 , s = 1):
    """
    Step in stepwise quantization of input array into Z levels.
    Default values for thresholding.
    Wyrowski F. Iterative quantization of digital amplitude holograms. Appl Opt 1989;28:3864-70.
    
    ========Input=========
    
    f : Real input array with [0,1] dynamic range
    Z : Number of quantization levels
    s : Epsilon for the current step (default value for last step; see paper for details)
    
    ========Output========
    
    g : Quantized input array
    
    """
    
    Z = int(Z)
    
    if( Z==2 and s==1 ): g = ( f > 0.5 )*1. # Simple thresholding case 
        
    else:
        D = 0.5 / ( Z - 1 )
        g = tl.copy(f)
        for z in range(Z): g[ ( ( 2*z - s )*D  < f ) * ( f <= ( 2*z + s )*D ) ] = z
        g *= 2*D
        
    return g

#%% Bidimensional Fourier Analysis

def FFT2( f ):
    """
    Fast bidimensional Fourier transforms input array and shifts zero-frequency to center.
    
    ========Input=========
    
    f : Input 2D array

    ========output========
    
    h : Shifted and fourier transformed f
    
    """
    g = tl.ifftshift( tl.fft2( tl.fftshift( f ) ) )
    
    return g

def IFFT2( f ):
    """
    Inverse fast bidimensional Fourier transforms input array and shifts zero-frequency to center.
    
    ========Input=========
    
    f : Input 2D array

    ========output========
    
    h : Shifted and inverse Fourier transformed f
    
    """
    g = tl.ifftshift( tl.ifft2( tl.fftshift( f ) ) )
    
    return g

def LowPass( f , px = 0.5 , py = 0.5 , mode = 'rect' ):
    """
    Low pass filter.
    
    ========Input=========
    
    f :     Input 2D array
    px :    Defines the characteristic horizontal length of filter function
    py :    Defines the characteristic vertical length of filter function
    mode:   Defines the filter function

    ========output========
    
    g : Filterted signal
    
    """
    
    shape = tl.shape(f)
    
    filters = {
            'rect':Rect( px , py , shape ),
            'ellipse':Ellipse( px , py , shape ),
            'gauss':Gauss( px , py , shape )}
    
    F = FFT2( f )
    F *= filters.get(mode)
    g = IFFT2(F)

    return g

def HighPass( f , px = 0.5 , py = 0.5 , mode = 'rect' ):
    """
    High pass filter.
    
    ========Input=========
    
    f :     Input 2D array
    px :    Defines the characteristic horizontal length of filter function
    py :    Defines the characteristic vertical length of filter function
    mode:   Defines the filter function

    ========output========
    
    g : Filterted signal
    
    """
    
    shape = tl.shape(f)
    
    filters = {
            'rect':Rect( px , py , shape ),
            'ellipse':Ellipse( px , py , shape ),
            'gauss':Gauss( px , py , shape )}
    
    F = FFT2( f )
    F *= (1 - filters.get(mode))
    g = IFFT2(F)

    return g
    
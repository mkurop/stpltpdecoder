# -*- coding: utf-8

__author__ = 'markurop'

"""Module for computation of STP and LTP parameters. Provided decoder (encoder + decoder) works for any specified
sampling frequency.

Created 31.03.2014 """

import numpy as np
import collections as col
import audiolazy as al
import spectrum
from scipy.io import wavfile
import copy
from scipy.signal import deconvolve

def poly2lsf(a):
    """Prediction polynomial to line spectral frequencies.
    converts the prediction polynomial specified by A,
    into the corresponding line spectral frequencies, LSF.
    normalizes the prediction polynomial by A(1).
    """

    #Line spectral frequencies are not defined for complex polynomials.

    # Normalize the polynomial

    a = np.array(a)
    if a[0] != 1:
        a/=a[0]

    if max(np.abs(np.roots(a))) >= 1.0:
        error('The polynomial must have all roots inside of the unit circle.');


    # Form the sum and differnce filters

    p  = len(a)-1   # The leading one in the polynomial is not used
    a1 = np.concatenate((a, np.array([0])))
    a2 = a1[-1::-1]
    P1 = a1 - a2        # Difference filter
    Q1 = a1 + a2        # Sum Filter

    # If order is even, remove the known root at z = 1 for P1 and z = -1 for Q1
    # If odd, remove both the roots from P1

    if p%2: # Odd order
        P, r = deconvolve(P1,[1, 0 ,-1])
        Q = Q1
    else:          # Even order
        P, r = deconvolve(P1, [1, -1])
        Q, r = deconvolve(Q1, [1,  1])

    rP  = np.roots(P)
    rQ  = np.roots(Q)

    aP  = np.angle(rP[1::2])
    aQ  = np.angle(rQ[1::2])

    lsf = sorted(np.concatenate((-aP,-aQ)))

    return lsf

def xcorr(x, y=None, maxlags=None, norm='biased'):
    """Cross-correlation using np.correlate
    Estimates the cross-correlation (and autocorrelation) sequence of a random
    process of length N. By default, there is no normalisation and the output
    sequence of the cross-correlation has a length 2*N+1.

    :param array x: first data array of length N
    :param array y: second data array of length N. If not specified, computes the
        autocorrelation.
    :param int maxlags: compute cross correlation between [-maxlags:maxlags]
        when maxlags is not specified, the range of lags is [-N+1:N-1].
    :param str option: normalisation in ['biased', 'unbiased', None, 'coeff']

    The true cross-correlation sequence is
    .. math:: r_{xy}[m] = E(x[n+m].y^*[n]) = E(x[n].y^*[n-m])
    
    However, in practice, only a finite segment of one realization of the
    infinite-length random process is available.

    The correlation is estimated using np.correlate(x,y,'full').

    Normalisation is handled by this function using the following cases:
        * 'biased': Biased estimate of the cross-correlation function
        * 'unbiased': Unbiased estimate of the cross-correlation function
        * 'coeff': Normalizes the sequence so the autocorrelations at zero
           lag is 1.0.

    :return:
        * a np.array containing the cross-correlation sequence (length 2*N-1)
        * lags vector

    .. note:: If x and y are not the same length, the shorter vector is
        zero-padded to the length of the longer vector.
    """

    N = len(x)
    if y is None:
        y = x
    assert len(x) == len(y), 'x and y must have the same length. Add zeros if needed'

    if maxlags is None:
        maxlags = N-1
        lags = np.arange(0, 2*N-1)
    else:
        assert maxlags <= N, 'maxlags must be less than data length'
        lags = np.arange(N-maxlags-1, N+maxlags)

    res = np.correlate(x, y, mode='full')

    if norm == 'biased':
        Nf = float(N)
        res = res[lags] / float(N)    # do not use /= !!
    elif norm == 'unbiased':
        res = res[lags] / (float(N)-abs(np.arange(-N+1, N)))[lags]
    elif norm == 'coeff':
        Nf = float(N)
        rms = pylab_rms_flat(x) * pylab_rms_flat(y)
        res = res[lags] / rms / Nf
    else:
        res = res[lags]

    lags = np.arange(-maxlags, maxlags+1)
    return res, lags
def levinson(r, order=None, allow_singularity=False):

    r"""Levinson-Durbin recursion.
    Find the coefficients of a length(r)-1 order autoregressive linear process

    :param r: autocorrelation sequence of length N + 1 (first element being the zero-lag autocorrelation)
    :param order: requested order of the autoregressive coefficients. default is N.
    :param allow_singularity: false by default. Other implementations may be True (e.g., octave)
    :return:
        * the `N+1` autoregressive coefficients :math:`A=(1, a_1...a_N)`
        * the prediction errors
        * the `N` reflections coefficients values

    This algorithm solves the set of complex linear simultaneous equations
    using Levinson algorithm.

    .. math::
        \bold{T}_M \left( \begin{array}{c} 1 \\ \bold{a}_M \end{array} \right) =
        \left( \begin{array}{c} \rho_M \\ \bold{0}_M  \end{array} \right)
    where :math:`\bold{T}_M` is a Hermitian Toeplitz matrix with elements
    
    :math:`T_0, T_1, \dots ,T_M`.
    .. note:: Solving this equations by Gaussian elimination would
        require :math:`M^3` operations whereas the levinson algorithm
        requires :math:`M^2+M` additions and :math:`M^2+M` multiplications.
    This is equivalent to solve the following symmetric Toeplitz system of
    linear equations
    .. math::
        \left( \begin{array}{cccc}
        r_1 & r_2^* & \dots & r_{n}^*\\
        r_2 & r_1^* & \dots & r_{n-1}^*\\
        \dots & \dots & \dots & \dots\\
        r_n & \dots & r_2 & r_1 \end{array} \right)
        \left( \begin{array}{cccc}
        a_2\\
        a_3 \\
        \dots \\
        a_{N+1}  \end{array} \right)
        =
        \left( \begin{array}{cccc}
        -r_2\\
        -r_3 \\
        \dots \\
        -r_{N+1}  \end{array} \right)

    where :math:`r = (r_1  ... r_{N+1})` is the input autocorrelation vector, and
    :math:`r_i^*` denotes the complex conjugate of :math:`r_i`. The input r is typically
    a vector of autocorrelation coefficients where lag 0 is the first
    element :math:`r_1`.

    """
    #from np import isrealobj
    T0  = np.real(r[0])
    T = r[1:]
    M = len(T)

    if order is None:
        M = len(T)
    else:
        assert order <= M, 'order must be less than size of the input data'
        M = order

    realdata = np.isrealobj(r)
    if realdata is True:
        A = np.zeros(M, dtype=float)
        ref = np.zeros(M, dtype=float)
    else:
        A = np.zeros(M, dtype=complex)
        ref = np.zeros(M, dtype=complex)

    P = T0

    for k in range(0, M):
        save = T[k]
        if k == 0:
            temp = -save / P
        else:
            #save += sum([A[j]*T[k-j-1] for j in range(0,k)])
            for j in range(0, k):
                save = save + A[j] * T[k-j-1]
            temp = -save / P
        if realdata:
            P = P * (1. - temp**2.)
        else:
            P = P * (1. - (temp.real**2+temp.imag**2))
        if P <= 0 and allow_singularity==False:
            raise ValueError("singular matrix")
        A[k] = temp
        ref[k] = temp # save reflection coeff at each step
        if k == 0:
            continue

        khalf = (k+1)//2
        if realdata is True:
            for j in range(0, khalf):
                kj = k-j-1
                save = A[j]
                A[j] = save + temp * A[kj]
                if j != kj:
                    A[kj] += temp*save
        else:
            for j in range(0, khalf):
                kj = k-j-1
                save = A[j]
                A[j] = save + temp * A[kj].conjugate()
                if j != kj:
                    A[kj] = A[kj] + temp * save.conjugate()

    return A, P, ref
def lsf2poly(lsf):
    """Convert line spectral frequencies to prediction filter coefficients

    returns a vector a containing the prediction filter coefficients from a vector lsf of line spectral frequencies.

    .. doctest::

        >>> lsf = [0.7842 ,   1.5605  ,  1.8776 ,   1.8984,    2.3593]
        >>> a = lsf2poly(lsf)
        array([  1.00000000e+00,   6.14837835e-01,   9.89884967e-01,
            9.31594056e-05,   3.13713832e-03,  -8.12002261e-03 ])

    .. seealso:: poly2lsf, rc2poly, ac2poly, rc2is
    """
    #   Reference: A.M. Kondoz, "Digital Speech: Coding for Low Bit Rate Communications
    #   Systems" John Wiley & Sons 1994 ,Chapter 4

    # Line spectral frequencies must be real.

    lsf = np.array(lsf)

    if max(lsf) > np.pi or min(lsf) < 0:
        raise ValueError('Line spectral frequencies must be between 0 and pi.')

    p = len(lsf)  # model order

    # Form zeros using the LSFs and unit amplitudes
    z = np.exp(1.j * lsf)

    # Separate the zeros to those belonging to P and Q
    rQ = z[0::2]
    rP = z[1::2]

    # Include the conjugates as well
    rQ = np.concatenate((rQ, rQ.conjugate()))
    rP = np.concatenate((rP, rP.conjugate()))

    # Form the polynomials P and Q, note that these should be real
    Q = np.poly(rQ);
    P = np.poly(rP);

    # Form the sum and difference filters by including known roots at z = 1 and
    # z = -1

    if p % 2:
        # Odd order: z = +1 and z = -1 are roots of the difference filter, P1(z)
        P1 = np.convolve(P, [1, 0, -1])
        Q1 = Q
    else:
        # Even order: z = -1 is a root of the sum filter, Q1(z) and z = 1 is a
        # root of the difference filter, P1(z)
        P1 = np.convolve(P, [1, -1])
        Q1 = np.convolve(Q, [1, 1])

    # Prediction polynomial is formed by averaging P1 and Q1

    a = .5 * (P1 + Q1)
    return a[0:-1:1]  # do not return last element


def InitEncStat(p, Frame):
    """ Function for initialization of the encoder
    Input:
    p - AR model order
    Frame - frame length in samples (typically 160-200 for 8kHz sampling rate and twice of that for 16kHz sampling
    frequency
    Output:
    CodStat - named tuple with fields:
        LsfQ - quantized initial LSF parameters
        Lsf - LSF parameters
        hlp - initial conditions for AR analysis filter
        ShortTermRes - array of initial short term residual samples
    """
    EncStat = col.namedtuple('EncStat', 'LsfQ, Lsf, hlp, ShortTermRes')
    EncStat = EncStat(LsfQ=np.linspace(0.2, np.pi - 0.1, p), Lsf=np.linspace(0.2, np.pi - 0.1, p), hlp=np.zeros((p,)), \
                      ShortTermRes=np.zeros((Frame,)))
    return EncStat

def InitEncStatFromData(p, Frame, LsfQ, Lsf, hlp, ShortTermRes):
    """ Function for initialization of the encoder
    Input:
    p - AR model order
    Frame - frame length in samples (typically 160-200 for 8kHz sampling rate and twice of that for 16kHz sampling
    frequency
    Output:
    CodStat - named tuple with fields:
        LsfQ - quantized initial LSF parameters
        Lsf - LSF parameters
        hlp - initial conditions for AR analysis filter
        ShortTermRes - array of initial short term residual samples
    """
    EncStat = col.namedtuple('EncStat', 'LsfQ, Lsf, hlp, ShortTermRes')
    EncStat = EncStat(LsfQ=LsfQ, Lsf=Lsf, hlp=hlp, \
                      ShortTermRes=ShortTermRes)
    return EncStat

def lpanafil(s,a,hlp):
    """ Function for the STP analysis.
    Input:
    s - signal frame
    a - STP parameters [1, a1 , ..., a_p]
    hlp - history of the analysis filter
    Output
    res - STP residual
    hlp - history for the next signal frame
    """
    s = np.concatenate([hlp,s])
    p = len(a)-1
    res = np.convolve(s,a,mode='full')
    res = res[p:-p]
    hlp = s[-p:]
    return res, hlp


def Encoder(fs, EncStat, p, LpcFrame, SubFrames, SubFramesOL, SmplRate):
    """
    Function for performing the STP & LTP analysis.

    Input

    fs - frame to encode
    EncStat - coder state
    p - order of the lpc analysis
    LpcFrame - length of the lpc frame
    SubFrames - number of lpc subframes
    SubFramesOL - number of the pitch subframes
    SmplRate - sampling rate of the input signal in Hz

    Output

    Par - structure with the following fields
    Lsf - current unquantized lsfs
    LsfQ - current quantized lsfs
    LtpTap - length SubFramesOL vector with the LTP taps
    LtpLag - length SubFramesOL vector with the LTP lags
    LtpVar - length SubFramesOL vector with the LTP variances

    EncStat - structure with the following fields#
    LsfQ - quantized previous frame lsfs
    hlp - history of the lpc filter
    ShortTermRes - previous frame of the short term residual

    ShortTermRes - short term residual

    LongTermRes - long term residual energy normalized


    Author: Marcin Kuropatwinski
    Created 17.02.2007 Last modification 17.02.2007 Python port 13.11.2013
    """


    #declare output structures
    CodStatOut = col.namedtuple('CodStatOut', 'LsfQ, Lsf, hlp, ShortTermRes')
    Par = col.namedtuple('Par', 'LsfQ, Lsf, LtpTap, LtpLag, LtpVar, StpVar')

    #compute frame length
    Frame = len(fs)
    fl = Frame
    #length of the subframe
    sfl = int(Frame / SubFrames)

    #length of the open loop subframe
    sflp = int(Frame / SubFramesOL)

    g1 = 0.994
    gk1 = np.cumprod(np.ones((p,)) * g1)
    Lmin = int(20 * SmplRate / 8000)  #minimal pitch
    Lmax = int(120 * SmplRate / 8000)  #maximal pitch
    resl = np.zeros((Frame,))
    hlp = EncStat.hlp

    #memory for the ltp parameters
    LtpTap=np.zeros((SubFramesOL,))
    LtpLag=np.zeros((SubFramesOL,))
    LtpVar=np.zeros((SubFramesOL,))

    #get frame for computing the lpc
    fs_lpc = fs[Frame - LpcFrame:]

    #compute current frame lpcs
    #  asd = al.lpc(fs_lpc, p)
    x,_ = xcorr(fs_lpc, maxlags=p)
    asd, stp_excitation_variance, reflection_coefficient = levinson(x[p:])
    asd = np.concatenate(([1.],asd))

    print(f"STP: {asd}")

    #  asd.numerator[1:] = asd.numerator[1:] * gk1
    print(gk1)
    print(asd)
    asd[1:] *= gk1

    #convert to lsf
    #  if al.lsf_stable(asd):
        #  asl = al.lsf(asd)[p + 1:-1]
    #  else:
        #  asl = EncStat.Lsf
    asl = poly2lsf(asd)
    
    #memorize lsfs in the parameters structure
    Lsf = asl

    #get previous frame quantized lsfs
    aslpq = EncStat.LsfQ

    #quantization
    aslq = asl  #quantize current frame lsfs - no quantization

    #memorize current frame quantized lsfs
    LsfQ = aslq

    #auxiliary parameter
    ic = 1. / SubFrames

    #allocate memory for res
    res = np.zeros((fl,))
    a = np.zeros((SubFrames, p + 1))

    aslpq = np.array(aslpq, ndmin=1)
    aslq = np.array(aslq, ndmin=1)

    for i in range(SubFrames):
        asli = (1 - (i + 1) * ic) * aslpq + (i + 1) * ic * aslq
        print(asli)
        a = lsf2poly(asli)
        res[i * sfl:(i + 1) * sfl], hlp = lpanafil(fs[i * sfl:(i + 1) * sfl], a, hlp)

    StpVar = np.sum(np.power(res,2))/Frame

    swp = EncStat.ShortTermRes
    ShortTermRes = res
    hlp = hlp

    #pitch analysis
    swpitch = np.concatenate([swp, res],axis=0)

    pcorr = np.zeros((Lmax - Lmin + 1,))
    stp_vars = np.zeros((SubFramesOL,))
    for i in range(SubFramesOL):
        # compute correlations for each sample in the subframe
        a_aux = res[i * sflp:(i + 1) * sflp]

        max_ = -np.Inf
        for k in range(Lmin, Lmax + 1):
            b_aux = swpitch[fl + i * sflp - k:fl + (i + 1) * sflp - k]
            pcorr[k - Lmin] = np.sum(a_aux * b_aux)
            merit = pcorr[k-Lmin]*pcorr[k-Lmin]/np.sum(b_aux*b_aux)
            if  merit > max_:
                max_ = merit
                t1 = k - Lmin

        print(max_)


        stp_vars[i] = np.sum(a_aux * a_aux) / sflp

        #find maximum
        O1 = pcorr[t1]

        #compute weighting factors
        k1 = t1 + Lmin
        S1 = np.sum(
            swpitch[fl + i * sflp - k1:fl + (i + 1) * sflp - k1] * swpitch[fl + i * sflp - k1:fl + (i + 1) * sflp - k1])

        if not S1 == 0:
            M1 = O1 / np.sqrt(S1)
        else:
            M1 = O1

        #weigth
        Top = k1
        Mop = M1
        S = S1
        t = t1

        LtpLag[i] = int(Top)
        #compute ltp taps
        if S < np.spacing(1):
            LtpTap[i] = 0.99
        else:
            LtpTap[i] = -pcorr[t] / S
            #compute variance of the long term residual

        LtpVar[i] = stp_vars[i] + LtpTap[i] * pcorr[t] / sflp
    Par.StpVar = np.sum(stp_vars) / len(stp_vars)
    #compute long term residual

    for i in range(SubFramesOL):
        if LtpVar[i] > 10e-30:

            resl[i * sflp:(i + 1) * sflp] = (res[i * sflp:(i + 1) * sflp] + \
                                             LtpTap[i] * swpitch[
                                                 fl + i * sflp - int(LtpLag[i]):fl + (i + 1) * sflp - int(LtpLag[ \
                                                                 i])]) / np.sqrt(LtpVar[i])
        else:
            resl[i * sflp:(i + 1) * sflp] = (res[i * sflp:(i + 1) * sflp] + \
                                             LtpTap[i] * swpitch[
                                                             fl + i * sflp - int(LtpLag[i]):fl + (i + 1) * sflp - int(LtpLag[
                                                                 i])])

    #store output
    LongTermRes = resl

    #form output (named) tuples
    CodStatOut = col.namedtuple('CodStatOut', 'LsfQ, Lsf, hlp, ShortTermRes')
    CodStatOut = CodStatOut(LsfQ=LsfQ,Lsf = Lsf, hlp = hlp, ShortTermRes = ShortTermRes)
    Par = col.namedtuple('Par', 'LsfQ, Lsf, LtpTap, LtpLag, LtpVar, StpVar')
    Par = Par(LsfQ = LsfQ, Lsf=Lsf, LtpTap=LtpTap, LtpLag = LtpLag, LtpVar=LtpVar, StpVar=StpVar)

    #return output
    return CodStatOut, Par, LongTermRes

def InitDecStat(p,Frame):
    """Function initializing the decoder
    Input:
    p - AR model order
    Frame - length of the frame
    Author: Marcin Kuropatwiński
    Created: 31.03.2014
    """
    DecStat = col.namedtuple('DecStat','LsfQ, hlp, ShortTermRes')
    DecStat = DecStat(LsfQ = np.linspace(0.2,np.pi-0.1,p), hlp = np.zeros((p,)), ShortTermRes = np.zeros((Frame,)))
    return DecStat


def Decoder(LongTermRes,Par,DecStat,p,SubFrames,SubFramesOL):

    """
    Function for decoding speech based on the input from the complementary function Encoder.

    Input:
    LongTermRes - a frame of the long term residual
    Par - parameters structure
    DecState - structure of the decoder state with the following fields

    hlp - short term residual synthesis filter state
    ShortTermRes - previous frame of the short term residual

    Remaining inputs are obvious, see the Encoder function.

    Output:

    fs - current frame synthesized speech
    DecState - decoder state structure

    Author: Marcin Kuropatwinski
    Created: 31.03.2014
    """
    #declare output DecStat
    DecStatOut = col.namedtuple('DecStatOut','LsfQ, hlp, ShortTermRes')

    #length of the step
    Frame = len(LongTermRes)
    #number of subframes per frame
    n = SubFrames
    #length of the subframe
    sfl = Frame/n
    np_ = SubFramesOL # number of open loop frames
    # length of the open loop pitch subframe
    sflp = Frame/np_

    # get the current frame parameters
    LsfQ = Par.LsfQ
    LtpLag = Par.LtpLag
    LtpTap = Par.LtpTap
    LtpVar = Par.LtpVar
    hlp = copy.copy(DecStat.hlp)


    #resynthesise speech
    #initial state of the long term synthesis filter - length of 2 frames
    ssr = np.concatenate([DecStat.ShortTermRes,np.zeros((Frame,))],axis=0)

    #resynthesise frame of the short term residual
    start = 0
    for i in range(np_):
        #    print len(LtpLag), len(LtpVar), len(LtpTap)
        for j in range(i*sflp,(i+1)*sflp):
            #raw_input()
            ssr[Frame+j] = np.sqrt(LtpVar[i])*LongTermRes[start+j] - \
            LtpTap[i]*ssr[Frame+j-LtpLag[i]]

    ShortTermRes = ssr[Frame:]
    aslpq = DecStat.LsfQ
    aslq = np.array(LsfQ,ndmin=1)

    LsfQ = aslq
    ic = 1./n
    fs = np.zeros((Frame,))
    for i in range(n):
        asli = (1-(i+1)*ic)*aslpq + (i+1)*ic*aslq
        # convert back into lpc domain
        a = lsf2poly(asli)
        for j in range(i*sfl,(i+1)*sfl):
            fs[j] = -np.sum(a[1:]*hlp) + ssr[Frame+j]
            hlp[1:] = hlp[:-1]
            hlp[0] = fs[j]

    DecStatOut = DecStatOut(LsfQ = LsfQ, hlp = hlp, ShortTermRes = ShortTermRes)

    return fs, DecStatOut

if __name__ == "__main__":
    sr, s = wavfile.read("VLRecording10591_05.04.13_25.168.wav")
    s = s/2.**15
    s_out = np.zeros_like(s)
    CodStat = InitEncStat(16, 512)
    DecStat = InitDecStat(16, 512)
    start = 0
    Frame = 512
    p = 16
    while start + Frame < len(s):
        fs = s[start:start+Frame]
        CodStat, Par, LongTermRes = Encoder(fs, CodStat, p, Frame, 4, 4, 16000)
        fs_out, DecStat = Decoder(LongTermRes,Par,DecStat,p,4,4)
        s_out[start:start+Frame] = fs_out
        start = start + Frame

    wavfile.write("s_out.wav", 16000, np.int16(s_out*2.**15))


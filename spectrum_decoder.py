#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import deconvolve

__all__ = ['xcorr', 'levinson', 'lsf2poly', 'poly2lsf']

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


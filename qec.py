from functools import reduce
import operator as op
import math
from itertools import count


def error_prob(px_correct, px_incorrect, pz_correct, pz_incorrect):
    error_x = px_correct/(px_correct+px_incorrect)
    error_z = pz_correct/(pz_correct+pz_incorrect)
    print(error_x, error_z)
    error_probability = error_x + error_z
    return error_probability


def ncr(n, k):
    n = int(n)
    k = int(k)
    k = min(k, n - k)
    value = int (reduce(op.mul, range(n, n - k, -1), 1) / reduce (op.mul, range(1, k + 1), 1))
    return value


def is_prime(N):
    if N % 2 == 0 and N > 2:
        return False
    for i in range(3, int(math.sqrt(N)) + 1, 2):
        if N % i == 0:
            return False
    return True


def qpc_x_failure(p, p_complete, p_partial, eta_x, n, m):
    var_1 = 0
    for i in range (0, n + 1, 2):
        var_1 = var_1 + ncr(n, i) * ncr(i, (i / 2)) * (p_complete ** i) * (p_partial ** (n - i)) * (eta_x ** (i / 2)) * ((1 - eta_x) ** (i / 2))
    px_unknown = var_1 + 1 - (1 - ((1 - p) ** m)) ** n
    return px_unknown


def qpc_z_failure(p, epx, n, m):
    var_1 = 0
    for i in range (2, m + 1, 2):
        var_1 = var_1 + ncr (m, i) * ncr(i, (i / 2)) * (p ** i) * ((1 - p) ** (m - i)) * (epx ** (i / 2)) * (
                    (1 - epx) ** (i / 2))
    deltaz = var_1 + (1 - p) ** m
    pz_unknown = 1 - ((1 - deltaz) ** n)
    return pz_unknown, deltaz


def css_correct(N, eta, t):
    var_1 = 0
    for k in range (0, t + 1):
        var_1 = var_1 + ncr(N, k) * (eta ** k) * ((1 - eta) ** (N - k))
    return var_1


def css_incorrect(N, eta, t):
    var_1 = 0
    for k in range (t + 1, N + 1):
        var_1 = var_1 + ncr(N, k) * (eta ** k) * ((1 - eta) ** (N - k))
    return var_1


def qpc_x_correct(p_complete, p_partial, eta_x, n):
    var_1 = 0
    for i in range (1, n + 1):
        l = math.ceil ((i + 1) / 2)
        for j in range (l, i + 1):
            var_1 = var_1 + ncr(n, i) * ncr(i, j) * (p_complete ** i) * (p_partial ** (n - i)) * (eta_x ** j) * (
                        (1 - eta_x) ** (i - j))
    return var_1


def qpc_x_incorrect(p_complete, p_partial, eta_x, n):
    var_1 = 0
    for i in range (1, n + 1):
        l = math.floor ((i - 1) / 2)
        for j in range (0, l + 1):
            var_1 = var_1 + ncr (n, i) * ncr (i, j) * (p_complete ** i) * (p_partial ** (n - i)) * (eta_x ** j) * (
                        (1 - eta_x) ** (i - j))
    return var_1


def qpc_z_correct(epx, n, m, p, deltaz):
    var_1 = 0
    for j in range(1, m + 1):
        l = math.floor((j - 1) / 2)
        for k in range (0, l + 1):
            var_1 = var_1 + ncr(m, j) * ncr (j, k) * (p ** j) * ((1 - p) ** (m - j)) * (epx ** k) * (
                        (1 - epx) ** (j - k))

    pz_correct = ((1 - deltaz) ** n + (2 * var_1 + deltaz - 1) ** n) * 0.5
    return var_1, pz_correct


def qpyc_failure(N, p, k):
    var_1 = 0
    for i in range(k+1, N+1):
        var_1 = var_1 + ncr(N, i)*((1-p)**i)*(p**(N-i))
    return var_1


def qpyc_incorrect(N, p, k, ep, loss):
    var_1 = 0
    for i in range(0,k+1):
        for j in range(math.ceil(((k-i)/2)+0.5), N-i+1):
            var_1 = var_1 + ncr(N, i)*ncr(N-i, j)*(loss**i)*(p**(N-i))*(ep**j)*((1-ep)**(N-i-j))
    return var_1


def qpyc_correct(N, p, k, ep, loss):
    var_1 = 0
    for i in range(0, k+1):
        for j in range(0,math.floor((k-i)/2)):
            var_1 = var_1 + ncr(N, i)*ncr(N-i, j)*(loss**i)*(p**(N-i))*(ep**j)*((1-ep)**(N-i-j))
    return var_1


def qrsc_failure(N, k, loss):
    var_1 = 0
    for i in range(N-k+1, N+1):
        var_1 = var_1 + ncr(N, i)*(loss**i)*((1-loss)**(N-i))
    return var_1


def qrsc_incorrect(N, p, k, ep, loss):
    var_1 = 0
    for i in range(0,N-k+1):
        for j in range(math.ceil(((N-k-i)/2)+0.5), N-k-i+1):
            var_1 = var_1 + ncr(N, i)*ncr(N-i, j)*(loss**i)*(p**(N-i))*(ep**j)*((1-ep)**(N-i-j))
    return var_1


def qrsc_correct(N, p, k, ep, loss):
    var_1 = 0
    for i in range(0,N-k+1):
        for j in range(0, math.floor((N-k-i)/2)):
            var_1 = var_1 + ncr(N, i)*ncr(N-i, j)*(loss**i)*(p**(N-i))*(ep**j)*((1-ep)**(N-i-j))
    return var_1


# [[N,k,2t+1]]_q code
def css(epg, epm, loss, M, nEG, N=7, t=1):
    p = 1 - loss
    fo = 1 - (5/4)*epg
    eta = 2*epm + epg + (2/3) * (1-fo)

    px_correct = css_correct(N, eta, t)

    px_incorrect = css_incorrect(N, eta, t)

    pz_correct = px_correct

    pz_incorrect = px_incorrect

    p_success = 1 - ((1 - p)**(M*nEG))

    return px_correct, px_incorrect, pz_correct, pz_incorrect, p_success


# (n,m) quantum parity codes
def qpc(epg, epm, loss, n, m):
    p = 1 - loss
    epx = ((epg / 2) + epm) * p
    epz = ((epg / 2) + epm) * p
    p_complete = p ** m
    p_partial = 1 - p ** m - (1 - p) ** m
    eta_x = 0.5 * (((1 - (2 * epz)) ** m) + 1)

    px_unknown = qpc_x_failure (p, p_complete, p_partial, eta_x, n, m)

    pz_unknown, deltaz = qpc_z_failure (p, epx, n, m)

    p_success = (1 - px_unknown) * (1 - pz_unknown)

    px_correct = qpc_x_correct(p_complete, p_partial, eta_x, n)

    px_incorrect = qpc_x_incorrect(p_complete, p_partial, eta_x, n)

    eta_z, pz_correct = qpc_z_correct(epx, n, m, p, deltaz)

    pz_incorrect = ((1 - deltaz) ** n - (2 * eta_z + deltaz - 1) ** n) * 0.5

    return px_correct, px_incorrect, pz_correct, pz_incorrect, p_success


# [[2k+1,1,k+1]]_q code
def qpyc(epg, epd, loss, N):

    p = 1-loss
    k = int((N-1)/2)

    if is_prime(N):
        q = N
    else:
        q = next(filter(is_prime, count(N)))

    epz = ((epg/(q**4))*((q**4)-(q**3))) + ((4*epd*((q**2)-q))/(q**2))
    epx = ((epg/(q**4))*((q**4)-(q**3))) + ((4*epd*((q**2)-q))/(q**2))

    if epx>1:
        epx = 1
    if epz>1:
        epz = 1

    p_success = 1-qpyc_failure(N, p, k)

    px_incorrect = qpyc_incorrect(N, p, k, epx, loss)

    px_correct = qpyc_correct(N, p, k, epx, loss)

    pz_incorrect = qpyc_incorrect(N, p, k, epz, loss)

    pz_correct = qpyc_correct(N, p, k, epz, loss)

    error_probability = error_prob(px_correct, px_incorrect, pz_correct, pz_incorrect)

    return px_correct, px_incorrect, pz_correct, pz_incorrect, p_success


# [[d,2k-d,d-k+1]]_d code
def qrsc(epg, epd, loss, N, k):

    p = 1-loss

    if k<(N+1)/2:
        print('Code for the given parameters does not exist')
        exit()

    if is_prime(N) == False:
        print('QRSC require N to be a prime number')
    else:
        q = N

    epz = ((epg/(q**4))*((q**4)-(q**3))) + ((4*epd*((q**2)-q))/(q**2))
    epx = ((epg/(q**4))*((q**4)-(q**3))) + ((4*epd*((q**2)-q))/(q**2))

    if epx>1:
        epx = 1
    if epz>1:
        epz = 1

    p_success = 1-qrsc_failure(N, k, loss)

    px_incorrect = qrsc_incorrect(N, p, k, epx, loss)

    px_correct = qrsc_correct(N, p, k, epx, loss)

    pz_incorrect = qrsc_incorrect(N, p, k, epz, loss)

    pz_correct = qrsc_correct(N, p, k, epz, loss)

    error_probability = error_prob(px_correct, px_incorrect, pz_correct, pz_incorrect)

    return px_correct, px_incorrect, pz_correct, pz_incorrect, p_success


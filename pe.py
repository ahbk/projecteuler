# -*- coding: utf-8 -*-
from material import *

def recurse(r, t, y, v=None):
    """
    Recurse v=r(v) while t(v) and yield v on y(v)

    """
    if v is None:
        v = r()

    while t(v):
        if y(v): yield v
        v = r(v)

def build(r, t, l=None):
    """
    Append r(l[-1]) to l if y(l) while t(l)
    Return l
    """
    if l is None:
        l = [r()]

    while t(l):
        l.append(r(l[-1]))

    return l

def running_op(l, op, i):
    """
    Running total
    =============
    l = [1,2,3,4,5]
    fn = lambda i,v: i+v
    i = 0

    result = [1,3,6,10,15]


    Running product
    ===============
    l = [1,2,3,4,5]
    fn = lambda i,v: i*v
    i = 1

    result = [1,2,6,24,120]

    """
    for v in l:
        i = op(i, v)
        yield i


def dict_merge(dicts, fn, default=0):
    return {k:fn(*(d.get(k, default) for d in dicts)) for k in set(key for d in dicts for key in d)}

def factorial(n):
    return n * factorial(n-1) if n else 1

def decimals(n, d):
    """
    Convert a fraction to its decimal representation
    Returns a tuple with the reptend as the second element
    Example (1,7) -> ((0), (1,4,2,8,5,7))
    """
    ns, result = zip(*map(lambda n: (n, n//d), [n] + build(
            r=lambda n=n: (n % d) * 10,
            t=lambda l: len(l) == len(set(l)),
            )))

    i = ns.index(ns[-1])

    return result[:i], result[i:-1]

def primes(n):
    """
    Generate all primes up to and including n

    Bounds: 1 < n < PYSSIZE_T_MAX (usually 536870912)
    Example: 12 -> (2,3,5,7,11)
    """
    sieve = bytearray(n+1)
    sieve[0] = sieve[1] = 1

    for i, m in enumerate(sieve):
        if not m:
            yield i
            for divisibles in xrange(i*i, n+1, i):
                sieve[divisibles] = 1


def prime_factors(v):
    """
    Generate all prime factors for v

    Example: 12 -> (2,2,3)
    """
    n = 2
    while v > 1024*1024:
        if not v%n:
            v /= n
            yield n
        else:
            n += 1

    for p in primes(int(v**.5)):
        while not v%p:
            yield p
            v /= p

    if not v == 1:
        yield v


def divisors(v):
    import itertools
    fs = sorted(list(prime_factors(v)))
    ds = []

    for r in range(1, len(fs)):
        for c in itertools.combinations(fs, r):
            d = reduce(lambda x,y: x*y, c)
            if not d in ds:
                ds.append(d)
                yield d

def cartesian_product(sets, n):

    sets = sets[::-1]

    bases = [1] + list(running_fn(
        l=[len(s) for s in sets],
        fn=lambda i,v: i*v,
        i=1
        ))

    return [sets[i][n % bases[i+1] // bases[i]] for i in range(0, len(bases)-1)][::-1]


def prime_factors_count(v):
    """
    Create a prime factorization of v with prime factors as keys and multiples as values

    Example: 12 -> {2:2, 3:1}
    """
    fs = list(prime_factors(v))
    return {f: fs.count(f) for f in set(fs)}


def greatest_common_factor(*values):
    common_factors = dict_merge([prime_factors_count(v) for v in values], min)
    return max([f**common_factors[f] for f in common_factors if common_factors[f]] + [1])

def pythagorean_triples(m=2, n=0, limit=lambda *v: True, condition=lambda *v: True):

    def increase_mn(m, n):
        n += 1
        if m > n:
            return m, n
        else:
            return m+1, 1

    def make_triple(m, n):
        return m**2 - n**2, 2*m*n, m**2 + n**2 # a,b,c, a²+b²=c²

    kwargs = {
            'r': lambda v=(m,n): increase_mn(*v),
            't': lambda v: limit(*make_triple(*v)),
            'y': lambda v: condition(*make_triple(*v))
            }

    for m, n in recurse(**kwargs):
        yield make_triple(m, n)

def binomial_coefficient(n, k):

    r = 1
    if k > n:
        return 0

    for d in range(0, k):
        r, n = r*n/(d+1), n-1

    return r


def p001():
    return sum(n for n in range(3,1000,3)) + sum(n for n in range(5,1000,5) if n%3)

def p002():
    kwargs = {
            'r': lambda v=(0,1): (max(v), sum(v)), # Transform (a, b) -> (b, a+b), b >= a
            't': lambda v: max(v) < 4000000,
            'y': lambda v: not max(v)%2
            }

    return sum(zip(*recurse(**kwargs))[1])


def p003():
    v = 600851475143
    return max(prime_factors(v))

def p004():
    def is_palindrome(v):
        return str(v) == str(v)[::-1]

    def split_and_multiply(v):
        return int(str(v)[3:]) * int(str(v)[:3])

    kwargs = {
            'r': lambda v=100100: v+1,
            't': lambda v: v <= 999999,
            'y': lambda v: is_palindrome(split_and_multiply(v))
            }

    return split_and_multiply(sorted(recurse(**kwargs), key=split_and_multiply)[-1])

def p005():
    max_factors_count = dict_merge([prime_factors_count(n+1) for n in range(0,20)], max)
    return reduce(lambda x, y: x*y, (f ** max_factors_count[f] for f in max_factors_count))

def p006():
    n = 100
    square_of_sum = n**2 * (n+1)**2 * 1/4
    sum_of_squares = n * (n+1) * (2*n+1) * 1/6

    return square_of_sum - sum_of_squares

def p007():
    guess = 110000
    return list(primes(guess))[10000]

def p008():
    n = rp008()
    m = 0

    for i in range(0, len(n)-13):
        m = max(m, reduce(lambda x, y: int(x)*int(y), n[i:i+13]))

    return m

def p009():
    for p in pythagorean_triples():
        if sum(p) == 1000:
            return reduce(lambda x, y: x*y, p)

def p010():
    return sum(primes(2000000))

def p011():
    m = rp011()
    steps = (0,1), (0,20), (0,21), (3,19)

    return max(max(reduce(lambda x,y: x*y, l) for l in (m[i+o:i+4*s:s] for o,s in steps)) for i in range(0, len(m)-3))

def p012():
    for i in xrange(2, 15000):
        # n = (i**2 + i)/2
        # divisors = product of (prime factor multiples + 1)
        # prime factors of n = all prime factors of i and (i + 1) minus one multiple of 2

        factors = (list(prime_factors(i)) + list(prime_factors(i+1)))[1:]
        if reduce(lambda x,y: x*y, (factors.count(f)+1 for f in set(factors))) > 500:
            return int((i**2+i)/2)

def p013():
    m = rp013()
    precision = 10
    return str(int(sum(value * (10 ** digit) for digit, value in [(precision-i, sum(map(int, n))) for i,n in enumerate(zip(*m))])))[:10]

def p014():
    lengths = {1:1, 2:2}

    for n in xrange(999999, 2, -2):
        if n in lengths:
            continue

        chain = list(reversed([v for v in recurse(
                r=lambda v=n: 3*v+1 if v%2 else v/2,
                t=lambda v: v > 1 and v not in lengths,
                y=lambda v: True,
                )])) + [n]

        l = lengths[3*v+1 if v%2 else v/2]

        lengths.update({v: i+1 + l for i,v in enumerate(chain)})

    return max(lengths, key=lengths.get)

def p015():
    return binomial_coefficient(40,20)

def p016():
    return sum(map(int, str(2**1000)))

def p017():
    import inflect
    p = inflect.engine()
    return len(''.join([p.number_to_words(i+1) for i in range(0,1000)]).replace('-', '').replace(' ', ''))

def p018():
    numbers = rp018()

    N = len(numbers)
    row = lambda i: int((-1 + (1 + 8*(i-1))**.5)/2) + 1

    for i in range(N-row(N), 0, -1):
        numbers[i-1] += max(numbers[i+row(i)-1:i+row(i)+1])

    return numbers[0]

def p019():
    import datetime

    return len([d for d in recurse(
        r=lambda v=datetime.date(1901,1,1): v+datetime.timedelta(days=1),
        t=lambda v: v <= datetime.date(2000,12,31),
        y=lambda v: v.day == 1 and v.weekday() == 6,
        )])


def p020():
    return sum(map(int, str(factorial(100))))

def p021():

    def is_amicable(v, amicables=[]):
        if v in amicables:
            return True

        vp = sum(divisors(v)) + 1
        if v != vp and v == sum(divisors(vp)) + 1:
            amicables += [v, vp]
            return True

        return False

    return sum(list([v for v in recurse(
        r=lambda v=0: v+1,
        t=lambda v: v < 10000,
        y=lambda v: is_amicable(v),
        )]))


def p022():
    names = sorted(rp022())
    return sum((i+1) * sum(ord(l) - ord('A') + 1 for l in names[i]) for i in range(0, len(names)))

def p023():

    sum_of_abundants = [False]*28124

    abundant_numbers = [a for a in recurse(
        r=lambda v=0: v+1,
        t=lambda v: v <= 28123,
        y=lambda v: sum(divisors(v)) + 1 > v,
        )]

    for i,a in enumerate(abundant_numbers):
        for b in abundant_numbers[:i+1]:
            try:
                sum_of_abundants[a + b] = True
            except IndexError:
                break

    return sum([n for n,a in enumerate(sum_of_abundants) if not a])

def p024():
    import itertools
    return [p for i,p in enumerate(itertools.permutations(range(0,10), 10)) if i == 1000000-1][0]

def p025():
    kwargs = {
            'r': lambda v=(0,1): (max(v), sum(v)), # Transform (a, b) -> (b, a+b), b >= a
            't': lambda v: len(str(v[0])) < 1000,
            'y': lambda v: True,
            }

    return len(list(recurse(**kwargs))) + 1

def p026():

    m = (0, None)

    for n in range(999, 1, -1):
        l = len(decimals(1,n)[1])
        if l > m[0]:
            m = (l, n)

    return m[1]

def p027():
    ps = list(primes(1000))

    m = (0, None)

    for am in range(0, 1000):
        for a in (am, -am):
            for bm in ps:
                for b in (bm, -bm):
                    for n in range(0,80):
                        if not (n**2 + a*n + b) in ps:
                            break

                    if n > m[0]:
                        m = (n, a*b)

    return m[1]

def p028():
    return sum(4*(4*r*r + r + 1) for r in range(0,501)) - 3

def p029():
    return len(set([a**b for a in range(2, 101) for b in range(2, 101)]))

def p030():
    return sum(recurse(
            r=lambda v=1: v+1,
            y=lambda v: v == sum(int(c) ** 5 for c in str(v)),
            t=lambda v: v <= len(str(v)) * (9 ** 5),
            ))

def p031():
    from bisect import bisect

    coins = 1, 2, 5, 10, 20, 50, 100, 200
    value = 200

    def ways(v, i):
        i = bisect(coins[:i], v) - 1
        return ways(v - coins[i], i + 1) + ways(v, i) if i > 0 else 1

    return ways(value, len(coins))


def p032():
    from itertools import permutations

    s = []
    for p in permutations('123456789'):
        a, b, c, d, e = map(lambda v: int(''.join(v)), (p[:1], p[1:5], p[:2], p[2:5], p[5:9]))
        if e not in s:
            if a * b == e:
                s.append(e)
            if c * d == e:
                s.append(e)
    
    return sum(s)

print(p031())

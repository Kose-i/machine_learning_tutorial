## expression matcher
from kanren import run, var, fact
import kanren.assoccomm as la

add = 'addition'
mul = 'multiplication'

fact(la.commutative, mul)
fact(la.commutative, add)
fact(la.commutative, mul)
fact(la.commutative, add)

a,b,c = var('a'), var('b'), var('c')

# (3*(-2)) + ((1+2*3)*(-1))
expression_orig = (add, (mul, 3, -2), (mul, (add, 1 , (mul, 2, 3)), -1))
# (1 + (2*a))*b + 3*c
expression1 = (add, (mul, (add, 1, (mul, 2, a)), b), (mul, 3, c))
# c*3 + b*(2*a + 1)
expression2 = (add, (mul, c, 3), (mul, b, (add, (mul, 2, a), 1)))
# 2*a*b + b + 3*c
expression3 = (add, (add, (mul, (mul, 2, a), b)), (mul, 3, c))

print(run(0, (a,b,c), la.eq_assoccomm(expression1, expression_orig)))
print(run(0, (a,b,c), la.eq_assoccomm(expression2, expression_orig)))
print(run(0, (a,b,c), la.eq_assoccomm(expression3, expression_orig)))

## Check prime number
import itertools as it
from kanren import isvar, membero, var, run, eq
from kanren.core import success, fail, condeseq
from sympy.ntheory.generate import prime, isprime

def check_prime(x):
    if isvar(x):
        return condeseq([eq(x,p)] for p in map(prime, it.count(1)))
    else:
        return success if isprime(x) else fail
x = var()

list_nums = (23, 4, 27, 17, 13, 10, 21, 29, 3, 32, 11, 19)
print('List of primes in the list:')
print(set(run(0, x, (membero, x, list_nums), (check_prime, x))))
print('List of first 7 prime numbers:')
print(run(7, x, (check_prime, x)))

## family tree
import json
from kanren import Relation, facts, run, conde, var, eq

with open('datasets/relationships.json') as f:
    d = json.loads(f.read())
father = Relation()
mother = Relation()

for item in d['father']:
    facts(mother, (list(item.keys())[0], list(item.values())[0]))
for item in d['mother']:
    facts(father, (list(item.keys())[0], list(item.values())[0]))

def parent(x, y):
    return conde((father(x, y),), (mother(x,y),))
def grandparent(x, y):
    temp = var()
    return conde((parent(x, temp), parent(temp, y)))
def sibling(x, y):
    temp = var()
    return conde((parent(temp, x), parent(temp, y)))
def uncle(x, y):
    temp = var()
    return conde((father(temp, x), grandparent(temp, y)))

x = var()
output = run(0, x, (father, 'John', x))
for item in output:
    print(item)

output = run(0, x, (mother, x, 'William'))
for item in output:
    print(item)

output = run(0, x, parent(x, 'Adam'))
for item in output:
    print(item)

output = run(0, x, grandparent(x, 'Wayne'))
for item in output:
    print(item)

output = run(0, x, grandparent('Megan', x))
for item in output:
    print(item)

name = 'David'
output = run(0, x, sibling(x, name))

siblings = [x for x in output if x != name]
for item in siblings:
    print(item)

name = 'Tiffany'
name_father = run(0, x, father(x, name))[0]
output = run(0, x, uncle(x, name))
output  = [x for x in output if x != name_father]
for item in output:
    print(item)
a, b, c = var(), var(), var()
output = run(0, (a,b), father(a, c), mother(b, c))
for item in output:
    print('Husband:', item[0], '<==> Wife:', item[1])

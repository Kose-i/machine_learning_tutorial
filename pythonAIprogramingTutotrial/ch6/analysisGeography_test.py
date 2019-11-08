from kanren import run, fact, eq, Relation, var

coastal  = Relation()
adjacent = Relation()

file_coastal = 'datasets/coastal_states.txt'

with open(file_coastal, 'r') as f:
    line = f.read()
    coastal_states = line.split(',')

for state in coastal_states:
    fact(coastal, state)

file_adjacent = 'datasets/adjacent_states.txt'
with open(file_adjacent, 'r') as f:
    line = f.read()
    adjlist = [line.strip(',') for line in f if line and line[0].isalpha()]

for L in adjlist:
    head, tail = L[0], L[1:]
    for state in tail:
        fact(adjacent, head, state)
x = var()
output = run(0, x, adjacent('Nevada', 'Louisiana'))
print('Yes' if len(output) else 'No')

output = run(0, x, adjacent('Oregon', x))
for item in output:
    print(item)

output = run(0, x, adjacent('Mississippi', x), coastal(x))
for item in output:
    print(item)

y = var()
output = run(7, x, coastal(y), adjacent(x, y))
for item in output:
    print(item)

output = run(0, x, adjacent('Arkansas', x), adjacent('Kentucky', x))
for item in output:
    print(item)

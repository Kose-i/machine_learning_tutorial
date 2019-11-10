from simpleai.search import CspProblem, backtrack

names = ('Mark', 'Julia', 'Steve', 'Amanda', 'Brian', 'Joanne', 'Derek', 'Allan', 'Michelle', 'Kelly', 'Chris')

colors = dict((name, ['red', 'green', 'blue', 'gray']) for name in names)

def constraint_func(names, values):
    return values[0] != values[1]

constraints =[
    (('Mark',   'Julia'),    constraint_func),
    (('Mark',   'Steve'),    constraint_func),
    (('Julia',  'Steve'),    constraint_func),
    (('Julia',  'Amanda'),   constraint_func),
    (('Julia',  'Derek'),    constraint_func),
    (('Julia',  'Brian'),    constraint_func),
    (('Steve',  'Amanda'),   constraint_func),
    (('Steve',  'Allan'),    constraint_func),
    (('Steve',  'Michelle'), constraint_func),
    (('Amanda', 'Michelle'), constraint_func),
    (('Amanda', 'Joanne'),   constraint_func),
    (('Amanda', 'Derek'),    constraint_func),
    (('Brian',  'Derek'),    constraint_func),
    (('Brian',  'Kelly'),    constraint_func),
    (('Joanne', 'Michelle'), constraint_func),
    (('Joanne', 'Derek'),    constraint_func),
    (('Joanne', 'Chris'),    constraint_func),
    (('Derek',  'Kelly'),    constraint_func),
    (('Derek',  'Chris'),    constraint_func),
    (('Kelly',  'Chris'),    constraint_func),
    (('Allan',  'Michelle'), constraint_func),
]

problem = CspProblem(names, colors, constraints)
output = backtrack(problem)
print('Color mapping:\n')
for k, v in output.items():
    print(k, '==>', v)

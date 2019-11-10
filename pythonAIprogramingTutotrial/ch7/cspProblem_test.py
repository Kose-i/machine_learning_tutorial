variables = ('John', 'Anna', 'Tom', 'Patricia')

domains = {
  'John': [1,2,3],
  'Anna': [1,3],
  'Tom' : [2,4],
  'Patricia': [2,3,4],
}

def constraint_unique(variables, values):
    return len(values) == len(set(values))
def constraint_bigger(variables, values):
    return values[0] > values[1]
def constraint_odd_even(variables, values):
    return values[0] % 2 != values[1] % 2

constraints = [
    (('John', 'Anna', 'Tom'), constraint_unique),
    (('Tom', 'Anna'), constraint_bigger),
    (('John', 'Patricia'), constraint_odd_even),
]

from simpleai.search import CspProblem, backtrack, min_conflicts, MOST_CONSTRAINED_VARIABLE, HIGHEST_DEGREE_VARIABLE, LEAST_CONSTRAINING_VALUE
problem = CspProblem(variables, domains, constraints)
print('Normal:', backtrack(problem))

print('Most constrained variable:', backtrack(problem, variable_heuristic=MOST_CONSTRAINED_VARIABLE))

print('Highest degree variable:', backtrack(problem, variable_heuristic=HIGHEST_DEGREE_VARIABLE))

print('Least constraining value:', backtrack(problem, variable_heuristic=LEAST_CONSTRAINING_VALUE))

print('Most constrained variable and least constraining value:', backtrack(problem, variable_heuristic=MOST_CONSTRAINED_VARIABLE, value_heuristic=LEAST_CONSTRAINING_VALUE))

print('Highest degree and least constraining value:', backtrack(problem, variable_heuristic=HIGHEST_DEGREE_VARIABLE, value_heuristic=LEAST_CONSTRAINING_VALUE))

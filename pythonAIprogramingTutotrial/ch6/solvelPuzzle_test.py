from kanren import *

people = var()
rules = lall(
          (eq, (var(), var(), var(), var()), people),
          (membero, ('Steve', var(), 'blue', var()), people),
          (membero, (var(), 'cat', var(), 'Canada'), people),
          (membero, ('Matthew', var(), var(), 'USA'), people),
          (membero, (var(), var(), 'black', 'Australia'), people),
          (membero, ('Jack', 'cat', var(), var()), people),
          (membero, ('Alfred', var(), var(), 'Australia'), people),
          (membero, (var(), 'dog', var(), 'Frence'), people),
          (membero, (var(), 'rabbit', var(), var()), people),
          )
solutions = run(0, people, rules)
output = [house for house in solutions[0] if 'rabbit' in house][0][0]
print(output + ' is the owner of the rabbit')

print('Here are all the details:')
attribs = ['Name', 'Pet', 'Color', 'Country']
print('\n' + '\t\t'.join(attribs))
print('='*57)
for item in solutions[0]:
    print('\t\t'.join([str(x) for x in item]))

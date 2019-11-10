import simpleai.search as ss

class CustomProblem(ss.SearchProblem):
    def set_target(self, target_string):
        self.target_string = target_string
    def actions(self, cur_state):
        if len(cur_state) < len(self.target_string):
            alphabets = 'abcdefghijklmnopqrstuvwxyz'
            return list(alphabets + ' ' + alphabets.upper())
        else:
            return []
    def result(self, cur_state, action):
        return cur_state + action
    def is_goal(self, cur_state):
        return cur_state == self.target_string
    def heuristic(self, cur_state):
        dist = sum([1 if cur_state[i]!=self.target_string[i] else 0 for i in range(len(cur_state))])
        diff  = len(self.target_string) - len(cur_state)
        return dist + diff

problem = CustomProblem()

#input_string  = 'Artificial Intelligence'
#initial_state = ''
input_string  = 'Artificial Intelligence with Python'
initial_state = 'Artificial Inte'

problem.set_target(input_string)
problem.initial_state = initial_state

output = ss.greedy(problem)
print('Target string:', input_string)
print('\nPath to the solution:')
for item in output.path():
    print(item)


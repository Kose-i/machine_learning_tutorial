import numpy as np
from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax, SSS

class GameController(TwoPlayersGame):
    def __init__(self, players):
        self.players = players
        self.nplayer = 1
        self.board = np.zeros((6,7), dtype=np.int)
        self.pos_dir = np.array(
          [[[i,0],[0,1]] for i in range(6)]   +
          [[[0,i],[1,0]] for i in range(7)]   +
          [[[i,0],[1,1]] for i in range(1,3)] +
          [[[0,i],[1,1]] for i in range(4)]   +
          [[[i,6],[1,-1]] for i in range(1,3)] +
          [[[0,i],[1,-1]] for i in range(3,7)]
        )
    #def possible_moves(self):
    #    return [i for i in range(7) if (self.board[:,i].min() == 0)]
    def possible_moves(self):
        return np.random.permutation([i for i in range(7) if (self.board[:,i].min()==0)])
    def make_move(self, column):
        line =  np.argmin(self.board[:,column] != 0)
        self.board[line, column] = self.nplayer
    def show(self):
        print('\n 0 1 2 3 4 5 6')
        print(13 * '-')
        for j in range(5,-1,-1):
            print(' '.join(['.OX'[self.board[j][i]] for i in range(7)]))
    def loss_condition(self):
        for pos, direction in self.pos_dir:
            streak = 0
            while (0 <= pos[0] <= 5) and (0 <= pos[1] <= 6):
                if self.board[pos[0], pos[1]] == self.nopponent:
                    streak += 1
                    if streak == 4:
                        return True
                else:
                    streak = 0
                pos = pos + direction
        return False
    def is_over(self):
        return (self.board.min() > 0) or self.loss_condition()
    def scoring(self):
        return -100 if self.loss_condition() else 0

algo_neg = Negamax(5)
algo_sss = SSS(5)
game = GameController([AI_Player(algo_neg), AI_Player(algo_sss)])
game.play()
if game.loss_condition():
    print('\nPlayer', game.nopponent, 'wins.')
else:
    print("\nIt's a draw.")

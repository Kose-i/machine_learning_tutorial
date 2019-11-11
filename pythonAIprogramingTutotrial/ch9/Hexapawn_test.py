from easyAI import TwoPlayersGame, AI_Player, Human_Player, Negamax

def to_tuple(s):
    return (3 - int(s[1]), 'abc'.index(s[0]))
def to_string(moves):
    pre, post = moves
    return 'abc'[pre[1]] + str(3 - pre[0]) + ' ' + 'abc'[post[1]] + str(3 - post[0])

class GameController(TwoPlayersGame):
    def __init__(self, players):
        self.players = players
        self.nplayer = 1
        players[0].direction = 1
        players[0].goal_line = 2
        players[0].pawns = [(0,0), (0,1), (0,2)]
        players[1].direction = -1
        players[1].goal_line = 0
        players[1].pawns = [(2,0), (2,1), (2,2)]
    def possible_moves(self):
        moves = []
        opponent_pawns = self.opponent.pawns
        d = self.player.direction
        for i,j in self.player.pawns:
            if (i+d, j) not in opponent_pawns:
                moves.append(((i,j),(i+d, j)))
            if (i+d, j+1) not in opponent_pawns:
                moves.append(((i,j),(i+d, j+1)))
            if (i+d, j-1) not in opponent_pawns:
                moves.append(((i,j),(i+d, j-1)))
        return list(map(to_string, [(pre, post) for pre, post in moves]))
    def make_move(self, moves):
        pre, post = tuple(map(to_tuple, moves.split(' ')))
        ind = self.player.pawns.index(pre)
        self.player.pawns[ind] = post

        if post in self.opponent.pawns:
            self.opponent.pawns.remove(post)
    def loss_condition(self):
        return any([i==self.opponent.goal_line for i,_ in self.opponent.pawns]) or self.possible_moves()==[]
    def is_over(self):
        return self.loss_condition()
    def grid(self, pos):
        if pos in self.players[0].pawns:
            return '1'
        elif pos in self.players[1].pawns:
            return '2'
        else:
            return '.'
    def show(self):
        print('  a b c')
        for i in range(3):
            print(3 - i , ' '.join([self.grid((i,j)) for j in range(3)]))
    def scoring(self):
        return -100 if self.loss_condition() else 0
    
algorithm = Negamax(12)
game = GameController([AI_Player(algorithm), AI_Player(algorithm)])
game.play()
print('Player', game.nopponent, 'wins after', game.nmove, 'turns')

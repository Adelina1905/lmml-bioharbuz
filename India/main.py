import sys
import time

# Constants
TIME_LIMIT = 0.090  # 90ms per move

def read_input():
    """Read input from CodinGame"""
    opponent_row, opponent_col = [int(i) for i in input().split()]
    valid_action_count = int(input())
    valid_moves = []
    for _ in range(valid_action_count):
        row, col = [int(i) for i in input().split()]
        valid_moves.append((row, col))
    return opponent_row, opponent_col, valid_moves

# Track board state
board = [['.' for _ in range(9)] for _ in range(9)]
my_symbol = None
opp_symbol = None
turn = 0

def is_win(cells, player):
    """Check if player won in 3x3 board"""
    for i in range(3):
        if cells[i][0] == cells[i][1] == cells[i][2] == player:
            return True
        if cells[0][i] == cells[1][i] == cells[2][i] == player:
            return True
    if cells[0][0] == cells[1][1] == cells[2][2] == player:
        return True
    if cells[0][2] == cells[1][1] == cells[2][0] == player:
        return True
    return False

def get_local_board(board_state, br, bc):
    """Extract 3x3 local board"""
    start_r, start_c = br * 3, bc * 3
    return [board_state[start_r + i][start_c:start_c + 3] for i in range(3)]

def get_global_state(board_state):
    """Get which local boards are won by each player"""
    global_board = [[None for _ in  range(3)] for _ in range(3)]
    for br in range(3):
        for bc in range(3):
            cells = get_local_board(board_state, br, bc)
            if is_win(cells, my_symbol):
                global_board[br][bc] = my_symbol
            elif is_win(cells, opp_symbol):
                global_board[br][bc] = opp_symbol
    return global_board

def check_global_win(global_board, player):
    """Check if player won globally"""
    for i in range(3):
        if all(global_board[i][j] == player for j in range(3)):
            return True
        if all(global_board[j][i] == player for j in range(3)):
            return True
    if all(global_board[i][i] == player for i in range(3)):
        return True
    if all(global_board[i][2-i] == player for i in range(3)):
        return True
    return False

def count_global_threats(global_board, player):
    """Count two-in-a-row on global board"""
    threats = 0
    for i in range(3):
        row = [global_board[i][j] for j in range(3)]
        if row.count(player) == 2 and row.count(None) >= 1:
            threats += 1
        col = [global_board[j][i] for j in range(3)]
        if col.count(player) == 2 and col.count(None) >= 1:
            threats += 1
    diag1 = [global_board[i][i] for i in range(3)]
    if diag1.count(player) == 2 and diag1.count(None) >= 1:
        threats += 1
    diag2 = [global_board[i][2-i] for i in range(3)]
    if diag2.count(player) == 2 and diag2.count(None) >= 1:
        threats += 1
    return threats

def count_two_in_row(cells, player):
    """Count two-in-a-rows in local board"""
    threats = 0
    for i in range(3):
        if cells[i].count(player) == 2 and cells[i].count('.') == 1:
            threats += 1
        col = [cells[j][i] for j in range(3)]
        if col.count(player) == 2 and col.count('.') == 1:
            threats += 1
    diag1 = [cells[i][i] for i in range(3)]
    if diag1.count(player) == 2 and diag1.count('.') == 1:
        threats += 1
    diag2 = [cells[i][2-i] for i in range(3)]
    if diag2.count(player) == 2 and diag2.count('.') == 1:
        threats += 1
    return threats

def evaluate(board_state):
    """Evaluate board position"""
    global_board = get_global_state(board_state)
    
    # Check for global win/loss
    if check_global_win(global_board, my_symbol):
        return 1000000
    if check_global_win(global_board, opp_symbol):
        return -1000000
    
    score = 0
    
    # Global threats (very important!)
    my_global_threats = count_global_threats(global_board, my_symbol)
    opp_global_threats = count_global_threats(global_board, opp_symbol)
    score += my_global_threats * 5000
    score -= opp_global_threats * 5000
    
    # Count won local boards
    for br in range(3):
        for bc in range(3):
            cells = get_local_board(board_state, br, bc)
            
            if is_win(cells, my_symbol):
                bonus = 2000 if (br == 1 and bc == 1) else 1500 if (br + bc) % 2 == 0 else 1000
                score += bonus
            elif is_win(cells, opp_symbol):
                bonus = 2000 if (br == 1 and bc == 1) else 1500 if (br + bc) % 2 == 0 else 1000
                score -= bonus
            else:
                my_threats = count_two_in_row(cells, my_symbol)
                opp_threats = count_two_in_row(cells, opp_symbol)
                score += my_threats * 100
                score -= opp_threats * 100
                
                flat = [cell for row in cells for cell in row]
                score += flat.count(my_symbol) * 10
                score -= flat.count(opp_symbol) * 10
                
                if cells[1][1] == my_symbol:
                    score += 50
                elif cells[1][1] == opp_symbol:
                    score -= 50
    
    return score

def order_moves(board_state, moves):
    """Order moves for better alpha-beta pruning"""
    scored = []
    for move in moves:
        r, c = move
        br, bc = r // 3, c // 3
        local_r, local_c = r % 3, c % 3
        priority = 0
        
        # Check if wins local board
        board_state[r][c] = my_symbol
        cells = get_local_board(board_state, br, bc)
        if is_win(cells, my_symbol):
            priority += 10000
        board_state[r][c] = '.'
        
        # Check if blocks opponent win
        board_state[r][c] = opp_symbol
        cells = get_local_board(board_state, br, bc)
        if is_win(cells, opp_symbol):
            priority += 5000
        board_state[r][c] = '.'
        
        # Strategic positions
        if local_r == 1 and local_c == 1:
            priority += 50
        if br == 1 and bc == 1:
            priority += 30
        
        scored.append((priority, move))
    
    scored.sort(reverse=True)
    return [move for _, move in scored]

def minimax(board_state, valid_moves, depth, alpha, beta, maximizing, start_time, nodes):
    """Minimax with alpha-beta pruning"""
    # Time and node limits
    if time.time() - start_time > TIME_LIMIT or nodes[0] > 10000 or depth == 0:
        return evaluate(board_state), None
    
    nodes[0] += 1
    
    if not valid_moves:
        return evaluate(board_state), None
    
    # Order moves for better pruning
    if depth >= 2:
        valid_moves = order_moves(board_state, valid_moves)
    
    # Limit branching factor
    valid_moves = valid_moves[:min(len(valid_moves), 12)]
    
    best_move = None
    
    if maximizing:
        max_eval = -float('inf')
        for move in valid_moves:
            r, c = move
            old = board_state[r][c]
            board_state[r][c] = my_symbol
            
            if depth == 1:
                eval_score = evaluate(board_state)
            else:
                eval_score, _ = minimax(board_state, valid_moves, depth - 1, alpha, beta, False, start_time, nodes)
            
            board_state[r][c] = old
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in valid_moves:
            r, c = move
            old = board_state[r][c]
            board_state[r][c] = opp_symbol
            
            if depth == 1:
                eval_score = evaluate(board_state)
            else:
                eval_score, _ = minimax(board_state, valid_moves, depth - 1, alpha, beta, True, start_time, nodes)
            
            board_state[r][c] = old
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        
        return min_eval, best_move

def main():
    global my_symbol, opp_symbol, turn
    
    while True:
        opponent_row, opponent_col, valid_moves = read_input()
        turn += 1
        
        # Update board with opponent's move
        if opponent_row != -1 and opponent_col != -1:
            if opp_symbol is None:
                if turn == 1:
                    my_symbol = 'X'
                    opp_symbol = 'O'
                else:
                    my_symbol = 'O'
                    opp_symbol = 'X'
            board[opponent_row][opponent_col] = opp_symbol
        elif my_symbol is None:
            my_symbol = 'X'
            opp_symbol = 'O'
        
        if not valid_moves:
            print("0 0", flush=True)
            continue
        
        # Start with first valid move as default
        best_move = valid_moves[0]
        best_score = -float('inf')
        
        # Check for immediate winning moves (global win)
        for move in valid_moves:
            r, c = move
            board[r][c] = my_symbol
            global_board = get_global_state(board)
            if check_global_win(global_board, my_symbol):
                # Immediate global win - highest priority!
                best_move = move
                best_score = float('inf')
                board[r][c] = '.'
                break
            board[r][c] = '.'
        
        # If no immediate win, run minimax
        if best_score < float('inf'):
            start_time = time.time()
            for depth in range(1, 8):
                nodes = [0]
                score, move = minimax(board, valid_moves, depth, -float('inf'), float('inf'), True, start_time, nodes)
                if move and move in valid_moves:
                    if score > best_score:
                        best_score = score
                        best_move = move
                if time.time() - start_time > TIME_LIMIT:
                    break
        
        # Final safety check
        if best_move not in valid_moves:
            best_move = valid_moves[0]
        
        # Update board
        board[best_move[0]][best_move[1]] = my_symbol
        
        # Output move
        print(f"{best_move[0]} {best_move[1]}", flush=True)

if __name__ == "__main__":
    main()

import sys

player_idx = int(input())
nb_games = int(input())

class MoveSimulator:
    """Simulates moves with perfect accuracy"""
    
    @staticmethod
    def execute_move(gpu, pos, stun, move):
        """
        Returns: (final_pos, got_stunned, actual_distance_moved)
        """
        if stun > 0:
            return pos, False, 0
        
        track_len = len(gpu)
        
        if move == 'LEFT':
            steps, jump = 1, False
        elif move == 'DOWN':
            steps, jump = 2, False
        elif move == 'RIGHT':
            steps, jump = 3, False
        elif move == 'UP':
            steps, jump = 2, True
        else:
            return pos, False, 0
        
        final_pos = pos
        
        for i in range(1, steps + 1):
            next_pos = pos + i
            
            # Reached end
            if next_pos >= track_len:
                return track_len, False, track_len - pos
            
            # UP skips checking position pos+1
            if jump and i == 1:
                continue
            
            # Hit hurdle - stop here and get stunned
            if gpu[next_pos] == '#':
                return next_pos, True, next_pos - pos
        
        return pos + steps, False, steps

class GameAnalyzer:
    """Analyzes game state and outcomes"""
    
    @staticmethod
    def get_race_outcome(my_final_pos, opp_final_positions, track_len):
        """
        Returns medal value: 3 for gold, 1 for silver, 0 for bronze
        """
        if my_final_pos < track_len:
            # Haven't finished yet
            return None
        
        # Count how many opponents finished
        finished_opps = sum(1 for p in opp_final_positions if p >= track_len)
        
        if finished_opps == 0:
            return 3  # Gold (first place)
        elif finished_opps == 1:
            return 1  # Silver (second place)
        else:
            return 0  # Bronze (third place)
    
    @staticmethod
    def evaluate_position(gpu, pos, track_len):
        """
        Evaluate how good a position is (0-1 scale)
        Considers: distance to finish, upcoming obstacles
        """
        if pos >= track_len:
            return 1.0
        
        progress = pos / track_len
        
        # Count hurdles in next 5 spaces
        hurdles_ahead = 0
        for i in range(pos + 1, min(track_len, pos + 6)):
            if gpu[i] == '#':
                hurdles_ahead += 1
        
        # Penalty for hurdles ahead
        obstacle_penalty = hurdles_ahead * 0.05
        
        return max(0, progress - obstacle_penalty)

class StrategyEngine:
    """Main decision-making engine"""
    
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.sim = MoveSimulator()
        self.analyzer = GameAnalyzer()
    
    def score_move_comprehensive(self, game, move):
        """
        Comprehensive scoring for a single game
        Returns score (higher = better)
        """
        gpu = game['gpu']
        
        if gpu == "GAME_OVER":
            return 0
        
        track_len = len(gpu)
        my_pos = game['positions'][self.player_idx]
        my_stun = game['stuns'][self.player_idx]
        
        opp_positions = [game['positions'][i] for i in range(3) if i != self.player_idx]
        opp_stuns = [game['stuns'][i] for i in range(3) if i != self.player_idx]
        
        # Simulate my move
        my_new_pos, i_hit_hurdle, my_dist = self.sim.execute_move(gpu, my_pos, my_stun, move)
        
        # Simulate opponent moves (assume they do same move - approximate)
        opp_new_positions = []
        for opp_pos, opp_stun in zip(opp_positions, opp_stuns):
            opp_new_pos, _, _ = self.sim.execute_move(gpu, opp_pos, opp_stun, move)
            opp_new_positions.append(opp_new_pos)
        
        score = 0
        
        # 1. CRITICAL: Avoid hurdles at all costs
        if i_hit_hurdle:
            return -100000  # Never hit hurdles
        
        # 2. Distance moved is good
        score += my_dist * 1000
        
        # 3. Finishing the race is VERY valuable
        if my_new_pos >= track_len:
            score += 10000
            
            # Check if we might win medal
            medal = self.analyzer.get_race_outcome(my_new_pos, opp_new_positions, track_len)
            if medal == 3:  # Gold
                score += 5000
            elif medal == 1:  # Silver
                score += 3000
        
        # 4. Position quality (avoid setting up bad next turn)
        if my_new_pos < track_len:
            pos_quality = self.analyzer.evaluate_position(gpu, my_new_pos, track_len)
            score += pos_quality * 500
        
        # 5. Relative position matters
        ahead_count = sum(1 for opp_pos in opp_new_positions if my_new_pos > opp_pos)
        score += ahead_count * 200
        
        return score
    
    def choose_move(self, games_data):
        """
        Choose best move across all games
        Uses min-max strategy to avoid catastrophic failures
        """
        moves = ['LEFT', 'DOWN', 'RIGHT', 'UP']
        
        # Count active games
        active_games = [g for g in games_data if g['gpu'] != "GAME_OVER"]
        
        if not active_games:
            return 'LEFT'
        
        # Score each move for each game
        move_game_scores = {move: [] for move in moves}
        
        for game in games_data:
            if game['gpu'] == "GAME_OVER":
                continue
            
            for move in moves:
                score = self.score_move_comprehensive(game, move)
                move_game_scores[move].append(score)
        
        # Strategy: Minimize worst-case + maximize average
        # This prevents sacrificing one game for others
        move_evaluations = {}
        
        for move in moves:
            scores = move_game_scores[move]
            if not scores:
                move_evaluations[move] = 0
                continue
            
            min_score = min(scores)  # Worst game
            avg_score = sum(scores) / len(scores)  # Average
            
            # Weight both: 60% average, 40% worst-case
            combined = 0.6 * avg_score + 0.4 * min_score
            move_evaluations[move] = combined
        
        best_move = max(move_evaluations.items(), key=lambda x: x[1])
        
        # Debug
        print(f"Evals: {move_evaluations}", file=sys.stderr, flush=True)
        print(f"Best: {best_move[0]} ({best_move[1]:.0f})", file=sys.stderr, flush=True)
        
        return best_move[0]

# Initialize strategy engine
strategy = StrategyEngine(player_idx)

# Game loop
turn_num = 0
while True:
    turn_num += 1
    
    # Read scores
    scores = []
    for i in range(3):
        scores.append(input())
    
    # Parse my score for debugging
    my_score_parts = scores[player_idx].split()
    my_total_score = int(my_score_parts[0])
    
    # Read games
    games_data = []
    for i in range(nb_games):
        parts = input().split()
        games_data.append({
            'gpu': parts[0],
            'positions': [int(parts[1]), int(parts[2]), int(parts[3])],
            'stuns': [int(parts[4]), int(parts[5]), int(parts[6])]
        })
    
    # Debug output
    print(f"\n=== Turn {turn_num} | Score: {my_total_score} ===", file=sys.stderr, flush=True)
    for idx, g in enumerate(games_data):
        if g['gpu'] != "GAME_OVER":
            p = g['positions'][player_idx]
            s = g['stuns'][player_idx]
            print(f"G{idx}: pos={p:2d} stun={s} | {g['gpu']}", file=sys.stderr, flush=True)
    
    # Make decision
    move = strategy.choose_move(games_data)
    print(move)

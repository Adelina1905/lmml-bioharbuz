# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing
from collections import deque


# Helper functions
def get_next_position(pos: typing.Dict, direction: str) -> typing.Dict:
    """Get the next position based on current position and direction"""
    x, y = pos["x"], pos["y"]
    if direction == "up":
        return {"x": x, "y": y + 1}
    elif direction == "down":
        return {"x": x, "y": y - 1}
    elif direction == "left":
        return {"x": x - 1, "y": y}
    elif direction == "right":
        return {"x": x + 1, "y": y}
    return pos


def manhattan_distance(pos1: typing.Dict, pos2: typing.Dict) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1["x"] - pos2["x"]) + abs(pos1["y"] - pos2["y"])


def flood_fill(start: typing.Dict, board_width: int, board_height: int,
               obstacles: typing.Set[tuple]) -> int:
    """Count available spaces using flood fill algorithm"""
    visited = set()
    queue = deque([start])
    count = 0

    while queue:
        pos = queue.popleft()
        pos_tuple = (pos["x"], pos["y"])

        if pos_tuple in visited or pos_tuple in obstacles:
            continue

        if pos["x"] < 0 or pos["x"] >= board_width or pos["y"] < 0 or pos[
                "y"] >= board_height:
            continue

        visited.add(pos_tuple)
        count += 1

        for direction in ["up", "down", "left", "right"]:
            next_pos = get_next_position(pos, direction)
            queue.append(next_pos)

    return count


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "AverageCEnjoyer",
        "color": "#FF0080",
        "head": "evil",
        "tail": "curled",
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    my_head = game_state["you"]["body"][0]
    my_body = game_state["you"]["body"]
    my_health = game_state["you"]["health"]
    my_length = game_state["you"]["length"]

    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    food = game_state["board"]["food"]
    opponents = game_state["board"]["snakes"]

    is_move_safe = {"up": True, "down": True, "left": True, "right": True}
    move_scores = {"up": 0, "down": 0, "left": 0, "right": 0}

    # Step 1: Prevent moving backwards (into neck)
    if len(my_body) > 1:
        my_neck = my_body[1]
        if my_neck["x"] < my_head["x"]:
            is_move_safe["left"] = False
        elif my_neck["x"] > my_head["x"]:
            is_move_safe["right"] = False
        elif my_neck["y"] < my_head["y"]:
            is_move_safe["down"] = False
        elif my_neck["y"] > my_head["y"]:
            is_move_safe["up"] = False

    # Build set of all obstacles (snake bodies)
    obstacles = set()

    # Step 2: Avoid self-collision
    for segment in my_body[:-1]:
        obstacles.add((segment["x"], segment["y"]))

    # Step 3: Identify opponents and their threat level
    smaller_opponents = []
    larger_opponents = []

    for snake in opponents:
        if snake["id"] == game_state["you"]["id"]:
            continue

        for segment in snake["body"]:
            obstacles.add((segment["x"], segment["y"]))

        opponent_head = snake["body"][0]
        opponent_length = snake["length"]

        if opponent_length < my_length:
            smaller_opponents.append(snake)
        else:
            larger_opponents.append(snake)
            for direction in ["up", "down", "left", "right"]:
                next_pos = get_next_position(opponent_head, direction)
                obstacles.add((next_pos["x"], next_pos["y"]))

    # Step 4: Check each possible move and evaluate safety
    for direction in ["up", "down", "left", "right"]:
        if not is_move_safe[direction]:
            continue

        next_pos = get_next_position(my_head, direction)

        # Check boundaries
        if next_pos["x"] < 0 or next_pos["x"] >= board_width:
            is_move_safe[direction] = False
            continue
        if next_pos["y"] < 0 or next_pos["y"] >= board_height:
            is_move_safe[direction] = False
            continue

        # Check obstacles
        if (next_pos["x"], next_pos["y"]) in obstacles:
            is_move_safe[direction] = False
            continue

        # Use flood fill to evaluate available space
        available_space = flood_fill(next_pos, board_width, board_height,
                                     obstacles)
        move_scores[direction] += available_space * 3

    # Step 5: AGGRESSIVE FOOD SEEKING - Eat to grow and dominate
    # Prioritize food much more aggressively (health < 75 instead of < 30)
    if food and (my_health < 75 or my_length < 15):
        closest_food = min(food, key=lambda f: manhattan_distance(my_head, f))

        for direction in ["up", "down", "left", "right"]:
            if not is_move_safe[direction]:
                continue

            next_pos = get_next_position(my_head, direction)
            current_distance = manhattan_distance(my_head, closest_food)
            next_distance = manhattan_distance(next_pos, closest_food)

            if next_distance < current_distance:
                if my_health < 20:
                    bonus = 200
                elif my_health < 40:
                    bonus = 120
                elif my_health < 60:
                    bonus = 80
                else:
                    bonus = 50
                move_scores[direction] += bonus

    # Step 6: AGGRESSIVE OPPONENT HUNTING
    # When we're longer, actively chase and corner smaller snakes
    if smaller_opponents and my_length > 5:
        for opponent in smaller_opponents:
            opponent_head = opponent["body"][0]
            opponent_tail = opponent["body"][-1]

            # Calculate opponent's current available space
            base_opponent_obstacles = set(obstacles)
            base_opponent_obstacles.discard(
                (opponent_head["x"], opponent_head["y"]))
            base_opponent_obstacles.discard(
                (opponent_tail["x"], opponent_tail["y"]))
            current_opponent_space = flood_fill(opponent_head, board_width,
                                                board_height,
                                                base_opponent_obstacles)

            for direction in ["up", "down", "left", "right"]:
                if not is_move_safe[direction]:
                    continue

                next_pos = get_next_position(my_head, direction)
                current_distance = manhattan_distance(my_head, opponent_head)
                next_distance = manhattan_distance(next_pos, opponent_head)

                # Chase them aggressively
                if next_distance < current_distance:
                    move_scores[direction] += 60

                # Calculate opponent space AFTER we make this move (trap detection)
                future_opponent_obstacles = set(base_opponent_obstacles)
                future_opponent_obstacles.add((next_pos["x"], next_pos["y"]))
                future_opponent_space = flood_fill(opponent_head, board_width,
                                                   board_height,
                                                   future_opponent_obstacles)

                # Reward moves that reduce opponent's available space (trapping)
                space_reduction = current_opponent_space - future_opponent_space
                if space_reduction > 0:
                    move_scores[direction] += space_reduction * 2

    # Step 7: Control center when dominant
    if my_length > 6:
        center_x = board_width // 2
        center_y = board_height // 2

        for direction in ["up", "down", "left", "right"]:
            if not is_move_safe[direction]:
                continue

            next_pos = get_next_position(my_head, direction)
            center_distance = abs(next_pos["x"] -
                                  center_x) + abs(next_pos["y"] - center_y)
            move_scores[direction] += (board_width + board_height -
                                       center_distance) * 1.5

    # Choose the best safe move
    safe_moves = [move for move, safe in is_move_safe.items() if safe]

    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves! Attempting down")
        return {"move": "down"}

    # Sort moves by score and pick the best one
    safe_moves.sort(key=lambda m: move_scores[m], reverse=True)
    best_move = safe_moves[0]

    print(
        f"MOVE {game_state['turn']}: {best_move} (score: {move_scores[best_move]:.1f}, health: {my_health})"
    )
    return {"move": best_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})

# Battlesnake Logic Overview

This Battlesnake is designed for competitive play on battlesnake.com. It balances survival, aggressive food seeking, hunting smaller snakes, and controlling the board center.

## General Behaviours

- **Avoids collisions** with itself, other snakes, walls, and dangerous squares near larger snakes' heads.
- **Seeks food aggressively** when health is low or snake is small.
- **Hunts smaller snakes** when longer, attempting to trap or corner them.
- **Controls the board center** when dominant, maximizing territory.

## Main Functions

### `get_next_position(pos, direction)`
Returns the position after moving one step in the specified direction.

### `manhattan_distance(pos1, pos2)`
Calculates grid (Manhattan) distance between two positions.

### `flood_fill(start, board_width, board_height, obstacles)`
Counts the number of reachable open tiles from a starting position, avoiding obstacles.

### `info()`
Describes the snake's appearance and author for the Battlesnake API.

### `start(game_state)`
Handles game start events (prints a message).

### `end(game_state)`
Handles game end events (prints a message).

### `move(game_state)`
Main decision function:
- Prevents moving into the neck.
- Avoids obstacles and dangerous squares.
- Scores moves by available space, food proximity, hunting/trapping opportunities, and center control.
- Returns the safest, highest-scoring move.

# Battlesnake Python Starter Project

## Overview
This is an advanced Battlesnake AI written in Python using Flask. Battlesnake is a multiplayer programming game where you control a snake by writing code to make decisions. The snake must navigate a grid, avoid walls and other snakes, and eat food to survive.

This project provides a competitive Battlesnake API server with advanced strategies that can be connected to the Battlesnake game at [play.battlesnake.com](https://play.battlesnake.com).

## Project Structure
- `main.py` - Contains the Battlesnake AI logic with advanced decision-making algorithms
- `server.py` - Flask web server that exposes the Battlesnake API endpoints
- `requirements.txt` - Python dependencies (Flask)

## How It Works
The Battlesnake API has four main endpoints:
- `GET /` - Returns snake appearance and metadata
- `POST /start` - Called when a game starts
- `POST /move` - Called each turn to get the snake's next move
- `POST /end` - Called when a game ends

## Advanced Features Implemented
The AI uses an **AGGRESSIVE** strategy designed to dominate opponents:

1. **Boundary Detection** - Prevents moving out of bounds
2. **Self-Collision Avoidance** - Never runs into its own body
3. **Opponent Collision Avoidance** - Avoids larger opponent snakes and their possible next moves
4. **Aggressive Food Seeking** - Eats frequently (health < 75 or length < 8) with high priority bonuses (50-200 points) to grow longer faster
5. **Flood Fill Algorithm** - Evaluates available space for every move to maintain escape routes
6. **Active Hunting** - When longer than opponents (length > 5), actively chases them with +60 point bonus for closing distance
7. **Space-Based Trapping** - Calculates how each move reduces opponent's available space, rewards moves that corner them (up to +2 points per cell reduced)
8. **Center Control** - Dominates the center when strong (length > 6) for territorial advantage
9. **Strategic Priorities** - Eat early → grow long → hunt smaller snakes → trap them → control board

## Snake Appearance
- **Color**: Hot Pink (#FF0080)
- **Head**: Evil
- **Tail**: Curled

## Development
The server runs on port 5000 and is configured to work with Replit's environment. The workflow automatically starts the server when you run the project.

## Deployment
This project is configured for deployment on Replit. Once published, you can use the live URL to register your Battlesnake on [play.battlesnake.com](https://play.battlesnake.com).

## Strategy Notes
The AI uses a multi-layered approach:
- Eliminates unsafe moves (boundaries, collisions)
- Scores remaining moves based on available space (flood fill)
- Adjusts priorities based on health level
- Chooses the highest-scoring safe move each turn

See the [Battlesnake Documentation](https://docs.battlesnake.com) for more information.

## Recent Changes
- November 2, 2025: Initial setup for Replit environment
  - Configured to run on port 5000
  - Set up workflow for automatic server start
  - Implemented advanced AI with collision avoidance, flood fill, and intelligent food seeking
  - Ready for competitive gameplay and deployment

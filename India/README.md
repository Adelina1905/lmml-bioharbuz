# HOW THE BOT WORKS

## Overview

The bot uses the **minimax algorithm** to decide which moves to play.

## The Evaluate Function

The heart of the program is the **evaluate function**, which assigns point values to different board patterns and strategic positions. This allows the bot to prioritize the best moves. 

The minimax algorithm then uses this evaluation function to explore two possibilities:
- **The bot's turn** (maximizing score)
- **The opponent's turn** (minimizing our score)

By doing this, the bot always assumes the opponent will play their best possible move, and we will respond optimally. This guarantees we keep our disadvantage as low as possible while taking advantage of any mistakes the opponent makes.

## Alpha-Beta Pruning

The algorithm also uses **alpha-beta pruning**, which is a technique to skip evaluating moves that clearly can't be better than moves already found. This saves significant time, especially when there are many possible moves to consider.

## The Search Tree

The minimax function is **recursive** and uses **iterative deepening** - it searches progressively deeper (1 move ahead, then 2, then 3...) until time runs out. 

We can visualize this as a tree: each move is a branch, and from each branch grow more branches representing possible responses. The bot explores this tree and chooses the path (sequence of moves) with the highest evaluation score, assuming both players play perfectly.

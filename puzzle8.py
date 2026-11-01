"""
Task 1 - 8-Puzzle with A* search
Introduction to AI and Data Science

The program creates random start states, checks if they are solvable and
solves them with A* using two heuristics:
    h1 = Hamming distance (number of misplaced tiles)
    h2 = Manhattan distance (sum of the distances of all tiles to their goal)
For both heuristics the number of expanded nodes (memory effort) and the
run time are measured. At the end mean and standard deviation are printed.

Parts of the program:
    * puzzle functions (goal state, moves, printing)
    * solvability check
    * heuristics
    * A* search
    * experiment and statistics
    * main program

A state is stored as a tuple with 9 numbers (row by row), 0 is the blank.
Example: (7, 2, 4, 5, 0, 6, 8, 3, 1) means
    7 2 4
    5 _ 6
    8 3 1
"""

import heapq
import random
import statistics
import time

# settings
NUMBER_OF_PUZZLES = 100
SEED = 42                  # same seed -> same random puzzles

# ---------------------------------------------------------------------------
# Puzzle functions
# ---------------------------------------------------------------------------

def get_goal_state():
    """
    Function: returns the goal state from the lecture (blank top left)
    Input:    -
    Output:   tuple (0, 1, 2, 3, 4, 5, 6, 7, 8)
    """
    return (0, 1, 2, 3, 4, 5, 6, 7, 8)


def get_successors(state):
    """
    Function: creates all states that can be reached with one move.
              The blank is swapped with the tile above, below, left or right.
              Every move costs 1.
    Input:    state (tuple with 9 numbers)
    Output:   list of successor states (2 to 4 states)
    """
    blank = state.index(0)
    row = blank // 3
    col = blank % 3

    # positions the blank can move to
    moves = []
    if row > 0:
        moves.append(blank - 3)  # up
    if row < 2:
        moves.append(blank + 3)  # down
    if col > 0:
        moves.append(blank - 1)  # left
    if col < 2:
        moves.append(blank + 1)  # right

    successors = []
    for new_pos in moves:
        new_state = list(state)
        new_state[blank] = new_state[new_pos]
        new_state[new_pos] = 0
        successors.append(tuple(new_state))
    return successors


def random_state(rng):
    """
    Function: creates a random state by shuffling the numbers 0-8.
              About half of these states are not solvable, so the state
              has to be checked with is_solvable() first.
    Input:    rng (random.Random object)
    Output:   tuple with 9 numbers
    """
    numbers = list(range(9))
    rng.shuffle(numbers)
    return tuple(numbers)


def print_state(state):
    """
    Function: prints a state as 3x3 board, the blank is shown as _
    Input:    state
    Output:   - (prints to console)
    """
    for row in range(3):
        line = ""
        for col in range(3):
            tile = state[row * 3 + col]
            if tile == 0:
                line += "_ "
            else:
                line += str(tile) + " "
        print(line.strip())


def print_solution(path):
    """
    Function: prints the solution step by step. For every step the moved
              tile and the board after the move are shown.
    Input:    path (list of states from start to goal)
    Output:   - (prints to console)
    """
    print("Start:")
    print_state(path[0])
    for i in range(1, len(path)):
        # the moved tile is now where the blank was before
        moved_tile = path[i][path[i - 1].index(0)]
        print("\nStep", i, "- move tile", moved_tile)
        print_state(path[i])
    print("\nGoal reached in", len(path) - 1, "moves")


# ---------------------------------------------------------------------------
# Solvability check
# ---------------------------------------------------------------------------

def count_inversions(state):
    """
    Function: counts the inversions of a state. An inversion is a pair of
              tiles where the bigger number comes before the smaller one.
              The blank (0) is not counted.
    Input:    state
    Output:   number of inversions
    """
    tiles = [t for t in state if t != 0]
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions


def is_solvable(state, goal):
    """
    Function: checks if the goal can be reached from the state.
              On a 3x3 board a move never changes whether the number of
              inversions is even or odd (horizontal move: no change,
              vertical move: the tile jumps over 2 other tiles, so the
              number changes by -2, 0 or +2). That means the state is
              solvable if it has the same parity as the goal.
    Input:    state, goal
    Output:   True or False
    """
    return count_inversions(state) % 2 == count_inversions(goal) % 2


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
# Both heuristics never overestimate the real number of moves (admissible),
# so A* finds the optimal solution with both of them.

def hamming(state, goal):
    """
    Function: heuristic h1, counts the tiles that are not on their goal
              position (the blank is not counted)
    Input:    state, goal
    Output:   number of misplaced tiles
    """
    count = 0
    for i in range(9):
        if state[i] != 0 and state[i] != goal[i]:
            count += 1
    return count


def manhattan(state, goal):
    """
    Function: heuristic h2, sum of the horizontal and vertical distances
              of every tile to its goal position (the blank is not counted)
    Input:    state, goal
    Output:   sum of the distances
    """
    distance = 0
    for i in range(9):
        tile = state[i]
        if tile != 0:
            goal_pos = goal.index(tile)
            distance += abs(i // 3 - goal_pos // 3) + abs(i % 3 - goal_pos % 3)
    return distance


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------

def a_star(start, goal, heuristic):
    """
    Function: A* search. Always expands the node with the lowest
              f(n) = g(n) + h(n)  (g = moves so far, h = heuristic).
    Input:    start, goal, heuristic (hamming or manhattan)
    Output:   (path, expanded)
              path = list of states from start to goal (None if no solution)
              expanded = number of expanded nodes
    """
    # entries in the open list: (f, h, g, state)
    # if two nodes have the same f, the one with the smaller h is taken first
    h_start = heuristic(start, goal)
    open_list = [(h_start, h_start, 0, start)]
    g_costs = {start: 0}
    parents = {start: None}
    closed = set()
    expanded = 0

    while open_list:
        f, h, g, state = heapq.heappop(open_list)

        # a state can be in the open list more than once,
        # so skip it if it was already expanded
        if state in closed:
            continue

        if state == goal:
            # follow the parents back to the start to get the path
            path = []
            while state is not None:
                path.append(state)
                state = parents[state]
            path.reverse()
            return path, expanded

        closed.add(state)
        expanded += 1

        for next_state in get_successors(state):
            new_g = g + 1
            # only add the state if it is new or we found a shorter way to it
            if next_state not in g_costs or new_g < g_costs[next_state]:
                g_costs[next_state] = new_g
                parents[next_state] = state
                new_h = heuristic(next_state, goal)
                heapq.heappush(open_list, (new_g + new_h, new_h, new_g, next_state))

    return None, expanded


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def create_puzzles(n, goal, rng):
    """
    Function: creates n random solvable start states. Unsolvable states
              are thrown away and counted.
    Input:    n, goal, rng
    Output:   (list of start states, number of unsolvable states)
    """
    puzzles = []
    unsolvable = 0
    while len(puzzles) < n:
        state = random_state(rng)
        if is_solvable(state, goal):
            puzzles.append(state)
        else:
            unsolvable += 1
    return puzzles, unsolvable


def run_experiment(puzzles, goal, heuristics):
    """
    Function: solves every puzzle with every heuristic and measures the
              expanded nodes and the run time. Both heuristics get the same
              puzzles, so the comparison is fair.
    Input:    puzzles, goal, heuristics (dict: name -> function)
    Output:   list of results (one dict per puzzle and heuristic)
    """
    results = []
    for i, start in enumerate(puzzles):
        moves_per_heuristic = []
        for name, heuristic in heuristics.items():
            t_start = time.perf_counter()
            path, expanded = a_star(start, goal, heuristic)
            t_end = time.perf_counter()

            moves = len(path) - 1
            moves_per_heuristic.append(moves)
            results.append({
                "puzzle": i + 1,
                "heuristic": name,
                "moves": moves,
                "expanded": expanded,
                "time_ms": (t_end - t_start) * 1000,
            })

        # both heuristics are admissible, so both solutions must be equally long
        if len(set(moves_per_heuristic)) > 1:
            print("Warning: different solution lengths for puzzle", i + 1)

        if (i + 1) % 10 == 0:
            print(i + 1, "of", len(puzzles), "puzzles solved")
    return results

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(results, heuristics):
    """
    Function: prints mean and standard deviation of the expanded nodes,
              the run time and the solution length for every heuristic
    Input:    results, heuristics
    Output:   - (prints to console)
    """
    for name in heuristics:
        expanded = [r["expanded"] for r in results if r["heuristic"] == name]
        times = [r["time_ms"] for r in results if r["heuristic"] == name]
        moves = [r["moves"] for r in results if r["heuristic"] == name]

        print("\n" + name + ":")
        print(f"  expanded nodes:  mean = {statistics.mean(expanded):10.1f}"
              f"   std = {statistics.stdev(expanded):10.1f}")
        print(f"  run time [ms]:   mean = {statistics.mean(times):10.1f}"
              f"   std = {statistics.stdev(times):10.1f}")
        print(f"  solution length: mean = {statistics.mean(moves):10.1f}"
              f"   std = {statistics.stdev(moves):10.1f}")

# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    """
    Function: shows one example with its solution, then runs the experiment
              with the random puzzles and prints the statistics
    Input:    - (uses the settings at the top)
    Output:   - (prints to console)
    """
    rng = random.Random(SEED)
    goal = get_goal_state()
    heuristics = {"Hamming": hamming, "Manhattan": manhattan}

    print("Goal state:")
    print_state(goal)

    # example puzzle
    example = create_puzzles(1, goal, rng)[0][0]

    print("\nExample start state:")
    print_state(example)
    print("Inversions:", count_inversions(example))

    if not is_solvable(example, goal):
        print("This state is not solvable, so there is no solution.")
    else:
        for name, heuristic in heuristics.items():
            path, expanded = a_star(example, goal, heuristic)
            print(f"{name}: h(start) = {heuristic(example, goal)}, "
                  f"{len(path) - 1} moves, {expanded} expanded nodes")
        print("\nSolution:")
        print_solution(path)

    # experiment
    print("\nStarting experiment with", NUMBER_OF_PUZZLES, "random puzzles ...")
    puzzles, unsolvable = create_puzzles(NUMBER_OF_PUZZLES, goal, rng)
    print("Unsolvable states thrown away:", unsolvable)

    results = run_experiment(puzzles, goal, heuristics)
    print_statistics(results, heuristics)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
8-Puzzle solver with A* search -- Task 1, Introduction to AI and Data Science.

The script
  1. generates random start states,
  2. checks each of them for solvability,
  3. solves the solvable ones with A* using two heuristics
     and prints the solution of one example move by move
     (Hamming = misplaced tiles, Manhattan = sum of tile distances),
  4. measures memory effort (number of expanded nodes) and run time,
  5. prints mean and standard deviation for each heuristic.

Usage:
    python puzzle8.py                 # 100 random puzzles, seed 42
    python puzzle8.py --n 100 --seed 7 --csv results.csv
    python puzzle8.py --start 724506831   # show the solution of an own state
    python puzzle8.py --no-steps          # only the statistics

Software structure (one section per submodule):
    1. Puzzle model     - state representation, goal state, successor function,
                          text output of states and solutions
    2. Solvability      - inversion count / parity check
    3. Heuristics       - Hamming and Manhattan
    4. Search           - A* algorithm
    5. Experiment       - random states, measurements for each heuristic
    6. Statistics       - mean / standard deviation, effective branching factor
    7. Main             - command line interface and report output

State representation:
    A state is a tuple of 9 integers in row-major order, 0 is the blank.
    Example: (7, 2, 4, 5, 0, 6, 8, 3, 1) is the board
        7 2 4
        5 _ 6
        8 3 1
    Tuples are immutable and hashable, so they can be used directly as
    dictionary keys / set members in the search.
"""

import argparse
import csv
import heapq
import random
import statistics
import time

# =============================================================================
# 1. PUZZLE MODEL
# =============================================================================

SIZE = 3                    # board is SIZE x SIZE
N_CELLS = SIZE * SIZE       # 9 cells, tiles 1..8 plus the blank (0)


def generate_goal_state():
    """
    Function: Builds the goal state used in the lecture (blank in the top-left
              corner, tiles in ascending order).
    Input:    none
    Output:   tuple of 9 ints -> (0, 1, 2, 3, 4, 5, 6, 7, 8)
    """
    return tuple(range(N_CELLS))


def build_neighbor_table():
    """
    Function: Precomputes, for every blank position, the positions the blank
              can move to (up, down, left, right). Computing this once saves
              time because the successor function is called very often.
    Input:    none
    Output:   list of 9 tuples; entry i holds the indices adjacent to cell i
    """
    table = []
    for index in range(N_CELLS):
        row, col = divmod(index, SIZE)
        neighbors = []
        if row > 0:
            neighbors.append(index - SIZE)      # blank moves up
        if row < SIZE - 1:
            neighbors.append(index + SIZE)      # blank moves down
        if col > 0:
            neighbors.append(index - 1)         # blank moves left
        if col < SIZE - 1:
            neighbors.append(index + 1)         # blank moves right
        table.append(tuple(neighbors))
    return table


NEIGHBORS = build_neighbor_table()


def get_successors(state):
    """
    Function: Transition model. Generates all states reachable with one move
              (sliding one tile into the blank). Every move has cost 1.
    Input:    state - tuple of 9 ints
    Output:   list of successor states (2 to 4 tuples of 9 ints)
    """
    blank = state.index(0)
    successors = []
    for target in NEIGHBORS[blank]:
        cells = list(state)
        cells[blank], cells[target] = cells[target], cells[blank]
        successors.append(tuple(cells))
    return successors


def generate_random_state(rng):
    """
    Function: Creates a random permutation of the 9 cells. About half of all
              permutations are NOT solvable, therefore the result has to be
              checked with is_solvable() before searching.
    Input:    rng - random.Random instance (allows reproducible experiments)
    Output:   tuple of 9 ints (random state, possibly unsolvable)
    """
    cells = list(range(N_CELLS))
    rng.shuffle(cells)
    return tuple(cells)


def format_state(state):
    """
    Function: Turns a state into a readable 3x3 text block.
    Input:    state - tuple of 9 ints
    Output:   string with 3 lines, blank shown as '_'
    """
    lines = []
    for row in range(SIZE):
        cells = state[row * SIZE:(row + 1) * SIZE]
        lines.append(" ".join("_" if c == 0 else str(c) for c in cells))
    return "\n".join(lines)


def describe_move(before, after):
    """
    Function: Describes the move that leads from one state to the next one in
              words, seen from the tile that slides into the blank.
    Input:    before - state before the move, after - state after the move
    Output:   string such as "tile 5 moves right"
    """
    blank_before = before.index(0)
    blank_after = after.index(0)
    tile = before[blank_after]          # this tile slides into the blank
    step = blank_before - blank_after   # direction of the tile on the board
    direction = {-SIZE: "up", SIZE: "down", -1: "left", 1: "right"}[step]
    return f"tile {tile} moves {direction}"


def print_solution(path):
    """
    Function: Prints a solution move by move. Every step shows the move in
              words and the board after that move, so the way from the start
              state to the goal state can be followed (and replayed by hand).
    Input:    path - list of states from start to goal (output of a_star)
    Output:   none (text on the console)
    """
    print("Start:")
    print(format_state(path[0]))
    for number in range(1, len(path)):
        move = describe_move(path[number - 1], path[number])
        print(f"\nStep {number} of {len(path) - 1}: {move}")
        print(format_state(path[number]))
    print(f"\nGoal reached after {len(path) - 1} moves.")


def parse_state(text):
    """
    Function: Converts a user input like "724506831" into a state and checks
              that it contains every digit 0..8 exactly once.
    Input:    text - string of 9 digits, 0 is the blank
    Output:   tuple of 9 ints; raises ValueError for invalid input
    """
    if len(text) != N_CELLS or sorted(text) != [str(d) for d in range(N_CELLS)]:
        raise ValueError("start state must contain the digits 0-8 exactly "
                         "once, e.g. 724506831")
    return tuple(int(character) for character in text)


# =============================================================================
# 2. SOLVABILITY CHECK
# =============================================================================

def count_inversions(state):
    """
    Function: Counts the inversions of a state. An inversion is a pair of
              tiles (a, b) where a appears before b but a > b. The blank is
              ignored.
    Input:    state - tuple of 9 ints
    Output:   int, number of inversions (0..28)
    """
    tiles = [tile for tile in state if tile != 0]
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions


def is_solvable(state, goal):
    """
    Function: Decides whether the goal can be reached from the state.
              On a board with odd width a move never changes the parity of the
              inversion count: a horizontal move changes nothing, a vertical
              move jumps a tile over exactly 2 others (change of -2, 0 or +2).
              So a state is solvable exactly when its inversion parity equals
              the inversion parity of the goal.
    Input:    state - tuple of 9 ints, goal - tuple of 9 ints
    Output:   True if solvable, False otherwise
    """
    return count_inversions(state) % 2 == count_inversions(goal) % 2


# =============================================================================
# 3. HEURISTICS
# =============================================================================
# Both heuristics are admissible (they never overestimate the true cost), so
# A* finds an optimal solution with either of them. Manhattan >= Hamming for
# every state, i.e. Manhattan dominates Hamming.

def build_distance_table(goal):
    """
    Function: Precomputes the Manhattan distance of every tile from every
              board position to its goal position. Turns the heuristic into
              simple table look-ups.
    Input:    goal - tuple of 9 ints
    Output:   9x9 list; table[tile][position] = horizontal + vertical distance
    """
    table = [[0] * N_CELLS for _ in range(N_CELLS)]
    for tile in range(1, N_CELLS):
        goal_row, goal_col = divmod(goal.index(tile), SIZE)
        for position in range(N_CELLS):
            row, col = divmod(position, SIZE)
            table[tile][position] = abs(row - goal_row) + abs(col - goal_col)
    return table


def make_hamming(goal):
    """
    Function: Creates the Hamming heuristic h1 for a given goal.
              h1(n) = number of tiles that are not on their goal position
              (relaxed problem: a tile may jump to any square in one move).
    Input:    goal - tuple of 9 ints
    Output:   function h(state) -> int
    """
    def hamming(state):
        misplaced = 0
        for position in range(N_CELLS):
            tile = state[position]
            if tile != 0 and tile != goal[position]:
                misplaced += 1
        return misplaced
    return hamming


def make_manhattan(goal):
    """
    Function: Creates the Manhattan heuristic h2 for a given goal.
              h2(n) = sum of horizontal + vertical distances of all tiles to
              their goal positions (relaxed problem: a tile may move to an
              adjacent square even if it is occupied).
    Input:    goal - tuple of 9 ints
    Output:   function h(state) -> int
    """
    table = build_distance_table(goal)

    def manhattan(state):
        distance = 0
        for position in range(N_CELLS):
            tile = state[position]
            if tile != 0:
                distance += table[tile][position]
        return distance
    return manhattan


# =============================================================================
# 4. SEARCH (A*)
# =============================================================================

def reconstruct_path(parents, state):
    """
    Function: Follows the parent links back from the goal to the start.
    Input:    parents - dict {state: parent state or None}, state - goal state
    Output:   list of states from start to goal (inclusive)
    """
    path = []
    while state is not None:
        path.append(state)
        state = parents[state]
    path.reverse()
    return path


def a_star(start, goal, heuristic):
    """
    Function: A* search. Always expands the frontier node with the smallest
              f(n) = g(n) + h(n), where g(n) is the path cost from the start
              and h(n) the heuristic estimate to the goal.
    Input:    start     - tuple of 9 ints (must be solvable)
              goal      - tuple of 9 ints
              heuristic - function h(state) -> int (admissible)
    Output:   dict with
                path      - list of states from start to goal
                moves     - solution length (number of moves)
                expanded  - number of expanded nodes (memory effort measure)
                generated - number of nodes put on the frontier
                stored    - number of distinct states kept in memory at the end
              or None if no solution exists.
    """
    # Frontier entries: (f, h, counter, g, state).
    # Ties in f are broken by smaller h (prefer nodes closer to the goal),
    # then by insertion order, so states themselves are never compared.
    counter = 0
    start_h = heuristic(start)
    frontier = [(start_h, start_h, counter, 0, start)]

    best_g = {start: 0}         # cheapest known path cost for every seen state
    parents = {start: None}     # for path reconstruction
    closed = set()              # states that were already expanded
    expanded = 0
    generated = 1

    while frontier:
        _, _, _, g, state = heapq.heappop(frontier)

        if state in closed:
            continue            # outdated duplicate entry, already expanded

        if state == goal:
            path = reconstruct_path(parents, state)
            return {
                "path": path,
                "moves": len(path) - 1,
                "expanded": expanded,
                "generated": generated,
                "stored": len(best_g),
            }

        closed.add(state)
        expanded += 1

        for successor in get_successors(state):
            new_g = g + 1
            if successor in best_g and best_g[successor] <= new_g:
                continue        # we already know a path that is at least as good
            best_g[successor] = new_g
            parents[successor] = state
            h = heuristic(successor)
            counter += 1
            heapq.heappush(frontier, (new_g + h, h, counter, new_g, successor))
            generated += 1

    return None


# =============================================================================
# 5. EXPERIMENT
# =============================================================================

def generate_solvable_states(count, goal, rng):
    """
    Function: Draws random states until 'count' solvable ones are found.
              Unsolvable states are counted and discarded.
    Input:    count - number of solvable states wanted
              goal  - tuple of 9 ints
              rng   - random.Random instance
    Output:   (list of solvable states, number of rejected unsolvable states)
    """
    states = []
    rejected = 0
    while len(states) < count:
        state = generate_random_state(rng)
        if is_solvable(state, goal):
            states.append(state)
        else:
            rejected += 1
    return states, rejected


def run_experiment(states, goal, heuristics):
    """
    Function: Solves every start state with every heuristic and records the
              measurements. All heuristics get the SAME start states, so the
              comparison is fair.
    Input:    states     - list of solvable start states
              goal       - tuple of 9 ints
              heuristics - dict {name: heuristic function}
    Output:   list of dicts, one per (puzzle, heuristic), with keys
              puzzle, start, heuristic, moves, expanded, generated, stored,
              time_s
    """
    records = []
    for number, start in enumerate(states, start=1):
        for name, heuristic in heuristics.items():
            begin = time.perf_counter()
            result = a_star(start, goal, heuristic)
            elapsed = time.perf_counter() - begin
            records.append({
                "puzzle": number,
                "start": "".join(str(c) for c in start),
                "heuristic": name,
                "moves": result["moves"],
                "expanded": result["expanded"],
                "generated": result["generated"],
                "stored": result["stored"],
                "time_s": elapsed,
            })
    return records


def check_optimality(records, heuristics):
    """
    Function: Sanity test. Both heuristics are admissible, so they must find
              solutions of identical (optimal) length for the same puzzle.
    Input:    records    - output of run_experiment
              heuristics - dict {name: heuristic function}
    Output:   number of puzzles where the solution lengths differ (expected 0)
    """
    lengths = {}
    for record in records:
        lengths.setdefault(record["puzzle"], set()).add(record["moves"])
    return sum(1 for moves in lengths.values() if len(moves) > 1)


# =============================================================================
# 6. STATISTICS
# =============================================================================

def effective_branching_factor(generated, depth):
    """
    Function: Estimates the effective branching factor b*. b* is the branching
              factor a uniform tree of the given depth would need in order to
              contain N + 1 nodes:  N + 1 = 1 + b* + (b*)^2 + ... + (b*)^d.
              Solved numerically by bisection.
    Input:    generated - N, number of nodes generated by the search
              depth     - d, solution length
    Output:   float b* (0.0 if depth is 0)
    """
    if depth == 0:
        return 0.0
    target = generated + 1

    def tree_size(b):
        return sum(b ** level for level in range(depth + 1))

    low, high = 1.0, float(max(generated, 2))
    for _ in range(100):
        middle = (low + high) / 2
        if tree_size(middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def summarize(records, name):
    """
    Function: Computes mean and (sample) standard deviation of the
              measurements of one heuristic.
    Input:    records - output of run_experiment
              name    - heuristic name to summarize
    Output:   dict {measure: (mean, standard deviation)} for the measures
              expanded, generated, stored, time_ms, moves, b_star
    """
    rows = [r for r in records if r["heuristic"] == name]
    series = {
        "expanded": [r["expanded"] for r in rows],
        "generated": [r["generated"] for r in rows],
        "stored": [r["stored"] for r in rows],
        "time_ms": [r["time_s"] * 1000 for r in rows],
        "moves": [r["moves"] for r in rows],
        "b_star": [effective_branching_factor(r["generated"], r["moves"])
                   for r in rows if r["moves"] > 0],
    }
    summary = {}
    for measure, values in series.items():
        mean = statistics.mean(values)
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[measure] = (mean, deviation)
    return summary


def print_report(summaries, puzzle_count, rejected, mismatches):
    """
    Function: Prints the comparison table of all heuristics.
    Input:    summaries    - dict {heuristic name: output of summarize}
              puzzle_count - number of solved puzzles
              rejected     - number of discarded unsolvable random states
              mismatches   - output of check_optimality
    Output:   none (text on the console)
    """
    labels = [
        ("expanded", "Expanded nodes (memory effort)"),
        ("generated", "Generated nodes"),
        ("stored", "States stored in memory"),
        ("time_ms", "Run time [ms]"),
        ("moves", "Solution length [moves]"),
        ("b_star", "Effective branching factor b*"),
    ]
    names = list(summaries)

    print(f"\nSolved puzzles: {puzzle_count}   "
          f"(unsolvable random states discarded: {rejected})")
    print(f"Puzzles with different solution length per heuristic: {mismatches}"
          f"   (0 = all solutions optimal)\n")

    header = f"{'Measure (mean +/- std)':<32}"
    for name in names:
        header += f"{name:>26}"
    print(header)
    print("-" * len(header))
    for measure, label in labels:
        line = f"{label:<32}"
        for name in names:
            mean, deviation = summaries[name][measure]
            line += f"{f'{mean:,.2f} +/- {deviation:,.2f}':>26}"
        print(line)

    if len(names) == 2:
        first, second = names
        ratio_nodes = (summaries[first]["expanded"][0]
                       / summaries[second]["expanded"][0])
        ratio_time = (summaries[first]["time_ms"][0]
                      / summaries[second]["time_ms"][0])
        print(f"\n{first} expands {ratio_nodes:.1f}x as many nodes and needs "
              f"{ratio_time:.1f}x the run time of {second}.")


def write_csv(records, filename):
    """
    Function: Saves the raw measurements, e.g. for diagrams in the report.
    Input:    records  - output of run_experiment
              filename - path of the CSV file to create
    Output:   none (file on disk)
    """
    with open(filename, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


# =============================================================================
# 7. MAIN
# =============================================================================

def parse_arguments():
    """
    Function: Reads the command line options.
    Input:    none (uses sys.argv)
    Output:   argparse.Namespace with n, seed, csv, start, no_steps
    """
    parser = argparse.ArgumentParser(description="8-puzzle: A* with Hamming "
                                                 "and Manhattan heuristic")
    parser.add_argument("--n", type=int, default=100,
                        help="number of random solvable start states")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (same seed = same puzzles)")
    parser.add_argument("--csv", default=None,
                        help="optional file name for the raw measurements")
    parser.add_argument("--start", default=None,
                        help="own start state for the example, 9 digits with "
                             "0 as blank, e.g. 724506831")
    parser.add_argument("--no-steps", action="store_true",
                        help="do not print the solution of the example "
                             "move by move")
    return parser.parse_args()


def main():
    """
    Function: Runs the whole task: goal state, one worked example with its
              solution move by move, the experiment with n random states and
              the final report.
    Input:    none
    Output:   none (text on the console, optional CSV file)
    """
    args = parse_arguments()
    rng = random.Random(args.seed)

    goal = generate_goal_state()
    heuristics = {
        "Hamming": make_hamming(goal),
        "Manhattan": make_manhattan(goal),
    }

    print("Goal state:")
    print(format_state(goal))

    # --- one worked example -------------------------------------------------
    if args.start:
        try:
            example = parse_state(args.start)
        except ValueError as error:
            print(f"\nInvalid --start value: {error}")
            return
    else:
        example = generate_solvable_states(1, goal, rng)[0][0]

    print("\nExample start state:")
    print(format_state(example))
    solvable = is_solvable(example, goal)
    print(f"Inversions: {count_inversions(example)}  ->  solvable: {solvable}")
    if not solvable:
        print("This state cannot reach the goal, so there is nothing to "
              "search. Choose another start state.")
        return

    solution = None
    for name, heuristic in heuristics.items():
        result = a_star(example, goal, heuristic)
        solution = result["path"]
        print(f"  {name:<10} h(start) = {heuristic(example):>2}, "
              f"solution = {result['moves']} moves, "
              f"expanded nodes = {result['expanded']}")

    if not args.no_steps:
        print("\nSolution of the example, move by move:")
        print_solution(solution)

    # --- experiment ---------------------------------------------------------
    print(f"\nRunning experiment with {args.n} random states "
          f"(seed {args.seed}) ...")
    states, rejected = generate_solvable_states(args.n, goal, rng)
    records = run_experiment(states, goal, heuristics)
    mismatches = check_optimality(records, heuristics)

    summaries = {name: summarize(records, name) for name in heuristics}
    print_report(summaries, len(states), rejected, mismatches)

    if args.csv:
        write_csv(records, args.csv)
        print(f"\nRaw measurements written to {args.csv}")


if __name__ == "__main__":
    main()
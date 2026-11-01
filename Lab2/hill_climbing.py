# Hill Climbing Search (Greedy Local Search using Heuristics)

# Graph connections
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

# Heuristic values h(n): estimated distance to Goal 'F'
# (Lower heuristic value means closer to the goal)
heuristics = {"A": 6, "B": 4, "C": 2, "D": 7, "E": 3, "F": 0}


def hill_climbing(graph, heuristics, start_node, goal_node):
    current = start_node
    path = [current]
    steps = 0

    while current != goal_node:
        neighbors = graph.get(current, [])
        if not neighbors:
            break

        # Find the neighbor with the minimum heuristic value
        best_neighbor = min(neighbors, key=lambda node: heuristics[node])

        # If the best neighbor is NOT better than the current node, stop (local minimum)
        if heuristics[best_neighbor] >= heuristics[current]:
            print(
                f"\n[Terminated] Local optimum reached at node {current} (no"
                " better neighbor)."
            )
            break

        # Move to the better neighbor
        current = best_neighbor
        path.append(current)
        steps += 1

    reached = current == goal_node
    return path, reached, steps


def main():
    print("--- Graph Nodes and Heuristic Values h(n) ---")
    for node, neighbors in graph.items():
        print(
            f"Node {node} (h={heuristics[node]}) ->"
            f" Neighbors: {', '.join(neighbors)}"
        )
    print()

    try:
        start = input("Enter start node [default A]: ").strip().upper() or "A"
        goal = input("Enter goal node [default F]: ").strip().upper() or "F"

        if start not in graph or goal not in graph:
            print(f"Error: Node must be one of {list(graph.keys())}")
            return

        path, goal_reached, steps = hill_climbing(
            graph, heuristics, start, goal
        )

        print("\n--- Search Details ---")
        print(f"Start Node   : {start}")
        print(f"Goal Node    : {goal}")
        print(
            f"Path Taken   :"
            f" {' -> '.join(f'{node}(h={heuristics[node]})' for node in path)}"
        )

        print("\n--- Performance Summary ---")
        print(f"Goal Reached : {goal_reached}")
        print(f"Steps Taken  : {steps}")
        print(f"Final Node   : {path[-1]} (h={heuristics[path[-1]]})")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

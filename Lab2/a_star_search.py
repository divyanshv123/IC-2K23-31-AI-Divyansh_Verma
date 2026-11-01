import heapq

# A* Search Algorithm: f(n) = g(n) + h(n)

# Graph with edge weights: (neighbor, edge_cost)
graph = {
    "A": [("B", 2), ("C", 4)],
    "B": [("A", 2), ("D", 3), ("E", 1)],
    "C": [("A", 4), ("F", 5)],
    "D": [("B", 3)],
    "E": [("B", 1), ("F", 3)],
    "F": [("C", 5), ("E", 3)],
}

# Heuristic values h(n): estimated distance to goal F
h = {"A": 6, "B": 4, "C": 2, "D": 7, "E": 3, "F": 0}


def a_star_search(start, goal):
    # Priority queue stores: (f_score, g_cost, current_node, path)
    pq = [(h[start], 0, start, [start])]
    visited = set()
    explored = 0

    while pq:
        f, g, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        explored += 1

        if current == goal:
            return path, g, explored

        for neighbor, weight in graph.get(current, []):
            if neighbor not in visited:
                new_g = g + weight
                new_f = new_g + h[neighbor]
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))

    return None, 0, explored


def main():
    start = input("Enter start node [default A]: ").strip().upper() or "A"
    goal = input("Enter goal node [default F]: ").strip().upper() or "F"

    if start not in graph or goal not in graph:
        print("Invalid node entered!")
        return

    path, total_cost, count = a_star_search(start, goal)

    if path:
        print("\nPath found:", " -> ".join(path))
        print("Total path cost:", total_cost)
        print("Total nodes explored:", count)
    else:
        print("\nNo path found!")


if __name__ == "__main__":
    main()

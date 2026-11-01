import heapq

# Best-First Search

graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

# Heuristic values h(n)
h = {"A": 6, "B": 4, "C": 2, "D": 7, "E": 3, "F": 0}


def best_first_search(start, goal):
    pq = [(h[start], start, [start])]
    visited = set()
    explored = 0

    while pq:
        cost, current, path = heapq.heappop(pq)

        if current not in visited:
            visited.add(current)
            explored += 1

            if current == goal:
                return path, explored

            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    heapq.heappush(
                        pq, (h[neighbor], neighbor, path + [neighbor])
                    )

    return None, explored


def main():
    start = input("Enter start node [default A]: ").strip().upper() or "A"
    goal = input("Enter goal node [default F]: ").strip().upper() or "F"

    if start not in graph or goal not in graph:
        print("Invalid node entered!")
        return

    path, count = best_first_search(start, goal)

    if path:
        print("\nPath found:", " -> ".join(path))
        print("Number of edges:", len(path) - 1)
        print("Total nodes explored:", count)
    else:
        print("\nNo path found!")


if __name__ == "__main__":
    main()

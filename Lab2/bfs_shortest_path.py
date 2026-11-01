from collections import deque

# Breadth-First Search (BFS) for Shortest Path in a Graph

# Graph represented as an adjacency list
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}


def bfs_shortest_path(graph, start, goal):
    if start == goal:
        return [start], 1

    visited = set([start])
    queue = deque([(start, [start])])
    nodes_explored = 0

    while queue:
        current, path = queue.popleft()
        nodes_explored += 1

        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                if neighbor == goal:
                    return path + [neighbor], nodes_explored + 1

                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, nodes_explored


def main():
    print("Available nodes in graph:", list(graph.keys()))

    try:
        start_node = (
            input("Enter Start Node [default A]: ").strip().upper() or "A"
        )
        end_node = (
            input("Enter End Node   [default F]: ").strip().upper() or "F"
        )

        if start_node not in graph or end_node not in graph:
            print(f"Error: Nodes must be from {list(graph.keys())}")
            return

        path, explored_count = bfs_shortest_path(graph, start_node, end_node)

        print("\n--- Search Details ---")
        print(f"Start Node   : {start_node}")
        print(f"End Node     : {end_node}")
        if path:
            print(f"Path Found   : {' -> '.join(path)}")
        else:
            print("Path Found   : No path exists")

        print("\n--- Performance Summary ---")
        print(f"Goal Reached   : {path is not None}")
        print(f"Path Length    : {len(path) - 1 if path else 0} edges")
        print(f"Nodes Explored : {explored_count}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

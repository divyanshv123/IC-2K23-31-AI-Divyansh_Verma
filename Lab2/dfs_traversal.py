# Depth-First Search (DFS) for Graph Traversal and Path Finding

# Graph represented as an adjacency list
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}


def dfs_traversal(graph, start_node):
    # Visit nodes deeply along each branch using a LIFO stack
    visited = set()
    stack = [start_node]
    order = []

    while stack:
        current = stack.pop()

        if current not in visited:
            visited.add(current)
            order.append(current)

            # Push in reverse so the leftmost neighbor is processed first
            for neighbor in reversed(graph.get(current, [])):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order


def dfs_path(graph, start_node, goal_node):
    # Finds a path from start to goal using DFS
    if start_node == goal_node:
        return [start_node]

    visited = set()
    # Stack stores tuples of: (current_node, path_so_far)
    stack = [(start_node, [start_node])]

    while stack:
        current, path = stack.pop()

        if current not in visited:
            visited.add(current)

            if current == goal_node:
                return path

            for neighbor in reversed(graph.get(current, [])):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    return None


def main():
    print("--- Graph Adjacency List ---")
    for node, neighbors in graph.items():
        print(f"Node {node} -> {', '.join(neighbors)}")
    print()

    try:
        start = input("Enter start node [default A]: ").strip().upper() or "A"
        goal = input("Enter goal node [default F]: ").strip().upper() or "F"

        if start not in graph or goal not in graph:
            print(f"Error: Node must be one of {list(graph.keys())}")
            return

        # 1. Full DFS Traversal
        traversal = dfs_traversal(graph, start)
        print(f"\nDFS Traversal Order from {start}:")
        print(" -> ".join(traversal))

        # 2. Path found using DFS
        path = dfs_path(graph, start, goal)
        if path:
            print(f"\nDFS Path from {start} to {goal}:")
            print(" -> ".join(path))
            print(f"Number of edges: {len(path) - 1}")
        else:
            print(f"\nNo path exists from {start} to {goal}.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

import itertools

# Generate and Test Search Strategy (4-Queens Problem)


def test(board):
    # Check if any two queens attack each other diagonally
    n = len(board)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(board[i] - board[j]) == abs(i - j):
                return False
    return True


def generate(n):
    # Generates all candidate configurations (one queen per row and column)
    return itertools.permutations(range(n))


def solve_n_queens(n=4):
    total_generated = 0
    solutions = []

    # Generate candidate states and test each one
    for candidate in generate(n):
        total_generated += 1
        if test(candidate):
            solutions.append(candidate)

    return solutions, total_generated


def print_board(solution):
    n = len(solution)
    for row in range(n):
        row_display = [" . "] * n
        row_display[solution[row]] = " Q "
        print("".join(row_display))
    print()


def main():
    try:
        user_input = input("Enter board size N [default 4]: ").strip()
        n = int(user_input) if user_input else 4

        if n <= 0:
            print("Board size must be a positive integer.")
            return
        if n in (2, 3):
            print(f"No solution exists for N = {n}.")
            return

        print(
            f"\nRunning Generate and Test for N = {n} (Total possible states:"
            f" {n}!)...\n"
        )
        solutions, count = solve_n_queens(n)

        print(f"Total states generated & tested : {count}")
        print(f"Total valid solutions found      : {len(solutions)}\n")

        # Display first 2 solutions
        for idx, sol in enumerate(solutions[:2], 1):
            print(f"Solution {idx} (Column index per row: {sol}):")
            print_board(sol)

    except ValueError:
        print("Invalid input! Please enter a valid number.")


if __name__ == "__main__":
    main()

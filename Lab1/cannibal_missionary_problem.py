from collections import deque


def check_safe_state(m_bank, c_bank, total_m, total_c):
    """Verifies that cannibals never outnumber missionaries on either bank."""
    other_m = total_m - m_bank
    other_c = total_c - c_bank

    # Bounds check
    if not (0 <= m_bank <= total_m and 0 <= c_bank <= total_c):
        return False

    # Left shore safety
    if m_bank > 0 and c_bank > m_bank:
        return False

    # Right shore safety
    if other_m > 0 and other_c > other_m:
        return False

    return True


def get_boat_combinations(max_m, max_c, boat_limit):
    """Generates all possible combinations of passengers for the boat."""
    options = []
    for m in range(min(max_m, boat_limit) + 1):
        for c in range(min(max_c, boat_limit - m) + 1):
            if m + c >= 1:  # boat must carry at least 1 person
                options.append((m, c))
    return options


def find_crossing_plan(total_m, total_c, boat_limit):
    # Validation
    if total_m < 0 or total_c < 0 or (total_m + total_c) == 0:
        raise ValueError("Total population must be at least 1 person.")
    if boat_limit < 1:
        raise ValueError("Boat capacity must be positive.")

    if not check_safe_state(total_m, total_c, total_m, total_c):
        raise ValueError("Initial arrangement is unsafe for missionaries.")

    # State: (m_left, c_left, boat_location)
    origin = (total_m, total_c, "L")
    search_queue = deque([origin])

    # Tracks visited states and their parent: state -> (parent_state, transfer)
    ancestors = {origin: (None, (0, 0))}
    candidate_moves = get_boat_combinations(total_m, total_c, boat_limit)

    while search_queue:
        curr_state = search_queue.popleft()
        curr_m, curr_c, curr_boat = curr_state

        # Check if everyone has safely reached the right bank
        if curr_m == 0 and curr_c == 0 and curr_boat == "R":
            sequence = []
            cursor = curr_state
            while cursor is not None:
                prev_node, transfer = ancestors[cursor]
                sequence.append((cursor, transfer))
                cursor = prev_node
            sequence.reverse()
            return sequence

        is_leaving_left = curr_boat == "L"
        next_boat_side = "R" if is_leaving_left else "L"

        for carry_m, carry_c in candidate_moves:
            if is_leaving_left:
                new_m = curr_m - carry_m
                new_c = curr_c - carry_c
            else:
                new_m = curr_m + carry_m
                new_c = curr_c + carry_c

            new_state = (new_m, new_c, next_boat_side)

            if (
                check_safe_state(new_m, new_c, total_m, total_c)
                and new_state not in ancestors
            ):
                ancestors[new_state] = (curr_state, (carry_m, carry_c))
                search_queue.append(new_state)

    return None


def run():
    try:
        val_m = input("Enter number of Missionaries [3]: ").strip()
        total_m = int(val_m) if val_m else 3

        val_c = input("Enter number of Cannibals [3]: ").strip()
        total_c = int(val_c) if val_c else 3

        val_cap = input("Enter Boat Capacity [2]: ").strip()
        boat_cap = int(val_cap) if val_cap else 2

        plan = find_crossing_plan(total_m, total_c, boat_cap)
    except (ValueError, EOFError) as err:
        print(f"Input Error: {err}")
        return

    if plan is None:
        print("No valid solution exists.")
        return

    print("\n--- Solution Found ---")
    for idx, (state, transfer) in enumerate(plan):
        m_l, c_l, boat_pos = state
        m_r = total_m - m_l
        c_r = total_c - c_l
        p_m, p_c = transfer

        if idx == 0:
            description = "Start State"
        else:
            direction = "L -> R" if boat_pos == "R" else "R -> L"
            description = f"{p_m}M + {p_c}C ({direction})"

        print(
            f"Step {idx:2d}: {description:<20} | Left: {m_l}M, {c_l}C | Right: {m_r}M, {c_r}C | Boat: {boat_pos}"
        )

    print(f"\nTotal boat trips required: {len(plan) - 1}")


if __name__ == "__main__":
    run()

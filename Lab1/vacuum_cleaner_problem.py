# Vacuum cleaner reflex agent simulation


class VacuumState:

    def __init__(self, pos, dirt_a, dirt_b):
        self.pos = pos
        self.dirt_a = dirt_a
        self.dirt_b = dirt_b

    def is_clean(self):
        return self.dirt_a == "Clean" and self.dirt_b == "Clean"

    def get_current_dirt(self):
        return self.dirt_a if self.pos == "A" else self.dirt_b

    def __repr__(self):
        return (
            f"Vacuum at {self.pos} | Room A: {self.dirt_a}, Room B:"
            f" {self.dirt_b}"
        )


def get_next_action(state):
    dirt = state.get_current_dirt()

    # If current room is dirty, clean it
    if dirt == "Dirty":
        if state.pos == "A":
            return (
                VacuumState("A", "Clean", state.dirt_b),
                "Suck dirt from Room A",
            )
        else:
            return (
                VacuumState("B", state.dirt_a, "Clean"),
                "Suck dirt from Room B",
            )

    # Move to the opposite room if current room is clean
    if state.pos == "A":
        return VacuumState("B", state.dirt_a, state.dirt_b), "Move to Room B"
    else:
        return VacuumState("A", state.dirt_a, state.dirt_b), "Move to Room A"


def run_agent(init_state, limit=20):
    steps = [(init_state, "Starting condition")]
    curr = init_state
    count = 0

    while not curr.is_clean() and count < limit:
        next_st, act = get_next_action(curr)
        steps.append((next_st, act))
        curr = next_st
        count += 1

    return steps, count, curr.is_clean()


def display_results(steps):
    print("\n--- Simulation Trace ---")
    for i, (state, act) in enumerate(steps):
        print(f"[{i}] Action : {act}")
        print(f"    State  : {state}\n")


def run():
    try:
        # Starting setup: Vacuum in Room A, both rooms dirty
        init_state = VacuumState("A", "Dirty", "Dirty")

        steps, count, success = run_agent(init_state)

        display_results(steps)

        print("--- Run Summary ---")
        print(f"Both rooms clean : {success}")
        print(f"Total steps taken: {count}")

    except Exception as err:
        print(f"Error during execution: {err}")


if __name__ == "__main__":
    run()

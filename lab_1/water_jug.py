def solve_jug(jug1_cap, jug2_cap, goal):
    jug1 = 0
    jug2 = 0

    while jug1 != goal and jug2 != goal:

        # If first jug is empty, fill it
        if jug1 == 0:
            jug1 = jug1_cap
            print(f"Fill {jug1_cap}L jug")

        # If second jug is full, empty it
        elif jug2 == jug2_cap:
            jug2 = 0
            print(f"Empty {jug2_cap}L jug")

        # Pour first jug into second jug
        else:
            transfer = min(jug1, jug2_cap - jug2)

            jug1 = jug1 - transfer
            jug2 = jug2 + transfer

            print(f"Pour {jug1_cap}L -> {jug2_cap}L")

        print(f"State: ({jug1}, {jug2})")
        print()

    print("Target reached!")


solve_jug(5, 7, 4)

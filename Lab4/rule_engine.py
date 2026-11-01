"""
Lab 4 - Task 5: Logical Rule Engine (Forward & Backward Chaining)
Author: Divyansh Verma (IC-2K23-31)
Course: IC-709D / IC-703 C | Artificial Intelligence Laboratory (Unit 4, CO1, CO3)

Description:
Implements a production rule engine with Forward Chaining (data-driven)
and Backward Chaining (goal-driven) reasoning.
Visualizes rule execution dependency graph saved to outputs/Task5.png.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Rule:
    def __init__(self, name, antecedents, consequent):
        self.name = name
        self.antecedents = set(antecedents)
        self.consequent = consequent


class LogicalRuleEngine:
    def __init__(self):
        self.rules = []
        self.working_memory = set()

    def add_rule(self, name, antecedents, consequent):
        self.rules.append(Rule(name, antecedents, consequent))

    def set_facts(self, facts):
        self.working_memory = set(facts)

    def forward_chain(self):
        """Derives all reachable conclusions from current working memory facts."""
        facts = set(self.working_memory)
        fired = []
        changed = True

        print("--- Forward Chaining Trace ---")
        cycle = 1
        while changed:
            changed = False
            for r in self.rules:
                if r.antecedents.issubset(facts) and r.consequent not in facts:
                    facts.add(r.consequent)
                    fired.append(r.name)
                    print(f"  Cycle {cycle}: Fired [{r.name}] -> Added fact '{r.consequent}'")
                    changed = True
            cycle += 1

        print(f"Total deduced facts: {len(facts)} (Rules fired: {len(fired)})\n")
        return facts

    def backward_chain(self, goal, visited=None, depth=0):
        """Recursively checks if goal can be proven from facts and rules."""
        if visited is None:
            visited = set()

        indent = "  " * depth
        print(f"{indent}? Evaluating Goal: '{goal}'")

        if goal in self.working_memory:
            print(f"{indent}-> MATCH: '{goal}' is an existing fact in working memory.")
            return True

        if goal in visited:
            return False
        visited.add(goal)

        candidates = [r for r in self.rules if r.consequent == goal]
        if not candidates:
            print(f"{indent}x No rule concludes '{goal}'.")
            return False

        for r in candidates:
            print(f"{indent}Trying [{r.name}]: Antecedents {r.antecedents}")
            all_true = True
            for ant in r.antecedents:
                if not self.backward_chain(ant, visited.copy(), depth + 1):
                    all_true = False
                    break
            if all_true:
                print(f"{indent}-> [{r.name}] SUCCEEDED! Verified '{goal}'")
                return True

        print(f"{indent}x Cannot verify '{goal}'.")
        return False

    def visualize(self, output_file):
        fig, ax = plt.subplots(figsize=(12, 7.5))
        ax.set_title("Task 5: Logical Rule Engine - Inference Dependency Flow", fontsize=13, fontweight="bold", pad=15)
        ax.set_facecolor("#F8FAFC")

        layers = {
            "cgpa_high": (0.12, 0.8),
            "good_attendance": (0.12, 0.6),
            "clean_record": (0.12, 0.4),
            "research_paper": (0.12, 0.2),

            "R1": (0.38, 0.8),
            "R2": (0.38, 0.5),
            "R3": (0.38, 0.2),

            "academically_strong": (0.62, 0.8),
            "good_conduct": (0.62, 0.5),
            "research_proven": (0.62, 0.2),

            "R4": (0.78, 0.65),
            "R5": (0.88, 0.4),

            "scholarship_eligible": (0.92, 0.8),
            "dean_honor_roll": (0.92, 0.15),
        }

        # Draw nodes
        for name, (x, y) in layers.items():
            is_rule = name.startswith("R")
            w, h = (0.08, 0.06) if is_rule else (0.15, 0.08)
            col = "#FED7AA" if is_rule else "#93C5FD" if "eligible" in name or "roll" in name else "#E2E8F0"
            border = "#C2410C" if is_rule else "#1E3A8A"

            box = patches.FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.015,rounding_size=0.02",
                facecolor=col, edgecolor=border, linewidth=1.2
            )
            ax.add_patch(box)
            ax.text(x, y, name.replace("_", "\n"), ha="center", va="center", fontsize=7, fontweight="bold")

        # Draw dependency flows
        connections = [
            ("cgpa_high", "R1"), ("R1", "academically_strong"),
            ("good_attendance", "R2"), ("clean_record", "R2"), ("R2", "good_conduct"),
            ("research_paper", "R3"), ("R3", "research_proven"),
            ("academically_strong", "R4"), ("good_conduct", "R4"), ("R4", "scholarship_eligible"),
            ("scholarship_eligible", "R5"), ("research_proven", "R5"), ("R5", "dean_honor_roll")
        ]

        for s, d in connections:
            x1, y1 = layers[s]
            x2, y2 = layers[d]
            dx, dy = (x2 - x1), (y2 - y1)
            norm = (dx**2 + dy**2)**0.5
            sx, sy = x1 + 0.05 * dx / norm, y1 + 0.05 * dy / norm
            ex, ey = x2 - 0.05 * dx / norm, y2 - 0.05 * dy / norm
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.2, shrinkA=2, shrinkB=2))

        ax.text(0.12, 0.95, "Layer 1: Input Facts", ha="center", va="center", fontsize=9, fontweight="bold", color="#1E3A8A")
        ax.text(0.38, 0.95, "Layer 2: Base Rules", ha="center", va="center", fontsize=9, fontweight="bold", color="#C2410C")
        ax.text(0.62, 0.95, "Layer 3: Derived Facts", ha="center", va="center", fontsize=9, fontweight="bold", color="#1E3A8A")
        ax.text(0.85, 0.95, "Layer 4: Higher Order Rules & Goal", ha="center", va="center", fontsize=9, fontweight="bold", color="#047857")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"[Task 5] Diagram saved successfully to: {output_file}")


def main():
    print("=" * 65)
    print("  TASK 5: LOGICAL RULE ENGINE (FORWARD & BACKWARD CHAINING)")
    print("=" * 65)

    engine = LogicalRuleEngine()

    # Rule Base
    engine.add_rule("R1", ["cgpa_high"], "academically_strong")
    engine.add_rule("R2", ["good_attendance", "clean_record"], "good_conduct")
    engine.add_rule("R3", ["research_paper"], "research_proven")
    engine.add_rule("R4", ["academically_strong", "good_conduct"], "scholarship_eligible")
    engine.add_rule("R5", ["scholarship_eligible", "research_proven"], "dean_honor_roll")

    initial_facts = ["cgpa_high", "good_attendance", "clean_record", "research_paper"]
    engine.set_facts(initial_facts)
    print(f"Initial Working Memory Facts: {initial_facts}\n")

    # Forward Chaining
    all_facts = engine.forward_chain()
    print("All Inferred Facts:")
    for f in sorted(all_facts):
        print(f"  + {f}")

    # Backward Chaining
    query_goal = "dean_honor_roll"
    print(f"\n--- Backward Chaining Query: Can we prove '{query_goal}'? ---")
    is_proven = engine.backward_chain(query_goal)
    print(f"\nGoal '{query_goal}' Proven: {is_proven}")

    output_path = os.path.join(os.path.dirname(__file__), "outputs", "rule_engine.png")
    engine.visualize(output_path)
    print("=" * 65)


if __name__ == "__main__":
    main()

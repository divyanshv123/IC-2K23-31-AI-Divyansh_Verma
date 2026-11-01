"""
Lab 4 - Task 3: Semantic Network Construction & Inheritance Reasoning
Author: Divyansh Verma (IC-2K23-31)
Course: IC-709D / IC-703 C | Artificial Intelligence Laboratory (Unit 4, CO1, CO3)

Description:
Implements a graph-based Semantic Network with property inheritance traversal.
Constructs a biological and university hierarchy and saves the visual graph to outputs/Task3.png.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SemanticNetwork:
    def __init__(self):
        self.nodes = set()
        self.edges = {}          # node -> list of (relation, target)
        self.properties = {}     # node -> dict of (attr, val)

    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = []
        if node not in self.properties:
            self.properties[node] = {}

    def add_relation(self, src, relation, dst):
        self.add_node(src)
        self.add_node(dst)
        self.edges[src].append((relation, dst))

    def add_property(self, node, prop, value):
        self.add_node(node)
        self.properties[node][prop] = value

    def inherit_property(self, node, prop, visited=None):
        """Recursively resolves properties up the IS_A taxonomy."""
        if visited is None:
            visited = set()
        if node in visited:
            return None
        visited.add(node)

        # 1. Direct attribute
        if prop in self.properties.get(node, {}):
            return self.properties[node][prop], node

        # 2. Inherit via IS_A
        for rel, parent in self.edges.get(node, []):
            if rel == "IS_A":
                res = self.inherit_property(parent, prop, visited)
                if res is not None:
                    return res
        return None

    def visualize(self, output_file):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        ax.set_title("Task 3: Semantic Network Construction & Property Inheritance", fontsize=13, fontweight="bold", pad=15)
        ax.set_facecolor("#F8FAFC")

        pos = {
            "Animal": (0.5, 0.9),
            "Mammal": (0.25, 0.65),
            "Bird": (0.75, 0.65),
            "Carnivore": (0.15, 0.38),
            "Herbivore": (0.35, 0.38),
            "FlyingBird": (0.65, 0.38),
            "FlightlessBird": (0.85, 0.38),
            "Lion": (0.15, 0.1),
            "Eagle": (0.65, 0.1),
            "Penguin": (0.85, 0.1),
        }

        labels = {
            "Animal": "Animal\n[breathes: True]",
            "Mammal": "Mammal\n[has_fur: True]",
            "Bird": "Bird\n[has_feathers: True]",
            "Carnivore": "Carnivore\n[eats: meat]",
            "Herbivore": "Herbivore\n[eats: plants]",
            "FlyingBird": "FlyingBird\n[can_fly: True]",
            "FlightlessBird": "FlightlessBird\n[can_fly: False]",
            "Lion": "Lion\n[apex_predator: Yes]",
            "Eagle": "Eagle\n[vision: sharp]",
            "Penguin": "Penguin\n[can_swim: True]",
        }

        colors = {
            "Animal": "#93C5FD",
            "Mammal": "#86EFAC",
            "Bird": "#86EFAC",
            "Carnivore": "#FDE047",
            "Herbivore": "#FDE047",
            "FlyingBird": "#FDE047",
            "FlightlessBird": "#FDE047",
            "Lion": "#FCA5A5",
            "Eagle": "#FCA5A5",
            "Penguin": "#FCA5A5",
        }

        edges = [
            ("Mammal", "IS_A", "Animal"),
            ("Bird", "IS_A", "Animal"),
            ("Carnivore", "IS_A", "Mammal"),
            ("Herbivore", "IS_A", "Mammal"),
            ("FlyingBird", "IS_A", "Bird"),
            ("FlightlessBird", "IS_A", "Bird"),
            ("Lion", "IS_A", "Carnivore"),
            ("Eagle", "IS_A", "FlyingBird"),
            ("Penguin", "IS_A", "FlightlessBird"),
            ("Eagle", "PREYS_ON", "Herbivore"),
            ("Lion", "PREYS_ON", "Herbivore"),
        ]

        # Draw boxes
        for key, (x, y) in pos.items():
            box = patches.FancyBboxPatch(
                (x - 0.08, y - 0.045), 0.16, 0.09,
                boxstyle="round,pad=0.015,rounding_size=0.02",
                facecolor=colors[key], edgecolor="#334155", linewidth=1.2
            )
            ax.add_patch(box)
            ax.text(x, y, labels[key], ha="center", va="center", fontsize=7.5, fontweight="bold", color="#0F172A")

        # Draw arrows
        for src, rel, dst in edges:
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            dx, dy = (x2 - x1), (y2 - y1)
            norm = (dx**2 + dy**2)**0.5
            sx, sy = x1 + 0.05 * dx / norm, y1 + 0.05 * dy / norm
            ex, ey = x2 - 0.05 * dx / norm, y2 - 0.05 * dy / norm

            edge_col = "#DC2626" if rel == "IS_A" else "#2563EB"
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=edge_col, lw=1.2, shrinkA=2, shrinkB=2))
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, rel, ha="center", va="center", fontsize=6.5, fontweight="bold",
                    color=edge_col, bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"[Task 3] Diagram saved successfully to: {output_file}")


def main():
    print("=" * 65)
    print("  TASK 3: SEMANTIC NETWORK CONSTRUCTION & INHERITANCE")
    print("=" * 65)

    sn = SemanticNetwork()

    # Taxonomy definitions
    sn.add_relation("Mammal", "IS_A", "Animal")
    sn.add_relation("Bird", "IS_A", "Animal")
    sn.add_relation("Carnivore", "IS_A", "Mammal")
    sn.add_relation("FlyingBird", "IS_A", "Bird")
    sn.add_relation("FlightlessBird", "IS_A", "Bird")
    sn.add_relation("Lion", "IS_A", "Carnivore")
    sn.add_relation("Eagle", "IS_A", "FlyingBird")
    sn.add_relation("Penguin", "IS_A", "FlightlessBird")

    # Associative relation
    sn.add_relation("Lion", "PREYS_ON", "Herbivore")

    # Attached properties
    sn.add_property("Animal", "breathes", True)
    sn.add_property("Animal", "has_cells", True)
    sn.add_property("Mammal", "has_fur", True)
    sn.add_property("Mammal", "warm_blooded", True)
    sn.add_property("Bird", "has_feathers", True)
    sn.add_property("FlyingBird", "can_fly", True)
    sn.add_property("FlightlessBird", "can_fly", False)
    sn.add_property("Lion", "sound", "Roar")
    sn.add_property("Penguin", "can_swim", True)

    print("\nDemonstrating Hierarchical Property Inheritance:")
    test_queries = [
        ("Lion", "breathes"),
        ("Lion", "has_fur"),
        ("Lion", "sound"),
        ("Eagle", "breathes"),
        ("Eagle", "can_fly"),
        ("Penguin", "can_fly"),
        ("Penguin", "can_swim"),
        ("Penguin", "has_feathers"),
    ]

    for entity, prop in test_queries:
        val, src = sn.inherit_property(entity, prop)
        print(f"  * Query '{prop}' on [{entity}]: {val} (Inherited from: [{src}])")

    output_path = os.path.join(os.path.dirname(__file__), "outputs", "semantic_network.png")
    sn.visualize(output_path)
    print("=" * 65)


if __name__ == "__main__":
    main()

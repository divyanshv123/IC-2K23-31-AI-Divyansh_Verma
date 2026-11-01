"""
Lab 4: Knowledge Representation & Logical Reasoning
Author: Divyansh Verma (IC-2K23-31)
Course: IC-709D / IC-703 C | Artificial Intelligence Laboratory (Unit 4, CO1, CO3)

This script implements three primary knowledge representation paradigms:
1. Semantic Networks (Graph-based associative memory with property inheritance)
2. Frame-Based Systems (Object-centered representation with slots, facets, and defaults)
3. Rule-Based Expert System (Forward & Backward Chaining inference engines)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# =============================================================================
# PART 1: SEMANTIC NETWORK REPRESENTATION & PROPERTY INHERITANCE
# =============================================================================

class SemanticNetwork:
    """Graph-based knowledge representation using nodes and labeled directed edges."""

    def __init__(self):
        self.nodes = set()
        # adjacency list: source -> list of (relation, target)
        self.edges = {}
        # direct attributes: node -> dict of (property, value)
        self.properties = {}

    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = []
        if node not in self.properties:
            self.properties[node] = {}

    def add_relation(self, source, relation, target):
        self.add_node(source)
        self.add_node(target)
        self.edges[source].append((relation, target))

    def add_property(self, node, prop, value):
        self.add_node(node)
        self.properties[node][prop] = value

    def query_property(self, entity, prop, visited=None):
        """Recursively query an attribute using hierarchical IS_A inheritance."""
        if visited is None:
            visited = set()
        if entity in visited:
            return None
        visited.add(entity)

        # 1. Check direct properties
        if prop in self.properties.get(entity, {}):
            return self.properties[entity][prop], entity

        # 2. Traverse IS_A relationships upward to inherit
        for relation, parent in self.edges.get(entity, []):
            if relation == "IS_A":
                result = self.query_property(parent, prop, visited)
                if result is not None:
                    return result
        return None

    def visualize(self, output_path):
        """Render and save the semantic network graph."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Semantic Network Knowledge Base (University Domain)", fontsize=14, fontweight="bold", pad=20)
        ax.set_facecolor("#F9FAFB")

        # Fixed coordinates for clean, readable layout
        pos = {
            "UniversityPerson": (0.5, 0.9),
            "Faculty": (0.25, 0.65),
            "Student": (0.75, 0.65),
            "Professor": (0.15, 0.35),
            "TeachingAssistant": (0.45, 0.35),
            "UndergradStudent": (0.85, 0.35),
            "IC_709D_Lab": (0.5, 0.08),
        }

        # Node display styling
        colors = {
            "UniversityPerson": "#93C5FD",
            "Faculty": "#86EFAC",
            "Student": "#FDE047",
            "Professor": "#6EE7B7",
            "TeachingAssistant": "#FDBA74",
            "UndergradStudent": "#FCA5A5",
            "IC_709D_Lab": "#C4B5FD",
        }

        # Draw nodes (boxes)
        for node, (x, y) in pos.items():
            color = colors.get(node, "#E5E7EB")
            box = patches.FancyBboxPatch(
                (x - 0.1, y - 0.04), 0.2, 0.08,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                facecolor=color, edgecolor="#1F2937", linewidth=1.5
            )
            ax.add_patch(box)
            ax.text(x, y, node, ha="center", va="center", fontsize=9, fontweight="bold", color="#111827")

        # Draw directed relationship edges with labels
        rendered_relations = [
            ("Faculty", "IS_A", "UniversityPerson"),
            ("Student", "IS_A", "UniversityPerson"),
            ("Professor", "IS_A", "Faculty"),
            ("TeachingAssistant", "IS_A", "Student"),
            ("TeachingAssistant", "ASSISTS", "Faculty"),
            ("UndergradStudent", "IS_A", "Student"),
            ("Professor", "TEACHES", "IC_709D_Lab"),
            ("UndergradStudent", "ENROLLED_IN", "IC_709D_Lab"),
            ("TeachingAssistant", "EVALUATES", "IC_709D_Lab"),
        ]

        for src, rel, dst in rendered_relations:
            if src in pos and dst in pos:
                x1, y1 = pos[src]
                x2, y2 = pos[dst]
                # Offset arrows slightly to avoid overlapping boxes
                dx, dy = (x2 - x1), (y2 - y1)
                norm = (dx**2 + dy**2)**0.5
                sx, sy = x1 + 0.05 * dx / norm, y1 + 0.05 * dy / norm
                ex, ey = x2 - 0.05 * dx / norm, y2 - 0.05 * dy / norm

                ax.annotate(
                    "", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5, shrinkA=3, shrinkB=3)
                )
                # Label relation
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y + 0.02, rel, ha="center", va="center", fontsize=8,
                        fontweight="bold", color="#DC2626" if rel == "IS_A" else "#2563EB",
                        bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[Semantic Network] Diagram generated and saved to: {output_path}")


# =============================================================================
# PART 2: FRAME-BASED KNOWLEDGE REPRESENTATION
# =============================================================================

class Frame:
    """Object-centered representation with slots, facets, and default values."""

    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.slots = {}
        self.facets = {}  # slot -> dict of facets (default, constraint)

    def set_slot(self, slot, value, default=None):
        self.slots[slot] = value
        if default is not None:
            if slot not in self.facets:
                self.facets[slot] = {}
            self.facets[slot]["default"] = default

    def get_slot(self, slot):
        """Retrieve slot value directly or inherit from parent frame."""
        # 1. Direct slot value
        if slot in self.slots and self.slots[slot] is not None:
            return self.slots[slot], self.name

        # 2. Check local default facet
        if slot in self.facets and "default" in self.facets[slot]:
            return self.facets[slot]["default"], f"{self.name} (Default)"

        # 3. Inherit from parent frame
        if self.parent:
            return self.parent.get_slot(slot)

        return None, "Not Found"

    def display(self):
        print(f"Frame: [{self.name}]" + (f" (Parent: {self.parent.name})" if self.parent else ""))
        for slot, val in self.slots.items():
            print(f"  * Slot: {slot:15s} = {val}")


# =============================================================================
# PART 3: RULE-BASED EXPERT SYSTEM (FORWARD & BACKWARD CHAINING)
# =============================================================================

class Rule:
    """Production rule representing IF <antecedents> THEN <consequent>."""

    def __init__(self, name, antecedents, consequent):
        self.name = name
        self.antecedents = set(antecedents)  # conditions that must hold true
        self.consequent = consequent        # deduced fact


class RuleBasedExpertSystem:
    """Inference engine supporting both Forward and Backward Chaining."""

    def __init__(self):
        self.rules = []
        self.facts = set()

    def add_rule(self, name, antecedents, consequent):
        self.rules.append(Rule(name, antecedents, consequent))

    def set_facts(self, facts):
        self.facts = set(facts)

    def forward_chain(self):
        """Data-driven inference: Derive all reachable facts from known premises."""
        inferred = set(self.facts)
        fired_rules = []
        changed = True

        print("--- [Forward Chaining Execution Trace] ---")
        cycle = 1
        while changed:
            changed = False
            for rule in self.rules:
                # If all antecedents are satisfied and consequent is not yet known
                if rule.antecedents.issubset(inferred) and rule.consequent not in inferred:
                    inferred.add(rule.consequent)
                    fired_rules.append(rule.name)
                    print(f"  Cycle {cycle}: Fired '{rule.name}' -> Inferred fact: '{rule.consequent}'")
                    changed = True
            cycle += 1

        print(f"Total facts derived: {len(inferred)} (Fired {len(fired_rules)} rules)\n")
        return inferred

    def backward_chain(self, goal, visited=None, depth=0):
        """Goal-driven inference: Recursively determine if goal can be established."""
        if visited is None:
            visited = set()

        indent = "  " * depth
        print(f"{indent}? Testing Goal: '{goal}'")

        # Base case 1: Goal is already a known fact
        if goal in self.facts:
            print(f"{indent}-> Confirmed directly from known facts.")
            return True

        if goal in visited:
            return False
        visited.add(goal)

        # Base case 2: Find rules that conclude the goal
        matching_rules = [r for r in self.rules if r.consequent == goal]
        if not matching_rules:
            print(f"{indent}x No rule found concluding '{goal}'.")
            return False

        for rule in matching_rules:
            print(f"{indent}Trying Rule [{rule.name}]: Requires {rule.antecedents}")
            all_subgoals_met = True
            for ant in rule.antecedents:
                if not self.backward_chain(ant, visited.copy(), depth + 1):
                    all_subgoals_met = False
                    break
            if all_subgoals_met:
                print(f"{indent}-> Rule [{rule.name}] SUCCEEDED! Verified '{goal}'")
                return True

        print(f"{indent}x Failed to establish '{goal}'.")
        return False


# =============================================================================
# MAIN EXECUTION & COMPARATIVE STUDY
# =============================================================================

def main():
    print("=" * 70)
    print("  LAB 4: KNOWLEDGE REPRESENTATION & REASONING (UNIT 4)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Semantic Network Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 1. SEMANTIC NETWORK KNOWLEDGE BASE ---")
    sn = SemanticNetwork()

    # Define nodes and hierarchical relations
    sn.add_relation("Professor", "IS_A", "Faculty")
    sn.add_relation("TeachingAssistant", "IS_A", "Student")
    sn.add_relation("TeachingAssistant", "ASSISTS", "Faculty")
    sn.add_relation("UndergradStudent", "IS_A", "Student")
    sn.add_relation("Faculty", "IS_A", "UniversityPerson")
    sn.add_relation("Student", "IS_A", "UniversityPerson")

    # Add attributes at appropriate hierarchical levels
    sn.add_property("UniversityPerson", "campus_access", "Granted")
    sn.add_property("UniversityPerson", "wifi_access", "eduroam_secure")
    sn.add_property("Faculty", "office_allocated", "Yes")
    sn.add_property("Student", "tuition_fees", "Active")
    sn.add_property("Professor", "voting_rights", "Senate_Member")
    sn.add_property("UndergradStudent", "degree", "B.Tech")

    # Query inherited properties
    queries = [
        ("UndergradStudent", "campus_access"),
        ("UndergradStudent", "degree"),
        ("Professor", "campus_access"),
        ("Professor", "voting_rights"),
    ]

    print("Inheritance Queries:")
    for entity, prop in queries:
        val, source = sn.query_property(entity, prop)
        print(f"  * Query '{prop}' for [{entity}]: {val} (Inherited from: [{source}])")

    # Output diagram
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    sn_img_path = os.path.join(output_dir, "semantic_network.png")
    sn.visualize(sn_img_path)

    # -------------------------------------------------------------------------
    # 2. Frame-Based Representation Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 2. FRAME-BASED KNOWLEDGE REPRESENTATION ---")

    # Parent Frame
    Course = Frame("Course")
    Course.set_slot("institution", "Engineering Institute")
    Course.set_slot("grading_mode", "Letter_Grade", default="Letter_Grade")
    Course.set_slot("attendance_threshold", "75%", default="75%")

    # Child Frame inheriting from Course
    LabCourse = Frame("LabCourse", parent=Course)
    LabCourse.set_slot("practical_exam", "Mandatory")
    LabCourse.set_slot("lab_safety_clearance", "Required", default="Required")

    # Instance Frame inheriting from LabCourse
    AICourse = Frame("IC_709D_AI_Lab", parent=LabCourse)
    AICourse.set_slot("course_title", "Artificial Intelligence Laboratory")
    AICourse.set_slot("instructor", "Dr. Faculty")
    AICourse.set_slot("credits", 2)

    print("\nDisplaying Frame Structure for 'IC_709D_AI_Lab':")
    AICourse.display()

    print("\nSlot Inheritance Resolution:")
    test_slots = ["course_title", "practical_exam", "attendance_threshold", "institution"]
    for slot in test_slots:
        val, src = AICourse.get_slot(slot)
        print(f"  * Slot [{slot:20s}]: {str(val):25s} (Resolved from: {src})")

    # -------------------------------------------------------------------------
    # 3. Rule-Based Expert System (Academic Honors & Scholarship Advisory)
    # -------------------------------------------------------------------------
    print("\n--- 3. RULE-BASED EXPERT SYSTEM (ACADEMIC ADVISORY) ---")
    rb = RuleBasedExpertSystem()

    # Define Domain Production Rules
    rb.add_rule("R1_High_CGPA", ["cgpa_above_8_5"], "academically_distinguished")
    rb.add_rule("R2_Research", ["published_paper"], "research_competent")
    rb.add_rule("R3_Conduct", ["no_disciplinary_action", "attendance_above_85"], "good_standing")
    rb.add_rule("R4_Merit_Scholarship", ["academically_distinguished", "good_standing"], "eligible_for_merit_scholarship")
    rb.add_rule("R5_Dean_Honor_List", ["eligible_for_merit_scholarship", "research_competent"], "awarded_dean_honor_roll")

    # Scenario: Student Profile Facts
    student_facts = [
        "cgpa_above_8_5",
        "attendance_above_85",
        "no_disciplinary_action",
        "published_paper"
    ]
    rb.set_facts(student_facts)
    print(f"Initial Working Memory (Student Facts): {student_facts}\n")

    # Forward Chaining
    derived_facts = rb.forward_chain()
    print("Final Inferred Facts:")
    for f in sorted(derived_facts):
        print(f"  + {f}")

    # Backward Chaining: Test hypothesis
    target_goal = "awarded_dean_honor_roll"
    print(f"\n--- Backward Chaining Query: Can we prove '{target_goal}'? ---")
    proven = rb.backward_chain(target_goal)
    print(f"\nHypothesis '{target_goal}' Proven: {proven}")

    # -------------------------------------------------------------------------
    # 4. Comparative Analysis Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  4. COMPARATIVE EVALUATION OF KNOWLEDGE REPRESENTATION SCHEMES")
    print("=" * 70)
    print(f"{'Paradigm':<20} | {'Primary Structure':<22} | {'Reasoning Mechanism':<22}")
    print("-" * 70)
    print(f"{'Semantic Networks':<20} | {'Graph (Nodes/Edges)':<22} | {'Inheritance & Traversal':<22}")
    print(f"{'Frames':<20} | {'Slots & Facets':<22} | {'Default Values & Methods':<22}")
    print(f"{'Rule-Based Systems':<20} | {'IF-THEN Rules':<22} | {'Forward/Backward Chaining':<22}")
    print("=" * 70)
    print("Lab 4 Knowledge Representation Module Executed Successfully!")


if __name__ == "__main__":
    main()

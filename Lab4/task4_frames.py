"""
Lab 4 - Task 4: Frame-Based Knowledge Representation
Author: Divyansh Verma (IC-2K23-31)
Course: IC-709D / IC-703 C | Artificial Intelligence Laboratory (Unit 4, CO1, CO3)

Description:
Implements Frame structures with slots, facets, default values, and class inheritance.
Visualizes frame hierarchical slot-value tables saved to outputs/Task4.png.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Frame:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.slots = {}
        self.facets = {}  # slot -> {default: val, constraint: type/range}

    def set_slot(self, slot, value, default=None, constraint=None):
        self.slots[slot] = value
        if default is not None or constraint is not None:
            if slot not in self.facets:
                self.facets[slot] = {}
            if default is not None:
                self.facets[slot]["default"] = default
            if constraint is not None:
                self.facets[slot]["constraint"] = constraint

    def get_slot(self, slot):
        """Resolves slot value: local value -> default facet -> inherited from parent."""
        # 1. Local slot value
        if slot in self.slots and self.slots[slot] is not None:
            return self.slots[slot], self.name

        # 2. Local default facet
        if slot in self.facets and "default" in self.facets[slot]:
            return self.facets[slot]["default"], f"{self.name} (Default)"

        # 3. Inherit from parent
        if self.parent:
            return self.parent.get_slot(slot)

        return None, "Not Found"


def visualize_frames(output_file):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_title("Task 4: Frame-Based Knowledge Representation Hierarchy", fontsize=13, fontweight="bold", pad=15)
    ax.set_facecolor("#F8FAFC")

    frames_data = [
        {"name": "Frame: Course (Root)", "pos": (0.5, 0.8), "color": "#93C5FD",
         "slots": ["institution: University", "grading: Letter_Grade", "attendance_min: 75%"]},
        {"name": "Frame: LabCourse (Inherits Course)", "pos": (0.28, 0.45), "color": "#86EFAC",
         "slots": ["practical_exam: Mandatory", "lab_safety: Required", "max_capacity: 30"]},
        {"name": "Frame: TheoryCourse (Inherits Course)", "pos": (0.72, 0.45), "color": "#FDE047",
         "slots": ["written_exam: Mandatory", "assignments: 4", "tutorial_sessions: Optional"]},
        {"name": "Instance: IC_709D_AI_Lab", "pos": (0.28, 0.12), "color": "#FCA5A5",
         "slots": ["title: AI Lab", "instructor: Dr. Sharma", "credits: 2", "room: Lab-302"]},
    ]

    # Draw frame cards
    for f in frames_data:
        x, y = f["pos"]
        h = 0.04 + len(f["slots"]) * 0.038
        w = 0.36
        box = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=f["color"], edgecolor="#1E293B", linewidth=1.3
        )
        ax.add_patch(box)
        ax.text(x, y + h / 2 - 0.025, f["name"], ha="center", va="center", fontsize=8.5, fontweight="bold")
        for idx, s in enumerate(f["slots"]):
            ax.text(x - w / 2 + 0.02, y + h / 2 - 0.06 - idx * 0.035, f"• {s}", ha="left", va="center", fontsize=7.5, color="#0F172A")

    # Draw inheritance arrows
    ax.annotate("", xy=(0.5, 0.68), xytext=(0.35, 0.58),
                arrowprops=dict(arrowstyle="<-", color="#0F172A", lw=1.3, shrinkA=4, shrinkB=4))
    ax.text(0.38, 0.65, "Inherits", fontsize=7.5, fontweight="bold", color="#1E3A8A")

    ax.annotate("", xy=(0.5, 0.68), xytext=(0.65, 0.58),
                arrowprops=dict(arrowstyle="<-", color="#0F172A", lw=1.3, shrinkA=4, shrinkB=4))
    ax.text(0.60, 0.65, "Inherits", fontsize=7.5, fontweight="bold", color="#1E3A8A")

    ax.annotate("", xy=(0.28, 0.32), xytext=(0.28, 0.22),
                arrowprops=dict(arrowstyle="<-", color="#0F172A", lw=1.3, shrinkA=4, shrinkB=4))
    ax.text(0.31, 0.27, "Instantiates", fontsize=7.5, fontweight="bold", color="#DC2626")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"[Task 4] Diagram saved successfully to: {output_file}")


def main():
    print("=" * 65)
    print("  TASK 4: FRAME-BASED KNOWLEDGE REPRESENTATION")
    print("=" * 65)

    # Root Frame
    Course = Frame("Course")
    Course.set_slot("institution", "Engineering Institute")
    Course.set_slot("grading_mode", "Letter_Grade", default="Letter_Grade")
    Course.set_slot("attendance_min", "75%", default="75%")

    # Sub-Frames
    LabCourse = Frame("LabCourse", parent=Course)
    LabCourse.set_slot("practical_exam", "Mandatory")
    LabCourse.set_slot("lab_safety", "Required", default="Required")

    TheoryCourse = Frame("TheoryCourse", parent=Course)
    TheoryCourse.set_slot("written_exam", "Mandatory")
    TheoryCourse.set_slot("assignments", 4)

    # Instance Frame
    AILab = Frame("IC_709D_AI_Lab", parent=LabCourse)
    AILab.set_slot("title", "Artificial Intelligence Laboratory")
    AILab.set_slot("instructor", "Dr. Sharma")
    AILab.set_slot("credits", 2)
    AILab.set_slot("room", "Lab-302")

    print(f"Instantiated Frame: [{AILab.name}] inheriting from [{LabCourse.name}] -> [{Course.name}]\n")
    print("Slot Resolution Trace:")
    query_slots = ["title", "instructor", "practical_exam", "lab_safety", "grading_mode", "institution"]
    for s in query_slots:
        val, src = AILab.get_slot(s)
        print(f"  * Slot [{s:16s}] = {str(val):32s} (Source: {src})")

    output_path = os.path.join(os.path.dirname(__file__), "outputs", "Task4.png")
    visualize_frames(output_path)
    print("=" * 65)


if __name__ == "__main__":
    main()

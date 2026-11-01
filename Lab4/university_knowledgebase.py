"""
Lab 4 - Task 1: University Knowledge Base Representation
Author: Divyansh Verma (IC-2K23-31)
Course: IC-709D / IC-703 C | Artificial Intelligence Laboratory (Unit 4, CO1, CO3)

Description:
Models a university knowledge base with entities (Students, Faculty, Courses, Departments),
relational facts, prerequisites, and inference rules for student enrollment eligibility.
Generates an ontological structure diagram saved to outputs/Task1.png.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class UniversityKnowledgeBase:
    def __init__(self):
        self.departments = {}
        self.faculty = {}
        self.students = {}
        self.courses = {}
        self.enrollments = []
        self.prerequisites = {}

    def add_department(self, dept_id, name):
        self.departments[dept_id] = name

    def add_faculty(self, faculty_id, name, dept_id, designation):
        self.faculty[faculty_id] = {"name": name, "department": dept_id, "designation": designation}

    def add_course(self, course_code, title, credits, dept_id, prereqs=None):
        self.courses[course_code] = {"title": title, "credits": credits, "department": dept_id}
        self.prerequisites[course_code] = prereqs or []

    def add_student(self, student_id, name, semester, cgpa, completed_courses=None):
        self.students[student_id] = {
            "name": name,
            "semester": semester,
            "cgpa": cgpa,
            "completed": set(completed_courses or [])
        }

    def check_enrollment_eligibility(self, student_id, course_code):
        """Inference rule: Student can enroll iff all prerequisites are completed and CGPA >= 6.0."""
        student = self.students.get(student_id)
        course = self.courses.get(course_code)
        if not student or not course:
            return False, "Student or course not found in knowledge base."

        missing_prereqs = [p for p in self.prerequisites.get(course_code, []) if p not in student["completed"]]
        if missing_prereqs:
            return False, f"Missing prerequisites: {missing_prereqs}"

        if student["cgpa"] < 6.0:
            return False, f"CGPA {student['cgpa']} is below minimum threshold (6.0)."

        return True, "Eligible for enrollment."

    def visualize(self, output_file):
        """Visualizes the university ontological hierarchy."""
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.set_title("Task 1: University Knowledge Base - Ontological Structure", fontsize=13, fontweight="bold", pad=15)
        ax.set_facecolor("#F8FAFC")

        nodes = {
            "University": (0.5, 0.9),
            "Dept_IC": (0.25, 0.65),
            "Dept_CS": (0.75, 0.65),
            "Faculty_IC": (0.15, 0.38),
            "Course_AI": (0.35, 0.38),
            "Faculty_CS": (0.65, 0.38),
            "Course_DSA": (0.85, 0.38),
            "Student_IC": (0.25, 0.1),
            "Student_CS": (0.75, 0.1),
        }

        labels = {
            "University": "University Knowledge Base\n(Core Entity)",
            "Dept_IC": "Dept of Instrumentation\n& Control",
            "Dept_CS": "Dept of Computer\nScience",
            "Faculty_IC": "Dr. Sharma\n(Associate Prof)",
            "Course_AI": "IC-709D: AI Lab\n(2 Credits)",
            "Faculty_CS": "Dr. Verma\n(Professor)",
            "Course_DSA": "CS-301: Data Structures\n(4 Credits)",
            "Student_IC": "Divyansh Verma\n(IC-2K23-31 | CGPA: 8.5)",
            "Student_CS": "Rahul Singh\n(CS-2K23-15 | CGPA: 7.8)",
        }

        colors = {
            "University": "#93C5FD",
            "Dept_IC": "#A7F3D0",
            "Dept_CS": "#A7F3D0",
            "Faculty_IC": "#FDE68A",
            "Course_AI": "#DDD6FE",
            "Faculty_CS": "#FDE68A",
            "Course_DSA": "#DDD6FE",
            "Student_IC": "#FECDD3",
            "Student_CS": "#FECDD3",
        }

        edges = [
            ("University", "CONTAINS", "Dept_IC"),
            ("University", "CONTAINS", "Dept_CS"),
            ("Dept_IC", "AFFILIATES", "Faculty_IC"),
            ("Dept_IC", "OFFERS", "Course_AI"),
            ("Dept_CS", "AFFILIATES", "Faculty_CS"),
            ("Dept_CS", "OFFERS", "Course_DSA"),
            ("Faculty_IC", "TEACHES", "Course_AI"),
            ("Student_IC", "ENROLLED_IN", "Course_AI"),
            ("Student_CS", "ENROLLED_IN", "Course_DSA"),
        ]

        # Draw boxes
        for key, (x, y) in nodes.items():
            box = patches.FancyBboxPatch(
                (x - 0.09, y - 0.05), 0.18, 0.10,
                boxstyle="round,pad=0.015,rounding_size=0.02",
                facecolor=colors[key], edgecolor="#334155", linewidth=1.2
            )
            ax.add_patch(box)
            ax.text(x, y, labels[key], ha="center", va="center", fontsize=7.5, fontweight="bold", color="#0F172A")

        # Draw arrows
        for src, rel, dst in edges:
            x1, y1 = nodes[src]
            x2, y2 = nodes[dst]
            dx, dy = (x2 - x1), (y2 - y1)
            norm = (dx**2 + dy**2)**0.5
            sx, sy = x1 + 0.06 * dx / norm, y1 + 0.06 * dy / norm
            ex, ey = x2 - 0.06 * dx / norm, y2 - 0.06 * dy / norm

            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.2, shrinkA=2, shrinkB=2))
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, rel, ha="center", va="center", fontsize=6.5, fontweight="bold",
                    color="#1E3A8A", bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"[Task 1] Diagram saved successfully to: {output_file}")


def main():
    print("=" * 65)
    print("  TASK 1: UNIVERSITY KNOWLEDGE BASE IMPLEMENTATION")
    print("=" * 65)

    kb = UniversityKnowledgeBase()

    # Populate KB
    kb.add_department("IC", "Instrumentation and Control Engineering")
    kb.add_department("CS", "Computer Science and Engineering")

    kb.add_faculty("F01", "Dr. Sharma", "IC", "Associate Professor")
    kb.add_faculty("F02", "Dr. Verma", "CS", "Professor")

    kb.add_course("CS101", "Intro to Programming", 3, "CS")
    kb.add_course("IC709D", "Artificial Intelligence Lab", 2, "IC", prereqs=["CS101"])
    kb.add_course("IC710", "Advanced Robotics", 4, "IC", prereqs=["IC709D"])

    kb.add_student("S31", "Divyansh Verma", 5, 8.5, completed_courses=["CS101"])
    kb.add_student("S45", "Aman Kumar", 3, 5.8, completed_courses=["CS101"])
    kb.add_student("S12", "Rohan Mehta", 3, 7.2, completed_courses=[])

    print("\nEvaluating Enrollment Queries:")
    test_queries = [
        ("S31", "IC709D"),
        ("S31", "IC710"),
        ("S45", "IC709D"),
        ("S12", "IC709D"),
    ]

    for sid, cid in test_queries:
        sname = kb.students[sid]["name"]
        cname = kb.courses[cid]["title"]
        eligible, reason = kb.check_enrollment_eligibility(sid, cid)
        status = "ELIGIBLE" if eligible else "INELIGIBLE"
        print(f"  * Student: {sname} ({sid}) -> Course: {cname} ({cid})")
        print(f"    Status : [{status}] Reason: {reason}\n")

    output_path = os.path.join(os.path.dirname(__file__), "outputs", "university_knowledgebase.png")
    kb.visualize(output_path)
    print("=" * 65)


if __name__ == "__main__":
    main()

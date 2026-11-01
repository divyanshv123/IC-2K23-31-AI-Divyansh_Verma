"""
Lab 4 - Task 2: Medical Knowledge Representation & Clinical Diagnosis
Author: Divyansh Verma (IC-2K23-31)
Course: IC-709D / IC-703 C | Artificial Intelligence Laboratory (Unit 4, CO1, CO3)

Description:
Represents medical knowledge correlating clinical symptoms with diagnoses,
risk factors, and treatment recommendations using heuristic rules.
Generates a symptom-disease diagnostic bipartite graph saved to outputs/Task2.png.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MedicalKnowledgeBase:
    def __init__(self):
        # Disease definitions: {disease_name: {symptoms: set(), severity: str, treatment: str}}
        self.disease_kb = {
            "Common Cold": {
                "symptoms": {"sneezing", "sore_throat", "runny_nose", "mild_cough"},
                "severity": "Mild",
                "treatment": "Rest, hydration, and over-the-counter decongestants."
            },
            "Influenza (Flu)": {
                "symptoms": {"high_fever", "body_ache", "fatigue", "dry_cough", "chills"},
                "severity": "Moderate",
                "treatment": "Antiviral medication (Tamiflu) within 48h, plenty of fluids."
            },
            "COVID-19": {
                "symptoms": {"high_fever", "dry_cough", "loss_of_smell", "fatigue", "shortness_of_breath"},
                "severity": "Moderate to Severe",
                "treatment": "Isolation, pulse oximetry monitoring, consult physician."
            },
            "Allergic Rhinitis": {
                "symptoms": {"sneezing", "runny_nose", "itchy_eyes", "nasal_congestion"},
                "severity": "Mild",
                "treatment": "Antihistamines and avoidance of known allergens."
            },
            "Bacterial Pneumonia": {
                "symptoms": {"high_fever", "chest_pain", "shortness_of_breath", "productive_cough", "chills"},
                "severity": "Severe",
                "treatment": "Immediate medical attention and broad-spectrum antibiotics."
            }
        }

    def diagnose(self, patient_symptoms):
        """Calculates match score (% symptoms matched) for each disease."""
        patient_symptoms = set(patient_symptoms)
        results = []

        for disease, info in self.disease_kb.items():
            req_symptoms = info["symptoms"]
            matched = patient_symptoms.intersection(req_symptoms)
            match_percentage = (len(matched) / len(req_symptoms)) * 100

            results.append({
                "disease": disease,
                "confidence": round(match_percentage, 1),
                "matched_symptoms": list(matched),
                "severity": info["severity"],
                "treatment": info["treatment"]
            })

        # Sort by confidence score descending
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def visualize(self, output_file):
        """Visualizes symptom-to-disease diagnostic network."""
        fig, ax = plt.subplots(figsize=(12, 7.5))
        ax.set_title("Task 2: Medical Knowledge Representation - Diagnostic Mapping", fontsize=13, fontweight="bold", pad=15)
        ax.set_facecolor("#FAFAFA")

        symptoms = [
            "high_fever", "body_ache", "loss_of_smell", "dry_cough",
            "sneezing", "runny_nose", "shortness_of_breath", "chest_pain"
        ]
        diseases = ["Common Cold", "Influenza (Flu)", "COVID-19", "Allergic Rhinitis", "Bacterial Pneumonia"]

        # Coordinates
        y_sym = [0.85 - i * 0.1 for i in range(len(symptoms))]
        y_dis = [0.80 - i * 0.16 for i in range(len(diseases))]

        sym_pos = {s: (0.2, y) for s, y in zip(symptoms, y_sym)}
        dis_pos = {d: (0.8, y) for d, y in zip(diseases, y_dis)}

        # Draw symptom boxes
        for s, (x, y) in sym_pos.items():
            box = patches.FancyBboxPatch(
                (x - 0.12, y - 0.035), 0.24, 0.07,
                boxstyle="round,pad=0.015,rounding_size=0.02",
                facecolor="#FEF08A", edgecolor="#CA8A04", linewidth=1.2
            )
            ax.add_patch(box)
            ax.text(x, y, s.replace("_", " ").title(), ha="center", va="center", fontsize=8, fontweight="bold")

        # Draw disease boxes
        dis_colors = ["#BFDBFE", "#FED7AA", "#FCA5A5", "#BBF7D0", "#E9D5FF"]
        for (d, (x, y)), col in zip(dis_pos.items(), dis_colors):
            box = patches.FancyBboxPatch(
                (x - 0.14, y - 0.045), 0.28, 0.09,
                boxstyle="round,pad=0.015,rounding_size=0.02",
                facecolor=col, edgecolor="#374151", linewidth=1.2
            )
            ax.add_patch(box)
            sev = self.disease_kb[d]["severity"]
            ax.text(x, y + 0.01, d, ha="center", va="center", fontsize=8.5, fontweight="bold")
            ax.text(x, y - 0.02, f"Severity: {sev}", ha="center", va="center", fontsize=7, color="#4B5563")

        # Draw connections
        for d, info in self.disease_kb.items():
            dx, dy = dis_pos[d]
            for s in info["symptoms"]:
                if s in sym_pos:
                    sx, sy = sym_pos[s]
                    ax.plot([sx + 0.12, dx - 0.14], [sy, dy], color="#94A3B8", lw=1.0, alpha=0.7)

        # Labels
        ax.text(0.2, 0.95, "Clinical Symptoms (Observations)", ha="center", va="center", fontsize=10, fontweight="bold", color="#854D0E")
        ax.text(0.8, 0.95, "Disease Classifications (Hypotheses)", ha="center", va="center", fontsize=10, fontweight="bold", color="#1E3A8A")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"[Task 2] Diagram saved successfully to: {output_file}")


def main():
    print("=" * 65)
    print("  TASK 2: MEDICAL KNOWLEDGE REPRESENTATION & DIAGNOSIS")
    print("=" * 65)

    med_kb = MedicalKnowledgeBase()

    # Patient Case Study
    patient_name = "Patient P-102"
    observed_symptoms = ["high_fever", "loss_of_smell", "dry_cough", "fatigue"]
    print(f"Case Study: {patient_name}")
    print(f"Reported Symptoms: {', '.join(s.replace('_', ' ') for s in observed_symptoms)}\n")

    diagnoses = med_kb.diagnose(observed_symptoms)
    print("Differential Diagnostic Rankings:")
    print(f"{'Disease':<24} | {'Confidence':<12} | {'Severity':<16} | Matched Symptoms")
    print("-" * 75)
    for res in diagnoses:
        matched = ", ".join(res["matched_symptoms"]) if res["matched_symptoms"] else "None"
        print(f"{res['disease']:<24} | {res['confidence']:>8.1f} %  | {res['severity']:<16} | {matched}")

    top_diagnosis = diagnoses[0]
    print(f"\nPrimary Diagnosis: [{top_diagnosis['disease']}] (Confidence: {top_diagnosis['confidence']}%)")
    print(f"Recommended Treatment Protocol: {top_diagnosis['treatment']}\n")

    output_path = os.path.join(os.path.dirname(__file__), "outputs", "medical_knowledgerepresentation.png")
    med_kb.visualize(output_path)
    print("=" * 65)


if __name__ == "__main__":
    main()

"""
Lab 5 - Application: Intelligent Decision Support System (IDSS)
Author: Divyansh Verma (IC-2K23-31)
Course: IC-709D / IC-703 C | Artificial Intelligence Laboratory (Unit 5, CO4, CO5)

Description:
A modular Intelligent Decision Support System (IDSS) integrating trained
Machine Learning models (Decision Trees) across three critical domains:
1. Education & Career: Student Placement Likelihood & Career Advisory
2. Healthcare: Metabolic & Cardiovascular Disease Risk Screening
3. Banking & Finance: Loan Creditworthiness Assessment
Provides automated recommendations, decision rules, and confidence probabilities.
"""

import os
import joblib
import pandas as pd
import numpy as np


class IntelligentDecisionSupportSystem:
    def __init__(self):
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        self.placement_model_path = os.path.join(models_dir, "placement_decision_tree.pkl")
        self.disease_model_path = os.path.join(models_dir, "disease_decision_tree.pkl")
        self.loan_model_path = os.path.join(models_dir, "loan_decision_tree.pkl")

        # Load models
        self.placement_model = joblib.load(self.placement_model_path) if os.path.exists(self.placement_model_path) else None
        self.disease_model = joblib.load(self.disease_model_path) if os.path.exists(self.disease_model_path) else None
        loan_pack = joblib.load(self.loan_model_path) if os.path.exists(self.loan_model_path) else None
        self.loan_model = loan_pack["model"] if loan_pack else None
        self.loan_encoder = loan_pack["label_encoder"] if loan_pack else None

    # -------------------------------------------------------------------------
    # 1. Domain: Student Placement Advisory
    # -------------------------------------------------------------------------
    def evaluate_placement(self, cgpa, iq, internships, work_exp_str):
        if not self.placement_model:
            return "Placement model not loaded. Please train model first."

        work_exp = 1 if str(work_exp_str).strip().lower() in ["yes", "1", "y"] else 0
        input_data = pd.DataFrame([[cgpa, iq, internships, work_exp]],
                                  columns=["cgpa", "iq", "internships", "work_experience"])

        pred = self.placement_model.predict(input_data)[0]
        prob = self.placement_model.predict_proba(input_data)[0]
        confidence = prob[pred] * 100

        result = {
            "Decision": "Likely to be Placed" if pred == 1 else "Needs Improvement / At Risk",
            "Confidence": f"{confidence:.1f}%",
            "Probability_Placed": f"{prob[1]*100:.1f}%",
            "Recommendation": (
                "Profile is competitive. Recommend targeting tier-1 campus placements."
                if pred == 1 else
                "Recommend enhancing CGPA (target > 7.5), building practical projects, and pursuing internships."
            )
        }
        return result

    # -------------------------------------------------------------------------
    # 2. Domain: Clinical Disease Risk Screening
    # -------------------------------------------------------------------------
    def evaluate_disease_risk(self, age, glucose, blood_pressure, bmi):
        if not self.disease_model:
            return "Disease model not loaded. Please train model first."

        input_data = pd.DataFrame([[age, glucose, blood_pressure, bmi]],
                                  columns=["age", "glucose", "blood_pressure", "bmi"])

        pred = self.disease_model.predict(input_data)[0]
        prob = self.disease_model.predict_proba(input_data)[0]
        confidence = prob[pred] * 100

        result = {
            "Diagnostic_Screening": "High Risk / Disease Detected" if pred == 1 else "Normal / Low Risk",
            "Confidence": f"{confidence:.1f}%",
            "Disease_Probability": f"{prob[1]*100:.1f}%",
            "Clinical_Advice": (
                "Glucose and BMI indicators warrant medical evaluation. Recommend HbA1c screening."
                if pred == 1 else
                "Vitals fall within normal range. Maintain balanced diet and routine health checkups."
            )
        }
        return result

    # -------------------------------------------------------------------------
    # 3. Domain: Banking Creditworthiness & Loan Approval
    # -------------------------------------------------------------------------
    def evaluate_loan(self, income, loan_amount_k, credit_history, education_str):
        if not self.loan_model:
            return "Loan model not loaded. Please train model first."

        edu_encoded = 0 if "grad" in education_str.lower() and "not" not in education_str.lower() else 1
        input_data = pd.DataFrame([[income, loan_amount_k, int(credit_history), edu_encoded]],
                                  columns=["applicant_income", "loan_amount_k", "credit_history", "education"])

        pred = self.loan_model.predict(input_data)[0]
        prob = self.loan_model.predict_proba(input_data)[0]
        confidence = prob[pred] * 100

        result = {
            "Underwriting_Decision": "Approved" if pred == 1 else "Rejected",
            "Confidence": f"{confidence:.1f}%",
            "Approval_Probability": f"{prob[1]*100:.1f}%",
            "Financial_Rationale": (
                "Credit history is satisfactory and income-to-loan ratio meets underwriting criteria."
                if pred == 1 else
                "High default risk detected. Insufficient credit score or inadequate debt-to-income margin."
            )
        }
        return result


def main():
    print("=" * 70)
    print("   INTELLIGENT DECISION SUPPORT SYSTEM (IDSS) - UNIT 5")
    print("=" * 70)

    idss = IntelligentDecisionSupportSystem()

    # Case 1: Student Placement Advisory Consultation
    print("\n[Case 1: Student Placement Decision Support]")
    student_profile = {"cgpa": 8.3, "iq": 118, "internships": 2, "work_exp_str": "Yes"}
    print(f"Candidate Profile: {student_profile}")
    res1 = idss.evaluate_placement(**student_profile)
    for k, v in res1.items():
        print(f"  * {k:22s}: {v}")

    # Case 2: Clinical Screening Consultation
    print("\n[Case 2: Healthcare Disease Screening Decision Support]")
    patient_vitals = {"age": 52, "glucose": 158.0, "blood_pressure": 88.0, "bmi": 32.4}
    print(f"Patient Profile: {patient_vitals}")
    res2 = idss.evaluate_disease_risk(**patient_vitals)
    for k, v in res2.items():
        print(f"  * {k:22s}: {v}")

    # Case 3: Banking Loan Approval Consultation
    print("\n[Case 3: Banking Credit Underwriting Decision Support]")
    applicant_data = {"income": 6200, "loan_amount_k": 120, "credit_history": 1, "education_str": "Graduate"}
    print(f"Applicant Profile: {applicant_data}")
    res3 = idss.evaluate_loan(**applicant_data)
    for k, v in res3.items():
        print(f"  * {k:22s}: {v}")

    print("\n" + "=" * 70)
    print(" Intelligent Decision Support Consultations Executed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

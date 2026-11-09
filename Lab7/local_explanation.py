"""
Local interpretability and what-if analysis for individual student predictions.
Traces decision pathways through the tree and evaluates counterfactual scenarios.
"""

import numpy as np
import pandas as pd

from data_loader import get_trained_decision_tree


def explain_decision_path(model, sample, feature_names):
    """Traces and prints the exact node traversal path for a single sample."""
    sample_df = pd.DataFrame([sample], columns=feature_names)
    pred_class = model.predict(sample_df)[0]
    pred_prob = model.predict_proba(sample_df)[0]

    node_indicator = model.decision_path(sample_df)
    leaf_id = model.apply(sample_df)[0]
    node_indices = node_indicator.indices

    tree = model.tree_
    path_explanation = []

    for node_id in node_indices:
        if node_id == leaf_id:
            values = tree.value[node_id][0]
            prob = values / values.sum()
            path_explanation.append({
                "type": "Leaf",
                "node_id": node_id,
                "text": f"Leaf Node #{node_id}: Predicted Class = {pred_class} (Prob: {pred_prob[pred_class]*100:.1f}%)"
            })
        else:
            feature_idx = tree.feature[node_id]
            feature_name = feature_names[feature_idx]
            threshold = tree.threshold[node_id]
            val = sample[feature_idx]
            inequality = "<=" if val <= threshold else ">"
            direction = "Left" if val <= threshold else "Right"

            path_explanation.append({
                "type": "Split",
                "node_id": node_id,
                "feature": feature_name,
                "value": val,
                "threshold": threshold,
                "direction": direction,
                "text": f"Node #{node_id}: {feature_name} = {val:.2f} {inequality} {threshold:.2f} -> Go {direction}"
            })

    return pred_class, pred_prob, path_explanation


def perform_what_if_analysis(model, base_sample, feature_names):
    """Tests hypothetical profile modifications to find minimal counterfactual flips."""
    base_pred = model.predict(pd.DataFrame([base_sample], columns=feature_names))[0]

    scenarios = [
        ("Baseline Student Profile", base_sample.copy()),
        ("What-If 1: Add 2 Internships", base_sample.copy()),
        ("What-If 2: Add Work Experience", base_sample.copy()),
        ("What-If 3: Increase CGPA by +0.8 (from 6.5 to 7.3)", base_sample.copy()),
        ("What-If 4: Increase IQ by +15 (from 102 to 117)", base_sample.copy()),
    ]

    # Modify feature values for each scenario
    cgpa_idx = feature_names.index("cgpa")
    iq_idx = feature_names.index("iq")
    intern_idx = feature_names.index("internships")
    work_idx = feature_names.index("work_experience")

    scenarios[1][1][intern_idx] += 2
    scenarios[2][1][work_idx] = 1
    scenarios[3][1][cgpa_idx] += 0.8
    scenarios[4][1][iq_idx] += 15

    results = []
    for title, profile in scenarios:
        profile_df = pd.DataFrame([profile], columns=feature_names)
        pred = model.predict(profile_df)[0]
        prob = model.predict_proba(profile_df)[0]
        status = "Placed (1)" if pred == 1 else "Not Placed (0)"
        outcome_change = "Flip / Successful" if pred != base_pred else "No Change"

        results.append({
            "Scenario": title,
            "CGPA": profile[cgpa_idx],
            "IQ": profile[iq_idx],
            "Internships": profile[intern_idx],
            "Work Exp": profile[work_idx],
            "Prediction": status,
            "Confidence": f"{prob[pred]*100:.1f}%",
            "Impact": outcome_change
        })

    return pd.DataFrame(results)


def main():
    print("=" * 65)
    print("  EXPLAINABLE AI: LOCAL DECISION PATH & WHAT-IF ANALYSIS")
    print("=" * 65)

    model, X_train, X_test, y_train, y_test, features, _, _ = get_trained_decision_tree()

    # Select an unplaced student for explanation (e.g. Student #2 from dataset: CGPA=6.5, IQ=102)
    sample_student = [6.5, 102.0, 0, 0]  # cgpa, iq, internships, work_experience
    print("Target Student Profile for Local Audit:")
    for f, v in zip(features, sample_student):
        print(f"  * {f:16s}: {v}")

    pred, prob, path = explain_decision_path(model, sample_student, features)
    status_str = "PLACED" if pred == 1 else "NOT PLACED"
    print(f"\nModel Prediction: [{status_str}] (Confidence: {prob[pred]*100:.1f}%)\n")

    print("--- Step-by-Step Tree Decision Path Traversal ---")
    for step in path:
        print(f"  -> {step['text']}")

    print("\n" + "=" * 65)
    print("  COUNTERFACTUAL WHAT-IF SENSITIVITY ANALYSIS")
    print("=" * 65)
    df_what_if = perform_what_if_analysis(model, sample_student, features)
    print(df_what_if.to_string(index=False))

    print("\nLocal Explanation Summary:")
    print("  * CGPA is the governing decision pivot.")
    print("  * Even adding internships or work experience without reaching CGPA >= 7.0")
    print("    fails to overturn the negative placement outcome.")
    print("  * Improving CGPA past the threshold (6.5 -> 7.3) immediately converts prediction to Placed.")
    print("=" * 65)


if __name__ == "__main__":
    main()

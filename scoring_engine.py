def evaluate_understanding(similarity_score):
    if similarity_score >= 75:
        return "Strong Understanding", "green"
    elif similarity_score >= 50:
        return "Moderate Understanding", "orange"
    else:
        return "Poor Understanding", "red"
from scripts.build_failure_taxonomy import auto_taxonomy


def _passing_dimensions() -> dict[str, bool]:
    return {
        "dim_task_completion": True,
        "dim_data_retrieval_accuracy": True,
        "dim_agent_sequence_correct": True,
        "dim_hallucinations": False,
        "dim_generalized_result_verification": True,
        "dim_clarity_and_justification": True,
    }


def test_auto_taxonomy_treats_hallucination_false_as_passing() -> None:
    assert auto_taxonomy(_passing_dimensions()) == ("low_unknown", [])


def test_auto_taxonomy_treats_hallucination_true_as_failure() -> None:
    row = _passing_dimensions()
    row["dim_hallucinations"] = True

    assert auto_taxonomy(row) == ("low_hallucinations", ["hallucinations"])

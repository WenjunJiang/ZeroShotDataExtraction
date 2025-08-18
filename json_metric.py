"""
All columns of the solution and submission dataframes are passed to your metric, except for the Usage column.

Your metric must satisfy the following constraints:
- You must have a function named score. Kaggle's evaluation system will call that function.
- You can add your own arguments to score, but you cannot change the first three (solution, submission, and row_id_column_name).
- All arguments for score must have type annotations.
- score must return a single, finite, non-null float.
"""

import pandas as pd
import pandas.api.types
import math
import json
import numpy as np


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def to_json(s):
    """
    Converts input string to JSON-like Python object, if fails returns empty string.

    **Returns:**
        - Python dictionary or empty string

    **Examples:**

    >>> to_json('{"a": 10, "b":12}')
    {'a': 10, 'b': 12}
    """
    try:
        s_json = json.loads(s)
    except:
        print(f'Error converting to JSON: {s}')
        s_json = ''

    return s_json


def compare_json(a, b, key_weight=0.25, value_weight=0.75, alpha=1.):
    """
    Recursively computes a similarity score between two JSON-like Python objects.

    The similarity score ranges from 0 to 1, where:
    - 1 indicates that the objects are identical.
    - 0 indicates no similarity.

    **Comparison Rules:**

    - **Numerical Values (int, float):**
      - Similarity is calculated using an exponential decay function based on the absolute difference:
        \[
        \text{Similarity} = e^{-\alpha \times |a - b|}
        \]
      - **alpha** controls the rate at which similarity decreases as the difference increases.

    - **Strings:**
      - Similarity is the proportion of matching characters at the same positions relative to the length of the longer string:
        \[
        \text{Similarity} = \frac{\text{Number of Matching Characters}}{\text{Length of Longer String}}
        \]

    - **Booleans:**
      - Similarity is 1 if both values are the same, 0 otherwise.

    - **NoneType:**
      - Similarity is 1 if both values are `None`, 0 otherwise.

    - **Different Data Types:**
      - If the types of `a` and `b` differ, the similarity is 0.

    - **Dictionaries:**
      - **Key Similarity:** Calculated as the ratio of shared keys to total unique keys.
      - **Value Similarity:** Recursively computed similarities of the values corresponding to shared keys.
      - **Combined Similarity:** Weighted sum of key similarity and value similarity:
        \[
        \text{Similarity} = (\text{key\_weight} \times \text{Key Similarity}) + (\text{value\_weight} \times \text{Value Similarity})
        \]

    - **Lists:**
      - Elements are compared recursively up to the length of the shorter list.
      - Similarity is adjusted for lists of different lengths by multiplying by the ratio of the shorter length to the longer length.

    **Parameters:**

    - **a**: The first JSON-like object to compare. Can be a `dict`, `list`, `int`, `float`, `str`, `bool`, or `None`.
    - **b**: The second JSON-like object to compare.
    - **key_weight** *(float, optional)*: Weight for the key similarity in dictionaries. Defaults to `0.25`.
    - **value_weight** *(float, optional)*: Weight for the value similarity in dictionaries. Defaults to `0.75`.
    - **alpha** *(float, optional)*: Decay rate for numerical similarity. Higher values make the similarity decrease more rapidly with increasing difference. Defaults to `1.0`.

    **Returns:**

    - **similarity** *(float)*: A float between `0` and `1` representing the similarity between the two JSON objects.

    **Examples:**

    >>> compare_json(10, 10)
    1.0

    >>> compare_json(10, 12)   # Similarity = exp(-1.0 * abs(10 - 12))
    0.1353352832366127

    >>> compare_json("hello", "hallo") # Matches at positions 0 ('h') and 2 ('l') and 1 ('o')
    0.8

    >>> compare_json(True, True)
    1.0

    >>> compare_json(None, None)
    1.0

    >>> compare_json({"a": 1, "b": 2}, {"a": 1, "b": 3})
    0.7629547904392908

    >>> compare_json([1, 2, 3], [1, 2, 4])
    0.6666666666666666

    **Notes:**

    - The function is designed to handle nested structures and will recursively compare elements within dictionaries and lists.
    - Adjusting the `key_weight` and `value_weight` allows you to prioritize matching keys over matching values in dictionaries.
    - The `alpha` parameter allows tuning how quickly the similarity decreases for numerical differences; a higher `alpha` means that even small differences lead to low similarity.

    **Usage Tips:**

    - Ensure that the input objects are JSON-serializable Python data types.
    - For best results, pre-process your data to handle any special types or structures not covered by this function.

    **Limitations:**

    - The function does not handle custom objects or data types outside of standard JSON types.
    - For large and deeply nested structures, the function may have performance considerations due to recursion depth.

    """
    # Base case for numerical values
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        abs_diff = abs(a - b)
        similarity = math.exp(-alpha * abs_diff)
        return similarity

    # Base case for strings
    elif isinstance(a, str) and isinstance(b, str):
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.  # Both strings are empty
        match_count = sum(1 for c1, c2 in zip(a, b) if c1 == c2)
        similarity = match_count / max_len
        return similarity

    # Base case for booleans
    elif isinstance(a, bool) and isinstance(b, bool):
        return 1. if a == b else 0.

    # Base case for None
    elif a is None and b is None:
        return 1.

    # Different types
    elif type(a) != type(b):
        return 0.

    # Recursive case for dictionaries
    elif isinstance(a, dict) and isinstance(b, dict):
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        all_keys = keys_a | keys_b
        if not all_keys:
            return 1.  # Both dicts are empty

        key_similarity = len(keys_a & keys_b) / len(all_keys)

        value_similarities = []
        for key in all_keys:
            if key in a and key in b:
                sim = compare_json(a[key], b[key], key_weight, value_weight, alpha)
                value_similarities.append(sim)
            else:
                value_similarities.append(0)

        value_similarity = sum(value_similarities) / len(value_similarities)

        # Combined similarity
        similarity = key_weight * key_similarity + value_weight * value_similarity
        return similarity

    # Recursive case for lists
    elif isinstance(a, list) and isinstance(b, list):
        a = set(a)
        b = set(b)
        mutual_items = a & b
        len_a = len(a)
        len_b = len(b)
        max_len = max(len_a, len_b)
        if max_len == 0:
            return 1.  # Both lists are empty
        else:
            return len(mutual_items) / max_len
    else:
        return 0.  # Unsupported data types or mismatch


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Computes the average similarity score between two DataFrames containing JSON-like data.

    This function compares corresponding JSON objects in the `solution` and `submission` DataFrames row by row and column by column. It calculates a similarity score for each pair of JSON objects and returns the overall average similarity score ranging from 0 to 1.

    **Parameters:**

    - **solution** *(pd.DataFrame)*: The DataFrame containing the ground truth JSON objects.
    - **submission** *(pd.DataFrame)*: The DataFrame containing the submitted JSON objects to be evaluated.
    - **row_id_column_name** *(str)*: The name of the column used as the row identifier, which will be removed before computation.

    **Returns:**

    - **float**: The average similarity score between the `solution` and `submission` DataFrames.

    Example:
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> json1 = """{"a": 10, "b": "Test", "c": {"a": -312.414, "z": [1,2,3]}}"""
    >>> json2 = """{"a": 0, "b": "Test", "c": {"a": -312.414, "z": [3,2,1]}}"""
    >>> y_pred = [json1, json2]
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))
    >>> y_true = [json2, json1]
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.7500113499824406
    '''

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    matched_cols = []
    for col in submission.columns:
        if col in solution.columns:
            matched_cols.append(col)
    if len(matched_cols) != len(solution.columns):
        raise ParticipantVisibleError(
            f'Submission does not contains expected columns: {",".join([x for x in solution.columns])}')

    total = 0.
    for match in solution.columns:
        res = 0.
        for sub, sol in zip(submission[match].values, solution[match].values):
            sub = to_json(sub)
            sol = to_json(sol)
            score = compare_json(sub, sol)
            res += float(score)
        res = res / len(solution)
        total += res
    return total / len(matched_cols)
#!/usr/bin/env python3
"""Merge rank outputs and compute InfoVQA EM, word-level F1, and ANLS."""

import argparse
import glob
import json
import math
import re
import string
from collections import Counter


def normalize(text):
    text = str(text).lower().strip()
    text = "".join(char for char in text if char not in string.punctuation)
    return " ".join(text.split())


def answers(value):
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return [normalize(value)]


def token_f1(prediction, target):
    predicted = normalize(prediction).split()
    expected = normalize(target).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def levenshtein(left, right):
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def anls(prediction, target):
    prediction = normalize(prediction)
    target = normalize(target)
    if not prediction or not target:
        return float(prediction == target)
    distance = levenshtein(prediction, target)
    score = 1 - distance / max(len(prediction), len(target))
    return score if score >= 0.5 else 0.0


def best_scores(prediction, targets):
    normalized_prediction = normalize(prediction)
    return {
        "em": max(float(normalized_prediction == target) for target in targets),
        "f1": max(token_f1(normalized_prediction, target) for target in targets),
        "anls": max(anls(normalized_prediction, target) for target in targets),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    history_paths = glob.glob(
        f"{args.input_dir}/rank_*/Qwen2.5-VL-7B-Instruct/**/*_history.jsonl",
        recursive=True,
    )
    if not history_paths:
        raise FileNotFoundError(f"No rank history files found under {args.input_dir}")

    records = {}
    for path in history_paths:
        with open(path, encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                qid = record["qid"]
                if qid in records:
                    raise ValueError(f"Duplicate qid found across rank outputs: {qid}")
                records[qid] = record

    totals = {"em": 0.0, "f1": 0.0, "anls": 0.0}
    for record in records.values():
        scores = best_scores(
            record["original_responds"], answers(record["original_answer"])
        )
        for metric, score in scores.items():
            totals[metric] += score

    count = len(records)
    metrics = {
        "count": count,
        "em": totals["em"] / count if count else math.nan,
        "word_level_f1": totals["f1"] / count if count else math.nan,
        "anls": totals["anls"] / count if count else math.nan,
    }
    with open(args.output_file, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

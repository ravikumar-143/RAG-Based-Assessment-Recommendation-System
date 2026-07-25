"""
LLM-based reranking using Ollama.
"""

from __future__ import annotations

import ast
import json
import math
from typing import Any, Dict, List

from ollama import chat

from reranker.domain_detection import detect_domains
from utils import get_logger

logger = get_logger(__name__)


TEST_TYPE_DOMAIN_MAP = {
    "K": "technical",
    "S": "technical",
    "E": "technical",
    "P": "behavioral",
    "B": "behavioral",
    "C": "cognitive",
    "A": "cognitive",
    "D": "cognitive",
}
REPORT_KEYWORDS = [
    "report",
    "participant report",
    "manager report",
    "profile report",
    "narrative report",
    "job profiling guide",
    "development action planner",
    "opq",
    "remoteworkq",
]


def _filter_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove report-type items before LLM reranking.
    """

    filtered = []

    for item in candidates:

        name = item.get("name", "").lower()

        if any(keyword in name for keyword in REPORT_KEYWORDS):
            continue

        filtered.append(item)

    logger.info(
        f"Filtered {len(candidates) - len(filtered)} report items."
    )

    return filtered


def _clean_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Prepare candidate objects before sending them to the LLM.
    """

    clean_candidates = []

    for item in candidates[:10]:

        obj = item.copy()

        # Convert string representation of list to list
        if isinstance(obj.get("test_type"), str):

            try:
                obj["test_type"] = ast.literal_eval(obj["test_type"])

            except Exception:
                obj["test_type"] = [obj["test_type"]]

        # Convert NaN duration to None
        if isinstance(obj.get("duration"), float):

            if math.isnan(obj["duration"]):
                obj["duration"] = None

        # Remove similarity score if present
        obj.pop("score", None)

        clean_candidates.append(obj)

    return clean_candidates


def _extract_json(text: str):
    """
    Extract JSON returned by Ollama.

    Handles responses like:

    ```json
    {...}
    ```

    or plain JSON.
    """

    text = text.strip()

    if text.startswith("```"):

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    return json.loads(text)


def _llm_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Use Ollama to rerank SHL assessments.
    """

    clean_candidates = _clean_candidates(candidates)


    prompt = f"""
You are an expert SHL recruitment assistant.

Your task is to rank the candidate assessments based on how well they match the user's hiring requirement.

User Query:
{query}

Candidate Assessments:
{json.dumps(clean_candidates, indent=2)}

Rules:

1. Use ONLY these assessments.
2. Do NOT invent assessments.
3. Do NOT remove fields.
4. Preserve every value exactly.
5. Rank from best to worst.
6. Return at most 10.
7. Return ONLY JSON.

Return exactly:

{{
  "recommended_assessments":[
    {{
      "url":"",
      "name":"",
      "adaptive_support":"",
      "description":"",
      "duration":null,
      "remote_support":"",
      "test_type":["K"]
    }}
  ]
}}
"""

    try:

        logger.info("Calling Ollama...")

        logger.info("Candidates sent to Ollama:")

        logger.info(json.dumps(clean_candidates[:3], indent=2))

        response = chat(
            model="qwen2.5:1.5b",
            format="json",
            options={
                "temperature": 0,
            },
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        text = response["message"]["content"]

        logger.info("Ollama response:")
        logger.info(text)

        data = _extract_json(text)

        if isinstance(data, list):

            ranked = data

        elif isinstance(data, dict):

            if "recommended_assessments" in data:

                ranked = data["recommended_assessments"]

            elif "assessments" in data:

                ranked = data["assessments"]

            else:

                raise ValueError(
                    "Unknown JSON structure returned by Ollama."
                )

        else:

            raise ValueError("Invalid JSON returned by Ollama.")

        # Remove duplicate assessments
        unique = []
        seen = set()

        for item in ranked:

            key = item.get("url") or item.get("name")

            if key in seen:
                continue

            seen.add(key)
            unique.append(item)

        logger.info(
            "Ollama reranking successful."
        )

        return unique[:10]

    except Exception as exc:

        logger.warning(
            f"Ollama reranking failed: {exc}"
        )

        if strict_mode:
            raise

        logger.warning(
            "Using FAISS retrieval order."
        )

        return candidates[:10]


def _balance_domains(
    query: str,
    ranked: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Balance technical, behavioral and cognitive assessments.
    """

    detected = detect_domains(query)

    active_domains = [
        domain
        for domain, score in detected.items()
        if score > 0
    ]

    if len(active_domains) < 2:
        return ranked[:10]

    buckets = {
        "technical": [],
        "behavioral": [],
        "cognitive": [],
        "other": [],
    }

    for item in ranked:

        test_type = item.get(
            "test_type",
            [],
        )

        if isinstance(test_type, str):
            test_type = [test_type]

        domains = {
            TEST_TYPE_DOMAIN_MAP.get(
                code,
                "other",
            )
            for code in test_type
        }

        for domain in domains:
            buckets[domain].append(item)

    balanced = []

    indexes = {
        key: 0
        for key in buckets
    }

    while len(balanced) < 10:

        added = False

        for domain in active_domains:

            if indexes[domain] < len(
                buckets[domain]
            ):

                item = buckets[domain][
                    indexes[domain]
                ]

                indexes[domain] += 1

                if item not in balanced:
                    balanced.append(item)

                added = True

                if len(balanced) >= 10:
                    break

        if not added:
            break

    for item in ranked:

        if len(balanced) >= 10:
            break

        if item not in balanced:
            balanced.append(item)

    return balanced[:10]
def rerank_assessments(
    query: str,
    candidates: List[Dict[str, Any]],
    apply_balance: bool = True,
    strict_mode: bool = False,
    enable_llm: bool = True,
) -> List[Dict[str, Any]]:
    """
    Main reranking pipeline.
    """

    if not candidates:
        return []

    # Remove report-type assessments before reranking
    candidates = _filter_candidates(candidates)

    if enable_llm:

        ranked = _llm_rerank(
            query=query,
            candidates=candidates,
            strict_mode=strict_mode,
        )

    else:

        ranked = candidates[:10]

    if apply_balance:
        ranked = _balance_domains(
            query,
            ranked,
        )

    # Final duplicate removal
    final_results = []
    seen = set()

    for item in ranked:

        key = item.get("url") or item.get("name")

        if key in seen:
            continue

        seen.add(key)

        final_results.append(
            {
                "url": item.get("url", ""),
                "name": item.get("name", ""),
                "adaptive_support": item.get(
                    "adaptive_support",
                    "No",
                ),
                "description": item.get(
                    "description",
                    "",
                ),
                "duration": item.get(
                    "duration",
                    None,
                ),
                "remote_support": item.get(
                    "remote_support",
                    "No",
                ),
                "test_type": item.get(
                    "test_type",
                    [],
                ),
            }
        )

        if len(final_results) >= 10:
            break

    logger.info(
        f"Returning {len(final_results)} reranked assessments."
    )

    return final_results
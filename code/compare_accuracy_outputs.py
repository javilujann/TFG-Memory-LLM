#!/usr/bin/env python3
"""
Compara dos archivos *_final_results.json por el evaluador de accuracy,
pregunta por pregunta.

Reglas de salida por question_id:
- Si ambos están bien: no imprime nada.
- Si uno está bien y otro mal: indica cuál está correcta.
- Si ambos están mal: lo indica.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_accuracy_evaluator(results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    evaluators = results.get("evaluators", {})
    if not isinstance(evaluators, dict) or not evaluators:
        raise ValueError("No se encontró el bloque 'evaluators' en el JSON.")

    for key, evaluator_data in evaluators.items():
        if not isinstance(evaluator_data, dict):
            continue

        per_question = evaluator_data.get("per_question_results")
        if not isinstance(per_question, list) or not per_question:
            continue

        has_correct_flag = any(
            isinstance(item, dict) and isinstance(item.get("correct"), bool)
            for item in per_question
        )
        if not has_correct_flag:
            continue

        key_lower = key.lower()
        if "llm_judge" in key_lower or "llmjudge" in key_lower:
            return key, evaluator_data

    available = ", ".join(evaluators.keys())
    raise ValueError(
        "No se encontró un evaluador LLM Judge en 'evaluators'. "
        f"Disponibles: {available}"
    )


def extract_correct_map(evaluator_data: Dict[str, Any]) -> Dict[str, bool]:
    per_question = evaluator_data.get("per_question_results", [])
    correct_map: Dict[str, bool] = {}

    for item in per_question:
        if not isinstance(item, dict):
            continue
        question_id = item.get("question_id")
        correct = item.get("correct")
        if isinstance(question_id, str) and isinstance(correct, bool):
            correct_map[question_id] = correct

    if not correct_map:
        raise ValueError("El evaluador no contiene resultados válidos por pregunta (question_id/correct).")

    return correct_map


def compare_questions(
    map_a: Dict[str, bool],
    map_b: Dict[str, bool],
    name_a: str,
    name_b: str,
) -> Iterable[str]:
    shared_questions = sorted(set(map_a.keys()) & set(map_b.keys()))

    for question_id in shared_questions:
        a_correct = map_a[question_id]
        b_correct = map_b[question_id]

        if a_correct and b_correct:
            continue

        if a_correct and not b_correct:
            yield f"{question_id} | correcta: {name_a}"
        elif b_correct and not a_correct:
            yield f"{question_id} | correcta: {name_b}"
        else:
            yield f"{question_id} | ambas mal"


def main() -> None:
    # Configuración de argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a", type=str, help="Primer archivo JSON de resultados")
    parser.add_argument("file_b", type=str, help="Segundo archivo JSON de resultados")

    args = parser.parse_args()

    # Lectura y validación de archivos
    file_a = Path(args.file_a)
    file_b = Path(args.file_b)

    if not file_a.exists():
        raise FileNotFoundError(f"No existe el archivo: {file_a}")
    if not file_b.exists():
        raise FileNotFoundError(f"No existe el archivo: {file_b}")

    data_a = load_json(file_a)
    data_b = load_json(file_b)

    # Saca del json el bloque del evaluador de accuracy 
    eval_key_a, eval_data_a = find_accuracy_evaluator(data_a)
    eval_key_b, eval_data_b = find_accuracy_evaluator(data_b)

    # Extrae el nombre del experimento o usa el nombre del archivo sin extensión
    name_a = data_a.get("experiment_name") or file_a.stem
    name_b = data_b.get("experiment_name") or file_b.stem

    # Extrae el mapa de question_id a correct para ambos evaluadores
    map_a = extract_correct_map(eval_data_a)
    map_b = extract_correct_map(eval_data_b)

    only_a = sorted(set(map_a.keys()) - set(map_b.keys()))
    only_b = sorted(set(map_b.keys()) - set(map_a.keys()))

    lines = list(
        compare_questions(
            map_a=map_a,
            map_b=map_b,
            name_a=name_a,
            name_b=name_b,
        )
    )

    print(f"Evaluador archivo A: {eval_key_a}")
    print(f"Evaluador archivo B: {eval_key_b}")
    print(f"Preguntas en común: {len(set(map_a.keys()) & set(map_b.keys()))}")

    if only_a:
        print(f"Aviso: {len(only_a)} preguntas solo en A (se ignoran en comparación).")
    if only_b:
        print(f"Aviso: {len(only_b)} preguntas solo en B (se ignoran en comparación).")

    if not lines:
        print("No hay diferencias a reportar con las reglas seleccionadas.")
        return

    print("\nComparación por pregunta:")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()

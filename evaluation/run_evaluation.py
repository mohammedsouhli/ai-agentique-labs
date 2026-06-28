"""
run_evaluation.py - Évaluation du système sur 20 questions
Teste l'agent sur 10 questions simples et 10 questions complexes,
mesure le temps de réponse et affiche les statistiques.
"""

import time
import statistics
from langchain_core.messages import HumanMessage
from questions import QUESTIONS_SIMPLES, QUESTIONS_COMPLEXES


def ask(app, question: str, retries: int = 3) -> dict:
    """Pose une question à l'agent et retourne la réponse avec le temps."""
    start = time.time()
    for attempt in range(retries):
        try:
            result = app.invoke({"messages": [HumanMessage(content=question)]})
            duration = round(time.time() - start, 2)
            answer = result["messages"][-1].content
            # Detect if model returned raw JSON tool call instead of a real answer
            if answer.strip().startswith('{"name"') or answer.strip().startswith("{'name'"):
                raise ValueError("Modèle a retourné un appel d'outil brut, nouvelle tentative...")
            return {"question": question, "answer": answer, "time": duration}
        except Exception as e:
            error_str = str(e)
            # Rate limit → wait longer before retry
            wait = 30 if "429" in error_str else 3
            print(f"  [Retry {attempt + 1}/{retries}] Erreur: {error_str[:80]} (attente {wait}s)")
            time.sleep(wait)
    duration = round(time.time() - start, 2)
    return {"question": question, "answer": f"Erreur après {retries} tentatives.", "time": duration}


def run_evaluation(app):
    """Lance l'évaluation complète sur les 20 questions."""

    resultats_simples = []
    print("=" * 60)
    print("QUESTIONS SIMPLES (10 questions)")
    print("=" * 60)
    for i, q in enumerate(QUESTIONS_SIMPLES, 1):
        print(f"\nQ{i}: {q}")
        try:
            r = ask(app, q)
        except Exception as e:
            r = {"question": q, "answer": f"Erreur: {str(e)[:100]}", "time": 0}
        resultats_simples.append(r)
        print(f"Réponse ({r['time']}s): {r['answer'][:300]}")
        print("-" * 60)
        time.sleep(5)

    resultats_complexes = []
    print("\n" + "=" * 60)
    print("QUESTIONS COMPLEXES (10 questions)")
    print("=" * 60)
    for i, q in enumerate(QUESTIONS_COMPLEXES, 1):
        print(f"\nQ{i}: {q}")
        try:
            r = ask(app, q)
        except Exception as e:
            r = {"question": q, "answer": f"Erreur: {str(e)[:100]}", "time": 0}
        resultats_complexes.append(r)
        print(f"Réponse ({r['time']}s): {r['answer'][:400]}")
        print("-" * 60)
        time.sleep(8)

    # Analyse des performances
    temps_simples = [r['time'] for r in resultats_simples]
    temps_complexes = [r['time'] for r in resultats_complexes]

    print("\n" + "=" * 60)
    print("ANALYSE DES PERFORMANCES")
    print("=" * 60)
    print(f"\nQuestions simples ({len(resultats_simples)} questions):")
    print(f"  Temps moyen   : {statistics.mean(temps_simples):.2f}s")
    print(f"  Temps minimum : {min(temps_simples):.2f}s")
    print(f"  Temps maximum : {max(temps_simples):.2f}s")
    print(f"\nQuestions complexes ({len(resultats_complexes)} questions):")
    print(f"  Temps moyen   : {statistics.mean(temps_complexes):.2f}s")
    print(f"  Temps minimum : {min(temps_complexes):.2f}s")
    print(f"  Temps maximum : {max(temps_complexes):.2f}s")
    print(f"\nTotal : {len(resultats_simples) + len(resultats_complexes)} questions")
    print(f"Temps total : {sum(temps_simples + temps_complexes):.2f}s")

    return resultats_simples, resultats_complexes

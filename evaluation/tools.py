"""
tools.py - Définition des outils disponibles pour l'agent
Chaque outil est une fonction que l'agent peut décider d'appeler.
"""

from langchain_core.tools import tool
from ddgs import DDGS


def create_tools(retriever):
    """Crée et retourne la liste des outils de l'agent."""

    @tool
    def recherche_tourisme(query: str) -> str:
        """Recherche des informations touristiques sur le Maroc dans la base documentaire.
        Utilise cette fonction pour répondre aux questions sur les villes, attractions,
        hébergements, transports et gastronomie du Maroc."""
        docs = retriever.invoke(query)
        if not docs:
            return "Aucune information trouvée pour cette requête."
        result = ""
        for doc in docs:
            source = doc.metadata.get('source', 'inconnu').split('\\')[-1].replace('.txt', '')
            result += f"[Source: {source}]\n{doc.page_content}\n\n"
        return result

    @tool
    def conseils_pratiques(sujet: str) -> str:
        """Fournit des conseils pratiques généraux pour voyager au Maroc.
        Utilise cette fonction pour les questions sur la monnaie, la langue,
        le visa, la sécurité, les habitudes locales et les conseils généraux."""
        conseils = {
            "monnaie": "La monnaie du Maroc est le Dirham marocain (MAD). 1 EUR ≈ 11 MAD. Les distributeurs automatiques sont disponibles dans toutes les grandes villes.",
            "langue": "Les langues officielles sont l'arabe et le tamazight. Le français est très largement parlé dans les villes touristiques.",
            "visa": "Les ressortissants de l'UE, des USA et du Canada n'ont pas besoin de visa pour séjourner au Maroc jusqu'à 90 jours.",
            "sécurité": "Le Maroc est globalement sûr pour les touristes. Restez vigilant dans les médinas bondées contre les pickpockets.",
            "transport": "Les trains ONCF relient les grandes villes. Les bus CTM et Supratours sont fiables pour les trajets inter-villes.",
            "nourriture": "L'eau du robinet n'est pas recommandée. Buvez de l'eau en bouteille. La cuisine marocaine est excellente.",
            "religion": "Le Maroc est un pays musulman. Respectez les lieux de culte et habillez-vous convenablement dans les médinas.",
            "meilleure_periode": "Printemps (mars-mai) et automne (septembre-novembre) sont les meilleures saisons pour visiter.",
        }
        sujet_lower = sujet.lower()
        for key, value in conseils.items():
            if key in sujet_lower:
                return value
        return "\n\n".join([f"{k.upper()}: {v}" for k, v in conseils.items()])

    @tool
    def recherche_web(query: str) -> str:
        """Recherche des informations sur internet via DuckDuckGo.
        Utilise cette fonction quand les informations ne sont pas dans la base documentaire :
        actualités, prix en temps réel, destinations non couvertes (Rabat, Tanger, Sahara...)."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query + " Maroc tourisme", max_results=3))
            if not results:
                return "Aucun résultat trouvé sur internet."
            output = ""
            for r in results:
                output += f"[{r['title']}]\n{r['body']}\n\n"
            return output
        except Exception as e:
            return f"Erreur lors de la recherche web : {str(e)}"

    return [recherche_tourisme, conseils_pratiques, recherche_web]

from __future__ import annotations

import logging
import requests

from .base import Tool, ToolParameter

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "MiniToolAgent/1.0 (educational project; contact: noreply@example.com)"}
MAX_OBSERVATION_LEN = 1500


class WikipediaSearchTool(Tool):
    """Search Wikipedia for factual information."""

    name = "wikipedia_search"
    description = (
        "Search Wikipedia and return a summary snippet. "
        "Use this to look up factual information about people, places, events, etc."
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="The search query string.",
        ),
    ]

    def __init__(self, brave_api_key: str = "", timeout: int = 15):
        self.brave_api_key = brave_api_key
        self.timeout = timeout

    def execute(self, query: str = "", **kwargs) -> str:
        if not query:
            return "Error: empty search query."
        if self.brave_api_key:
            result = self._brave_search(query)
            if result:
                return result
        return self._wiki_search(query)

    # ── Wikipedia API ──────────────────────────────────────────────

    def _wiki_search(self, query: str) -> str:
        try:
            titles = self._wiki_opensearch(query)
            if not titles:
                titles = self._wiki_fulltext_search(query)
            if not titles:
                return f"No Wikipedia results found for: {query}"

            for title in titles[:3]:
                extract = self._wiki_extract(title)
                if extract:
                    return self._truncate(f"[Wikipedia: {title}]\n{extract}")
            return f"No Wikipedia content found for: {query}"
        except Exception as e:
            logger.warning("Wikipedia search failed: %s", e)
            return f"Search error: {e}"

    def _wiki_opensearch(self, query: str) -> list[str]:
        resp = requests.get(
            WIKI_API,
            params={"action": "opensearch", "search": query, "limit": 5, "format": "json"},
            headers=WIKI_HEADERS,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data[1] if len(data) > 1 else []

    def _wiki_fulltext_search(self, query: str) -> list[str]:
        resp = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 5,
                "format": "json",
            },
            headers=WIKI_HEADERS,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        return [r["title"] for r in results]

    def _wiki_extract(self, title: str) -> str:
        resp = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "format": "json",
            },
            headers=WIKI_HEADERS,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                return extract
        return ""

    # ── Brave Search API (fallback) ────────────────────────────────

    def _brave_search(self, query: str) -> str:
        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": self.brave_api_key, "Accept": "application/json"},
                params={"q": query, "count": 3},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("web", {}).get("results", [])
            if not results:
                return ""
            snippets = []
            for r in results[:3]:
                snippets.append(f"[{r.get('title', '')}]\n{r.get('description', '')}")
            return self._truncate("\n\n".join(snippets))
        except Exception as e:
            logger.warning("Brave search failed, falling back to Wikipedia: %s", e)
            return ""

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) > MAX_OBSERVATION_LEN:
            return text[:MAX_OBSERVATION_LEN] + "\n... [truncated]"
        return text

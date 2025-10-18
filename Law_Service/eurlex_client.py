"""
EUR-Lex SOAP Client
Using pure HTTP
"""

import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import requests
from lxml import etree


class EURLexClient:
    """
    EUR-Lex SOAP web service client using pure HTTP.
    Provides methods to search for documents in the EUR-Lex database.
    """

    ENDPOINT = "https://eur-lex.europa.eu/EURLexWebService"

    VALID_LANGUAGES = {
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "ga",
        "hr",
        "hu",
        "it",
        "lt",
        "lv",
        "mt",
        "nl",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "sv",
    }

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """Initialize the EUR-Lex client."""
        load_dotenv()

        self.username = username or os.getenv("EURLEX_USERNAME")
        self.password = password or os.getenv("EURLEX_PASSWORD")

        if not self.username or not self.password:
            raise ValueError(
                "EUR-Lex credentials required via parameters or environment variables"
            )

    def search_documents(
        self,
        expert_query: str,
        page: int = 1,
        page_size: int = 10,
        search_language: str = "en",
        legislation: bool = True,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Search EUR-Lex documents.

        Args:
            expert_query: Expert query syntax (e.g., "QUICK_SEARCH ~ transport")
            page: Page number (default: 1)
            page_size: Results per page (default: 10, max: 1000)
            search_language: Language code (default: "en")
            exclude_all_consleg: Exclude consolidated legislation
            limit_to_latest_consleg: Limit to latest consolidated legislation
            timeout: Request timeout in seconds (default: 30)

        Returns:
            Dictionary with search results
        """
        # Validate
        if not expert_query or not expert_query.strip():
            raise ValueError("expert_query is required")
        if page < 1:
            raise ValueError("Page must be >= 1")
        if page_size < 1 or page_size > 1000:
            raise ValueError("Page size must be 1-1000")
        if search_language not in self.VALID_LANGUAGES:
            raise ValueError(f"Invalid language: {search_language}")

        # legislation filter
        legislation_str = " AND DTS_SUBDOM = LEGISLATION" if legislation else ""
        full_query = expert_query + legislation_str

        
        # Build SOAP envelope
        soap_body = f"""
        <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:sear="http://eur-lex.europa.eu/search">
  <soap:Header>
    <wsse:Security xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" soap:mustUnderstand="true">
      <wsse:UsernameToken xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd" wsu:Id="UsernameToken-1">
        <wsse:Username>{self.username}</wsse:Username>
        <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordText">{self.password}</wsse:Password>
      </wsse:UsernameToken>
    </wsse:Security>
  </soap:Header>
  <soap:Body>
    <sear:searchRequest>
      <sear:expertQuery><![CDATA[{full_query}]]></sear:expertQuery>
      <sear:page>{page}</sear:page>
      <sear:pageSize>{page_size}</sear:pageSize>
      <sear:searchLanguage>{search_language}</sear:searchLanguage>
    </sear:searchRequest>
  </soap:Body>
</soap:Envelope>
"""

        headers = {
            "Content-Type": "application/soap+xml; charset=utf-8",
        }

        try:
            # Make HTTP POST request
            response = requests.post(
                self.ENDPOINT,
                data=soap_body.encode("utf-8"),
                headers=headers,
                timeout=timeout,
            )

            # Check for HTTP errors
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    "numhits": 0,
                    "totalhits": 0,
                    "page": page,
                    "language": search_language,
                    "results": [],
                }

            self._save_xml_response(response.content, page, expert_query)

            # Parse XML response
            result = self._parse_xml_response(response.content)
            result["success"] = True
            result["error"] = None
            result["page"] = page
            result["language"] = search_language

            return result

        except requests.Timeout:
            return {
                "success": False,
                "error": f"Request timeout ({timeout}s)",
                "numhits": 0,
                "totalhits": 0,
                "page": page,
                "language": search_language,
                "results": [],
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error: {type(e).__name__}: {str(e)}",
                "numhits": 0,
                "totalhits": 0,
                "page": page,
                "language": search_language,
                "results": [],
            }

    # option to save xml responses for debugging
    def _save_xml_response(self, xml_content: bytes, page: int, query: str) -> None:
        """Save XML response to file for debugging."""
        try:
            # Create responses directory if it doesn't exist
            os.makedirs("responses", exist_ok=True)

            # Create a safe filename from query (remove special chars)
            safe_query = "".join(c if c.isalnum() else "_" for c in query)[:50]
            filename = f"responses/eurlex_response_page{page}_{safe_query}.xml"

            # Write XML content to file
            with open(filename, "wb") as f:
                f.write(xml_content)

            print(f"✓ XML response saved to: {filename}")

        except Exception as e:
            # Don't fail the main request if saving fails
            print(f"Warning: Could not save XML response: {str(e)}")

    def _parse_xml_response(self, xml_content: bytes) -> Dict[str, Any]:
        """Parse the SOAP XML response."""
        result = {"numhits": 0, "totalhits": 0, "results": []}

        try:
            # Parse XML
            root = etree.fromstring(xml_content)

            # Define namespaces
            ns = {
                "S": "http://www.w3.org/2003/05/soap-envelope",
                "sear": "http://eur-lex.europa.eu/search",
            }

            # Find searchResults element
            search_results = root.find(".//sear:searchResults", ns)

            if search_results is None:
                return result

            # Get totalhits and numhits
            totalhits_elem = search_results.find("sear:totalhits", ns)
            if totalhits_elem is not None and totalhits_elem.text:
                result["totalhits"] = int(totalhits_elem.text)

            numhits_elem = search_results.find("sear:numhits", ns)
            if numhits_elem is not None and numhits_elem.text:
                result["numhits"] = int(numhits_elem.text)

            # Parse result elements
            for result_elem in search_results.findall("sear:result", ns):
                doc_result = {}

                # Get reference
                ref_elem = result_elem.find("sear:reference", ns)
                if ref_elem is not None:
                    doc_result["reference"] = ref_elem.text

                # Get rank
                rank_elem = result_elem.find("sear:rank", ns)
                if rank_elem is not None and rank_elem.text:
                    doc_result["rank"] = int(rank_elem.text)

                # Get document links
                doc_links = []
                for link_elem in result_elem.findall("sear:document_link", ns):
                    link_type = link_elem.get("type")
                    link_url = link_elem.text
                    if link_url:
                        doc_links.append({"type": link_type, "url": link_url})
                if doc_links:
                    doc_result["document_links"] = doc_links

                # Get content (lang, and title)
                content_elem = result_elem.find("sear:content", ns)
                if content_elem is not None:
                    content = {}

                    # Navigate through NOTICE -> EXPRESSION -> EXPRESSION_TITLE
                    # ALL elements need the sear: prefix because they're in the default namespace
                    notice_elem = content_elem.find("sear:NOTICE", ns)
                    if notice_elem is not None:
                        expression_elem = notice_elem.find("sear:EXPRESSION", ns)
                        if expression_elem is not None:
                            expression_title = expression_elem.find(
                                "sear:EXPRESSION_TITLE", ns
                            )
                            if expression_title is not None:
                                # Extract language
                                lang_elem = expression_title.find("sear:LANG", ns)
                                if lang_elem is not None and lang_elem.text:
                                    content["language"] = lang_elem.text

                                # Extract title
                                value_elem = expression_title.find("sear:VALUE", ns)
                                if value_elem is not None and value_elem.text:
                                    content["title"] = value_elem.text

                if content:
                    doc_result["content"] = content

                result["results"].append(doc_result)

        except Exception as e:
            # If parsing fails, return what we have
            result["parse_error"] = str(e)

        return result

    def get_available_languages(self) -> List[str]:
        """Get available languages."""
        return sorted(self.VALID_LANGUAGES)


if __name__ == "__main__":
    print("EUR-Lex Client - (Pure HTTP)")
    print("=" * 60)

    try:
        client = EURLexClient()
        print("✓ Client initialized\n")

        print("Testing search...")
        results = client.search_documents(
            expert_query="QUICK_SEARCH ~ transport",
            page=1,
            page_size=5,
            search_language="en",
        )

        if results["success"]:
            print(f"\n✓✓✓ SUCCESS! ✓✓✓")
            print(f"Total: {results['totalhits']}")
            print(f"Page results: {results['numhits']}")

            if results["results"]:
                print(f"\nFirst 3 results:")
                for i, doc in enumerate(results["results"][:3], 1):
                    print(f"\n{i}. {doc.get('reference', 'N/A')}")
                    print(f"   Rank: {doc.get('rank', 'N/A')}")

                    if "content" in doc:
                        if "language" in doc["content"]:
                            print(f"   Language: {doc['content']['language']}")
                        if "title" in doc["content"]:
                            print(f"   Title: {doc['content']['title'][:100]}...")

                    if "document_links" in doc:
                        for link in doc["document_links"][:2]:
                            print(f"   {link['type']}: {link['url']}")

        else:
            print(f"\n✗ Failed: {results['error']}")

    except Exception as e:
        print(f"✗ Error: {str(e)}")

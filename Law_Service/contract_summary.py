from openai import OpenAI
from pydantic import BaseModel
import json
from typing import List
import logging


logger = logging.getLogger(__name__)


class LegalAnalysis(BaseModel):
    summary: str
    suggested_questions: List[str]
    # keywords: List[str]
    keywords: List[List[str]]


client = OpenAI(base_url="https://router.huggingface.co/v1", api_key="")


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_chunk_size(total_pages):
    return max(1, total_pages // 5)


def create_chunks(pages_dict, chunk_size):
    sorted_pages = sorted(pages_dict.items(), key=lambda x: int(x[0]))
    chunks = []
    for i in range(0, len(sorted_pages), chunk_size):
        chunks.append(sorted_pages[i : i + chunk_size])
    return chunks


def analyze_chunk(chunk, prev_analysis=None):
    chunk_text = "\n\n".join([page["text"] for _, page in chunk])
    page_nums = [page["page_number"] for _, page in chunk]
    page_range = f"{page_nums[0]}-{page_nums[-1]}"

    if prev_analysis is None:
        prompt = """write a concise summary, extract the three most relevant legal/law keywords as a list of list, and suggest three questions that can be answered from the content to be fed for a RAG system. The keywords should be relevant to legal contexts. the keywords should be returned as a list of lists, where each list will represent a page "Example:"
            '{'
            '  "summary": "Concise summary here.",\n'
            '  "suggested_questions": ["Q1", "Q2", "Q3"],\n'
            '  "keywords": [["k1", "k2", "k3"]]\n'
            '}' """
        system = f"Analyze this legal document (pages {page_range}):\n\n{chunk_text}"

    else:
        prompt = f"""Previous Summary: {prev_analysis.summary}
Previous Keywords: {prev_analysis.keywords}
Previous suggested Questions: {", ".join(prev_analysis.suggested_questions)}
New text (pages {page_range}):
{chunk_text}
Your task is to rewrite the previous summary with the updated information or leave it as it is if no new relevant information was given, add a new list of the three most relevant legal/law keywords for this page and append it as a list to the list of lists of keywords, update or leave the three suggested questions, make sure that the questions could be answered from the content."""
        system = """Update the summary and questions or keep previous if no significant changes and append the new list to the list of keywords.  
The questions should have an answer from the content and not be broad law info. 
The keywords are going to be used as a query for document retrieval, so make sure they are relevant and comprehensive. 
keywords should be max 3 words per list, that are relevant to legal/law contexts. as it will be used to search for related legal documents. the keywords should be returned as a list of lists, where each list will represent a page.
"Example:"
    '{'
    '  "summary": "new updated summary here.",'
    '  "suggested_questions": ["Q1", "Q2", "Q3"],'
    '  "keywords": [["contract breach", "statute of limitations", "damages"], ' # page one
    '["negligence", "duty of care", "proximate cause"]] # page two'
    '}'
"""

    response = client.responses.parse(
        model="google/gemma-2-9b-it",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        text_format=LegalAnalysis,
    )
    return response.output_parsed


def save_result(analysis, output_path):
    result = {
        "summary": analysis.summary,
        "keywords": analysis.keywords,
        "suggested_questions": analysis.suggested_questions,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def analyze_legal_document(data_dict, output_file=None):
    """
    Orchestrates the legal document analysis process.

    Args:
        input_file (str): Path to the input JSON file containing document pages
        output_file (str, optional): Path to save the analysis results. If None, results are returned but not saved.
        verbose (bool): If True, prints progress information

    Returns:
        dict: Analysis results containing summary, keywords, and suggested_questions
    """
    pages = data_dict.get("result", {}).get("pages", {})
    total_pages = len(pages)

    # Determine chunk size
    chunk_size = get_chunk_size(total_pages)
    # Create chunks
    chunks = create_chunks(pages, chunk_size)

    current_analysis = None

    for i, chunk in enumerate(chunks, 1):
        page_nums = [page["page_number"] for _, page in chunk]

        print(
            f"Processing chunk {i}/{len(chunks)} (pages {page_nums[0]}-{page_nums[-1]})..."
        )

        current_analysis = analyze_chunk(chunk, current_analysis)

        print(f"Summary: {current_analysis.summary[:80]}...")
        print(
            f"Keywords: {'; '.join([', '.join(page_kws) for page_kws in current_analysis.keywords])}\n"
        )
        print(
            f"Suggested Questions: {', '.join(current_analysis.suggested_questions)}\n"
        )

    result = {
        "summary": current_analysis.summary,
        "keywords": current_analysis.keywords,
        "suggested_questions": current_analysis.suggested_questions,
    }

    if output_file:
        save_result(current_analysis, output_file)
        print(f"Saved to: {output_file}")

    print("\n=== Final Analysis ===")
    print(f"Summary: {result['summary']}")
    print(f"Keywords: {result['keywords']}")

    return result


if __name__ == "__main__":
    # For testing, load from file
    data = load_json("temp_input.json")
    analyze_legal_document(data, "legal_analysis_output.json")

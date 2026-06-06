import os, json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from langchain_core.embeddings import Embeddings
import fitz
from docx import Document
import time
import pytesseract
from PIL import Image
import io

# Help function
def similarity_search_filtered(store, query, k=3, doc_type=None):
    if not query or not query.strip():
        return []
    raw_results = store.similarity_search(query, k=500)
    if doc_type is None:
        return raw_results[:k]
    filtered = [d for d in raw_results if d.metadata.get('type') == doc_type]
    if not filtered:
        print(f"    Warning: no '{doc_type}' documents found for this query")
    return filtered[:k]

# ── General retry function ──────────────────────────────────────────────
def invoke_with_retry(llm, prompt, max_retries=3, delay=5):
    """调用LLM并自动重试，确保返回合法JSON"""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            if not raw:
                raise ValueError("Empty response from model")
            return json.loads(raw)
        except Exception as e:  # Catch all exceptions to maximize robustness
            print(f"    DEBUG attempt {attempt+1} error type: {type(e).__name__}")
            print(f"    DEBUG error message: {e}")
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}/{max_retries-1}: retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"    All retries failed, using fallback.")
                return None
    return None

load_dotenv()

CONTRACT_TYPES = [
    "Public Research Contracts",
    "Commercial Research Contracts",
    "Subcontracts",
    "Material Transfer Agreements",
    "Data Transfer Agreements",
    "Collaboration Agreements",
    "Confidential Disclosure Agreements",
]

# Contract type → corresponding template filename mapping
CONTRACT_TYPE_TEMPLATES = {
    "Public Research Contracts":        ["UoA-Research Collaboration Agreement Template (1).docx",
                                         "UoA-Research Services Agreement (Agency) _June 2024 .docx"],
    "Commercial Research Contracts":    ["UoA-Research Services Agreement (Agency) _June 2024 .docx",
                                         "UoA-Master Services Agreement Template (1).docx"],
    "Subcontracts":                     ["UoA-Template Subcontractor Agreement_2025 (1) (1).docx"],
    "Material Transfer Agreements":     ["UoA-Material_Transfer_Agreement incoming-Aug 2024.docx",
                                         "UoA-Material_Transfer_Agreement_outgoing_Aug 2024.docx",
                                         "UoA-MTA_Outbound for Key Materials-April 2018.docx"],
    "Data Transfer Agreements":         ["UoA-Data Transfer Agreement Template (incoming) April 2024 .docx",
                                         "UoA-Data Transfer Agreement Template (outgoing) April 2024.docx"],
    "Collaboration Agreements":         ["UoA-Research Collaboration Agreement Template (1).docx"],
    "Confidential Disclosure Agreements": ["UoA-CDA Two Way Template.docx"],
}

STATUS_COLORS = {
    "GREEN": "✅ Compliant",
    "RED":   "🔴 Violation — Manager alert required",
    "BLUE":  "🔵 No matching clause — Manager alert required",
    "AMBER": "🟡 Uncertain — Historical check + Manager alert required",
}

# ── Embedding Class （keep consistent with setup_rag.py）─────────────────
class DirectAzureEmbeddings(Embeddings):
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint="https://ai-team-04-hack.openai.azure.com/",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
        )
        self.model = "text-embedding-3-small"

    def embed_documents(self, texts):
        result = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in result.data]

    def embed_query(self, text):
        result = self.client.embeddings.create(input=[text], model=self.model)
        return result.data[0].embedding

# ── Tool functions ──────────────────────────────────────────────────
def get_llm():
    return AzureChatOpenAI(
        azure_endpoint="https://ai-team-04-hack.openai.azure.com/",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version="2024-02-01",
    )

def extract_text(filepath):
    if filepath.endswith(".pdf"):
        doc = fitz.open(filepath)
        full_text = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if len(text) < 50:  # too few characters, indicating an image page, use OCR
                print(f"  Page {i+1}: using OCR...")
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
            full_text.append(text)
        return "\n".join(full_text)
    elif filepath.endswith(".docx"):
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return ""

def load_vector_store():
    embeddings = DirectAzureEmbeddings()
    return FAISS.load_local(
        "vector_store", embeddings,
        allow_dangerous_deserialization=True
    )

# ── Step 1: Identify contract type ──────────────────────────────────────
def identify_contract_type(text, llm):
    types_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(CONTRACT_TYPES))
    prompt = f"""
You are a contract classification expert at the University of Auckland.
Identify the type of this contract from the list below.

CONTRACT TYPES:
{types_list}

CONTRACT (first 3000 chars):
{text[:3000]}

IMPORTANT: Return ONLY valid JSON, no markdown, no backticks, no explanation.
{{
  "identified_type": "exact name from the list above",
  "confidence": "high|medium|low",
  "reasoning": "brief explanation"
}}
"""
    result = invoke_with_retry(llm, prompt)
    if result is None:
        # fallback：Let the user manually select
        return {"identified_type": "Unknown", "confidence": "low", "reasoning": "Could not identify automatically."}
    return result

# ── Step 2: User confirmation process ──────────────────────────────────────
def confirm_contract_type(identified):
    print("\n" + "="*60)
    print("CONTRACT TYPE IDENTIFICATION")
    print("="*60)
    print(f"Identified type : {identified['identified_type']}")
    print(f"Confidence      : {identified['confidence']}")
    print(f"Reasoning       : {identified['reasoning']}")
    print("="*60)

    confirm = input("\nIs this correct? (y/n): ").strip().lower()

    if confirm == "y":
        return identified["identified_type"]

    # Fail to identify, let the user manually select
    print("\nPlease select the correct contract type:")
    for i, t in enumerate(CONTRACT_TYPES):
        print(f"  {i+1}. {t}")

    while True:
        choice = input("\nEnter number (1-7): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(CONTRACT_TYPES):
            return CONTRACT_TYPES[int(choice) - 1]
        print("Invalid input, please enter a number between 1 and 7.")

# ── Step 3: Confirm whether to start the review ─────────────────────────────────
def confirm_review_start(contract_type):
    print("\n" + "="*60)
    print(f"Contract type confirmed: {contract_type}")
    print("="*60)
    confirm = input("\nProceed with compliance review? (y/n): ").strip().lower()
    return confirm == "y"

# ── Step 4: Extract contract clauses ──────────────────────────────────────
def extract_clauses(text, llm):
    """Segmented extraction terms to ensure that no content is omitted due to token limits."""
    
    # gpt-4o-mini security limit is approximately 80000 characters (leaving room for system prompt and output)
    MAX_CHARS = 80000
    
    if len(text) <= MAX_CHARS:
        # The document is short, process it in one go
        return _extract_clauses_single(text, llm)
    else:
        # The document is long, process it in segments
        print(f"  Document is long ({len(text)} chars), processing in segments...")
        return _extract_clauses_chunked(text, llm, MAX_CHARS)

def _extract_clauses_single(text, llm):
    prompt = f"""
You are a contract analyst. Extract ALL numbered or distinct clauses from this contract.
Do not skip any clause, including schedules, annexures, and appendices.

IMPORTANT: Return ONLY a valid JSON array. No markdown, no backticks, no explanation.
Format: [{{"clause_number": "1", "text": "..."}}]

CONTRACT:
{text}
"""
    result = invoke_with_retry(llm, prompt)
    if result is None:
        return [{"clause_number": "1", "text": text[:3000]}]
    return result

def _extract_clauses_chunked(text, llm, max_chars):
    paragraphs = text.split("\n\n")
    segments = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars:
            if current:
                segments.append(current)
            current = para
        else:
            current += "\n\n" + para
    if current:
        segments.append(current)

    print(f"  Split into {len(segments)} segments.")
    all_clauses = []

    for i, segment in enumerate(segments):
        print(f"  Extracting clauses from segment {i+1}/{len(segments)}...")
        prompt = f"""
You are a contract analyst. Extract ALL clauses from this contract segment (part {i+1} of {len(segments)}).

IMPORTANT: Return ONLY a valid JSON array. No markdown, no backticks, no explanation.
Format: [{{"clause_number": "1", "text": "..."}}]
If no distinct clauses found, return []

SEGMENT:
{segment}
"""
        result = invoke_with_retry(llm, prompt)
        if result:
            all_clauses.extend(result)

    for i, clause in enumerate(all_clauses):
        clause["clause_number"] = str(i + 1)
    return all_clauses

def _parse_clause_json(raw_content, fallback_text):
    """General function for JSON parsing and error handling"""
    raw = raw_content.strip()
    
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    if not raw or raw == "[]":
        return [{"clause_number": "1", "text": fallback_text[:3000]}]

    try:
        result = json.loads(raw)
        return result if result else [{"clause_number": "1", "text": fallback_text[:3000]}]
    except json.JSONDecodeError:
        return [{"clause_number": "1", "text": fallback_text[:3000]}]

# ── Step 5: Review clauses individually ──────────────────────────────────────────
def review_clause(clause, contract_type, store, llm):
    # 检索 position 文档
    position_docs = similarity_search_filtered(
        store, clause["text"], k=3, doc_type="position"
    )
    position_context = "\n---\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in position_docs
    )

    # 只检索与该合同类型匹配的模板
    matched_templates = CONTRACT_TYPE_TEMPLATES.get(contract_type, [])
    if matched_templates:
        raw_template_docs = similarity_search_filtered(
            store, clause["text"], k=20, doc_type="template"
        )
        # 只保留匹配的模板文件
        template_docs = [
            d for d in raw_template_docs
            if d.metadata.get('source') in matched_templates
        ][:3]
        template_context = "\n---\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in template_docs
        ) if template_docs else "No relevant template clause found."
    else:
        template_context = "No template mapped for this contract type."

    # 其余 prompt 和逻辑不变
    ...
    prompt = f"""
You are a contract compliance reviewer at the University of Auckland.
Contract type: {contract_type}

POSITION DOCUMENT RULES (primary authority):
{position_context}

UOA CONTRACT TEMPLATE (secondary reference — use to supplement position rules where position is general):
{template_context}

CONTRACT CLAUSE #{clause['clause_number']}:
{clause['text']}

IMPORTANT: Return ONLY valid JSON, no markdown, no backticks, no explanation.
{{
  "status": "GREEN|RED|BLUE|AMBER",
  "reasoning": "detailed explanation referencing both position rules and template where relevant",
  "cited_position_clause": {{
    "document": "exact filename from [Source: ...] tag",
    "section": "section number or heading if identifiable, or 'N/A'",
    "quote": "exact quoted text from POSITION DOCUMENT RULES or TEMPLATE above"
  }}
}}

Rules:
- GREEN: compliant with position rules AND/OR consistent with UoA template
- RED: directly violates a position rule
- BLUE: no relevant rule in position doc AND no relevant template clause — set all cited fields to 'None found'
- AMBER: uncertain after checking both position doc and template

CRITICAL:
- Check BOTH position rules and template before deciding status
- If position rule is general but template has a specific matching clause, use template to determine GREEN/RED
- cited_position_clause.quote must come from POSITION DOCUMENT RULES or TEMPLATE sections above
- Quote must NOT be copied from CONTRACT CLAUSE text
- Document field must match a [Source: ...] tag from above
"""

    result = invoke_with_retry(llm, prompt)
    if result is None:
        result = {
            "status": "AMBER",
            "reasoning": "Could not parse model response after retries, manual review required.",
            "cited_position_clause": {
                "document": "N/A",
                "section": "N/A",
                "quote": "None found"
            },
        }

    # AMBER: 查询历史合同
    if result["status"] == "AMBER":
        history_docs = similarity_search_filtered(
            store, clause["text"], k=3, doc_type="historical"
        )
        history_context = "\n---\n".join(
            d.page_content for d in history_docs
        )
        hist_prompt = f"""
This contract clause is in the AMBER zone (uncertain compliance).
Search these historical records for similar clauses and outcomes.

HISTORICAL RECORDS:
{history_context}

CLAUSE:
{clause['text']}

IMPORTANT: Return ONLY valid JSON, no markdown, no backticks, no explanation.
{{"historical_precedent": "description of similar past case and outcome, or 'No similar case found'"}}
"""
        hist_result = invoke_with_retry(llm, hist_prompt)
        if hist_result:
            result["historical_precedent"] = hist_result.get(
                "historical_precedent", "No similar case found"
            )
        else:
            result["historical_precedent"] = "Could not retrieve historical precedent."
    else:
        result["historical_precedent"] = None

    result["clause_number"] = clause["clause_number"]
    result["clause_text"] = clause["text"]
    return result

# ── The Main Review Process ────────────────────────────────────────────────────
def review_contract(filepath):
    print(f"\nLoading contract: {filepath}")
    text = extract_text(filepath)
    if not text.strip():
        print("ERROR: Could not extract text from file.")
        return None

    llm = get_llm()
    store = load_vector_store()

    # Step 1: Identify contract type
    print("Identifying contract type...")
    identified = identify_contract_type(text, llm)

    # Step 2: User confirmation
    contract_type = confirm_contract_type(identified)

    # Step 3: Confirm whether to start the review
    if not confirm_review_start(contract_type):
        print("Review cancelled.")
        return None

    # Step 4 & 5: Extract clauses and review them individually
    print("\nExtracting clauses...")
    clauses = extract_clauses(text, llm)
    print(f"Found {len(clauses)} clauses. Starting review...\n")

    results = []
    for clause in clauses:
        if not clause.get("text", "").strip():
            print(f"  Skipping clause {clause.get('clause_number', '?')} (empty text)")
            continue
        print(f"  Reviewing clause {clause['clause_number']}...", end=" ")
        result = review_clause(clause, contract_type, store, llm)
        results.append(result)
        print(result["status"])

    # Output review summary
    from collections import Counter
    counts = Counter(r["status"] for r in results)
    print("\n" + "="*60)
    print("REVIEW COMPLETE")
    print("="*60)
    for status, count in counts.items():
        print(f"  {STATUS_COLORS[status]}: {count} clause(s)")

    return {"contract_type": contract_type, "results": results}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python 2_review_contract.py <contract_file>")
    else:
        review_contract(sys.argv[1])
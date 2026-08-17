# Manufacturing Plant SOP & Safety Explainer Bot

A Streamlit RAG chatbot that explains approved manufacturing SOPs and safety procedures in simple language using Gemini Flash through Google AI Studio.

The bot is intentionally explanation-only. It does not approve actions, certify compliance, authorize operations, or replace supervisors.

## Features

- Retrieval-Augmented Generation over approved internal documents
- Semantic search using Gemini embeddings
- Gemini Flash response generation with a strict safety/system prompt
- Streamlit web interface
- Source-grounded answers with retrieved context visibility
- Support for `.txt`, `.md`, `.pdf`, and `.docx` approved documents

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- data/
|   `-- approved_documents/
|       |-- lockout_tagout_sop.md
|       |-- machine_area_ppe.md
|       `-- emergency_shutdown.md
`-- .rag_index/
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your Google AI Studio API key:

```bash
set GEMINI_API_KEY=your_api_key_here
```

PowerShell:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

You can also enter the key in the Streamlit sidebar.

Use `gemini-2.5-flash` in the sidebar's Gemini Flash model field. The free tier has a low daily/request quota, so creating another key in the same Google AI Studio project does not reset the quota.

3. Add approved SOPs, safety manuals, or internal documents to:

```text
data/approved_documents
```

4. Run the app:

```bash
streamlit run app.py
```

If `streamlit` is not recognized on Windows, run the project launcher instead:

```powershell
.\run_app.ps1
```

Or run Streamlit through the local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Tested Bot Queries

- Explain lockout-tagout in simple terms
- What safety gear is used near heavy machines?
- Summarize emergency shutdown procedure
- Explain this SOP step-by-step

## Safety Boundary

The assistant only explains approved documents. It must refuse or redirect requests that ask it to:

- Approve work
- Authorize machine operation
- Decide whether something is safe to do
- Certify compliance
- Replace supervisor or safety officer judgment

For real plant work, employees must follow official SOPs and contact responsible supervisors or safety officers.

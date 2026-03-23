from langchain_core.prompts import ChatPromptTemplate

GENERATE_PLAN_PROMPT = ChatPromptTemplate.from_template("""
You are an expert **Report Planning Architect** specializing in data analysis and technical documentation.
Your task is to generate a precise, structured plan consisting of exactly 5 sections for an analytical report about web scraping or data extraction activities.

### Input Context
You will receive the following information to inform your planning:

**Site Information:**
- Target URL: {url}
- Site Name: {site_name}

**Technical Data:**
- Raw Observations: {observations}
  - Core operational data containing type, stage, and message fields that document the extraction process
- Important Routes: {url_dict}
  - Dictionary of prioritized routes analyzed during data collection and it's selectors
- Selectors Dictionary: {selectors}
  - Technical selectors (CSS) used for content extraction inside articles (to extract articles content)

**Optional Refinements:** {improvements}
  - Additional instructions or adjustments to incorporate into the plan

### Section Structure Requirements
Each section object must contain **exactly 3 fields** (no more, no less):

1. **`title`** (string)
   - Concise and descriptive (4-8 words recommended)
   - Use clear, professional language
   - Example: "Data Collection Routes and Scope"
2. **`expected_format`** (list)
   - Must be a list containing one or more of: `["narrative", "bullets"]`
   - Choose based on content type:
     * `narrative`: For analytical explanations, methodologies, summaries
     * `bullets`: For lists, findings, discrete items, recommendations
   - Can include both if section benefits from mixed format
3. **`description`** (string)
   - 2-4 sentences explaining:
     * The section's purpose and analytical goal
     * Which specific data sources to use (observations, url_dict, selectors)
     * What insights or deliverables the section must provide
     * Any specific filtering or grouping criteria (e.g., by type, stage, status)
   - Be specific about what to include and how to analyze it

### Required Section Framework
Generate exactly **5 sections** following this thematic structure:

**Section 1: Data Collection Scope**
- Focus: Routes discovered and analyzed (from url_dict)
- Must document: Which URLs/routes were targeted and their significance
- Include: Selectors used for discovery

**Section 2: Extraction Methodology and Findings**
- Focus: Technical approach and data extraction results
- Must document: Selectors used (from selectors dict) and for articles content (selectors), extraction techniques
- Include: Success metrics and data quality observations from observations

**Section 3: Rate Limiting and Performance Analysis**
- Focus: System constraints and response behavior
- Must analyze: Rate limit encounters, throttling patterns, response times
- Source: Filter observations by relevant types (errors, warnings, rate_limit indicators)

**Section 4: Site Configuration Testing Results**
- Focus: Technical tests performed on site infrastructure
- Must analyze: Configuration checks, compatibility tests, edge cases
- Source: Observations related to testing stages and configuration validation

**Section 5: Executive Summary and Recommendations**
- Focus: High-level synthesis and actionable improvements
- Must include: Key findings recap, identified issues, optimization recommendations
- Format: Mix of narrative context with bulleted recommendations

### Critical Requirements
- Generate exactly 5 sections (no more, no fewer)
- The url_dict must be referenced in at least one section description
- The selectors dict must be referenced in at least one section description
""")


EVALUATE_PLAN_PROMPT = ChatPromptTemplate.from_template("""
You are a highly analytical **Report Planning Quality Assurance Expert**. Your task is to rigorously assess the quality and utility of the provided report plan (a list of sections).

### Evaluation Criteria (Weighted & Detailed)

Evaluate the plan against these criteria, mentally scoring each from 0.0 to 1.0:

1.  **Relevance & Alignment (40% Weight):**
    * Do the sections directly address the likely objective of a technical report based on observations, URLs, and selectors (e.g., findings, methodology, scope, summary)?
    * Are the sections focused and not overly generic?

2.  **Completeness & Flow (30% Weight):**
    * Does the plan cover all necessary components for a coherent report (e.g., Introduction/Summary, Detailed Findings, Context/Methodology, Conclusion)?
    * Is there a logical and non-redundant flow between sections?

3.  **Clarity & Actionability (30% Weight):**
    * Are the `title` and `description` of each section **concise, specific, and unambiguous**?
    * Is the `expected_format` (narrative/bullets) appropriate for the section's stated goal? (e.g., Summary should be narrative, a list of errors should be bullets).

### Instructions for Output

1.  **Overall Score:** Calculate a final **weighted average score** between **0.0 (unusable)** and **1.0 (perfect)** based on the weights above.
2.  **Improvements:** Provide a list of **specific, actionable suggestions** under 'improvements'. Focus on *what* to change and *why*.
    * If a section is weak, reference it by its **title** and explain how to refine its description or format.
    * If sections are missing, suggest the **title, format, and description** for the required missing sections (e.g., a "Methodology" section).
3.  **Optimal Plan:** If the plan scores **0.85 or higher**, return an empty list for 'improvements'.

### Report Plan to Evaluate

Planned sections (JSON list):
{sections}
""")
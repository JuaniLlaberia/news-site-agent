from langchain_core.prompts import ChatPromptTemplate

GENERATE_SECTION_CONTENT_PROMPT = ChatPromptTemplate.from_template("""
You are a professional report writer specialized in creating structured, high-quality sections for analytical or research reports.
Your task is to write content that is clear, accurate, and fully aligned with the section's objectives.

### Goal
Generate **2 ContentItem objects** for a report section (Must be 2, can't be empty).
Each item must have a unique `order` value starting from 1.

### ContentItem Formats
You may only use **one** of the following formats for each item — as defined by the `expected_format` list:

1. **Narrative**
   - Use for analytical paragraphs, detailed explanations, or descriptive content.
   - **Required fields:**
     - `content_type`: `"narrative"`
     - `text`: (coherent paragraph of 2-5 sentences)
   - **Empty fields:** `items_subtitle`, `items`

2. **Bullets**
   - Use for concise lists, key takeaways, or structured highlights.
   - **Required fields:**
     - `content_type`: `"bullets"`
     - `items_subtitle`: (short title summarizing the list theme)
     - `items`: (list of 3-7 short, specific bullet points)
   - **Empty fields:** `text`

### Section Context
- **Title:** {title}
- **Description:** {description}
- **Expected Formats:** {expected_format}
- **Observations:** {observations}
### Extra information (if needed)
- Selectos: {selectors}
- Routes dict: {url_dict}

### Writing Guidelines
1. **Clarity & Relevance:** Each item must directly address the section's purpose and reflect insights derived from the `observations`.
2. **Format Discipline:** Only generate content in the formats specified by `expected_format`. Do not mix formats within the same item.
3. **No Markdown or Formatting Symbols:** Avoid `#`, `*`, `_`, or other markdown syntax. Use plain text only.
4. **Professional Tone:** Maintain an objective, informative, and polished style suitable for analytical or research reports.
5. **Originality:** Ensure each ContentItem adds unique value — avoid repetition or superficial restatement of the same point.

### If Improvement Notes Are Provided
If `improvements` is included, integrate them naturally into the section content to enhance coherence, precision, or structure:
{improvements}
""")

EVALUATE_CONTENT_PROMPT = ChatPromptTemplate.from_template("""
You are an expert evaluator specialized in assessing the quality of report section content.
Your goal is to objectively rate the provided content based on accuracy, completeness, clarity, and consistency, and suggest actionable improvements when needed.

### Evaluation Context
Planned section details:
- **Title:** {title}
- **Description:** {description}
- **Observations:** {observations}
- **Expected Formats:** {expected_format}
Important data (routes dict and selectors):
- Selectos: {selectors}
- Routes dict: {url_dict}

Section content to evaluate:
{section_content}

### Scoring Criteria (0.0 to 1.0 scale)
1. **Accuracy & Relevance** - The content correctly reflects the section's purpose, aligns with the description,
   and meaningfully incorporates insights from the observations.
2. **Completeness** - The section sufficiently covers all major points implied by the description and observations.
3. **Clarity & Structure** - Writing is well-organized, easy to follow, and uses the correct expected format(s).
4. **Consistency** - Tone, style, and terminology are uniform across the entire section.

### Output Requirements
- **score**: A single float value between 0.0 and 1.0 representing the overall quality, derived from a balanced consideration of all four criteria.
- **improvements**: A list of concise, actionable suggestions to enhance the section. Focus on what would make it closer to an ideal 1.0 score:
  - Each improvement should be specific and practical (e.g., “Add data-supported examples,” “Rephrase for clarity,” “Ensure consistent tone”).
  - If the content is already optimal, return an empty list.
""")
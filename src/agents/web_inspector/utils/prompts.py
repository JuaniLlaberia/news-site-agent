from langchain_core.prompts import ChatPromptTemplate

EXTRACT_ROUTES_PROMPT = ChatPromptTemplate.from_template("""
You are an expert HTML analyzer for Spanish-language news websites.
Your task is to extract only the `href` values (relative routes) that represent **news sections or categories**, not individual articles or generic pages.

Context:
- All routes are in Spanish and **must be returned exactly as they appear** (keep accents, capitalization, etc.).
- Only include routes that belong to news-related sections such as: politics, economy, society, breaking news, security, international/world, technology, science, health, education, culture.
- Routes related to Argentina. Not other countries.

Do NOT include routes related to:
  contact, about, login, signup, privacy, terms, advertising, RSS, newsletters, opinion, services, classifieds, videos, photos, podcasts, sports or any route that doesn't have important information, etc.

Return the result as a JSON list of strings following the structure output.

Example:
HTML:
<a href="/registrarse">Registrarse</a>
<a href="/politica">Política</a>
<a href="/economia">Economía</a>
<a href="/deportes">Ultimos partidos</a>
<a href="/contacto">Contacto</a>
<a href="/tecnologia">Tecnología</a>

Output:
["/politica", "/economia", "/tecnologia"]

Now analyze this HTML content:
{html_content}
""")

EXTRACT_ARTICLES_TAGS_PROMPT = ChatPromptTemplate.from_template("""
You are an expert in web scraping and HTML structure analysis, specialized in Spanish-language news websites.
Your goal is to identify the **CSS selector(s)** that capture all **article links (<a> tags)** in the provided HTML content.

### Instructions
1. **Analyze the HTML structure** carefully and identify all `<a>` elements that correspond to news articles on the page (e.g., headlines, article cards, or news items).
2. **Exclude** any `<a>` elements unrelated to articles, such as navigation menus, ads, footers, social links, login/register links, or category labels.
3. **Extract the minimal CSS selector(s)** that would select **only the article links**.
   - The selector can include tag names, classes, attributes, or hierarchical combinations (e.g. `div.article-card a.headline`, etc.).
   - It must be as **specific and minimal** as possible to target all articles and avoid unrelated links.
4. **Validate** that your CSS selectors collectively capture *all* article links while minimizing false positives.

### Context
- Preserve the original characters, accents, and capitalization in all routes or attribute values.
- Your answer should **not** include the article URLs themselves, only the **CSS selectors** that can be used to extract them.

Return the result as a JSON list of strings following the structure output.

Now analyze this HTML content:
{html_content}
""")

EXTRACT_ARTICLE_CONTENT_PROMPT = ChatPromptTemplate.from_template("""
You are an expert in web structure analysis and HTML parsing, specialized in identifying patterns in Spanish-language news websites.

Your task is to analyze the provided HTML code of a news article and extract the **CSS selectors** needed to retrieve specific elements of the article.

### What to Identify
Extract CSS selectors for the following elements:
- `title_selector`: The main article title.
- `subtitle_selector`: The article subtitle or deck (if available, else None).
- `content_selector`: The main article content paragraphs or text blocks.
- `author_selector`: The author name(s) (if available, else None).
- `img_url_selector`: The main article image (not thumbnails or gallery images) (if available, else None).
- `date_selector`: The publication date (if available, else None).

### Important Guidelines
1. Always return **CSS selectors**, not XPath or text snippets.
2. Use **specific selectors** that uniquely identify the element (avoid overly generic ones like `p` or `div`).
3. If a selector does not exist, return `null` for that field.
4. The HTML is from Spanish-language news sources, so you may encounter labels like "Autor", "Publicado", "Fecha", etc.
5. Do **not** return explanations, reasoning, or commentary — only the final JSON.

Return the result as a JSON list of strings following the structure output.

Now analyze this HTML content:
{html_content}
""")
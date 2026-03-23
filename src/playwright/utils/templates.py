from jinja2 import Template

REPORT_TEMPLATE = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @page {
            margin: 20mm;
            @top-center {
                content: "{{ title }} — {{ date }}";
                font-size: 10pt;
                color: #555;
            }
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 10pt;
                color: #555;
            }
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            color: #003366;
            font-size: 26pt;
            margin-bottom: 5mm;
        }
        h2 {
            margin-top: 1.5em;
            border-bottom: 2px solid #003366;
            padding-bottom: 0.2em;
            color: #003366;
        }
        p {
            text-align: justify;
            margin-bottom: 4mm;
        }
        ul {
            margin-left: 1.2em;
        }
        li {
            margin-bottom: 2mm;
        }
        .page-break {
            page-break-before: always;
        }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    {% for paragraph in intro.split("\\n") %}
    <p>{{ paragraph }}</p>
    {% endfor %}

    {% for section in sections %}
    <div {% if loop.index % 2 == 0 %}class="page-break"{% endif %}>
        <h2>{{ section.section_title }}</h2>

        {% for content_item in section.section_content %}
            {% if content_item.content_type == "narrative" %}
                {% for paragraph in content_item.text.split("\\n") %}
                <p>{{ paragraph }}</p>
                {% endfor %}
            {% elif content_item.content_type == "bullets" %}
                <h4>{{ content_item.items_subtitle }}</h4>
                <ul>
                {% for item in content_item.items %}
                    <li>{{ item }}</li>
                {% endfor %}
                </ul>
            {% endif %}
        {% endfor %}
    </div>
    {% endfor %}

    <h2>Conclusion</h2>
    {% for paragraph in conclusion.split("\\n") %}
    <p>{{ paragraph }}</p>
    {% endfor %}
</body>
</html>
""")

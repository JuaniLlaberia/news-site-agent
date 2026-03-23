import logging
import asyncio
from datetime import datetime
from typing import List, Literal
from playwright.async_api import async_playwright
from src.utils.decorators.retry import retry_with_backoff
from .utils.templates import REPORT_TEMPLATE

class PDFGenerator:
    def __init__(self, format: Literal["A4", "A3"]):
        self.template = REPORT_TEMPLATE
        self.format = format

    @retry_with_backoff(retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            RuntimeError,
            Exception,
        ))
    async def _generate_pdf_bytes(self, title: str, intro: str, sections: List, conclusion: str) -> bytes:
        """
        Internal method to generate PDF bytes asynchronously

        Args:
            title (str): Report title
            intro (str): Report introduction
            sections (list): Report sections
            conclusion (str): Report conclusion
        Returns:
            bytes: The generated PDF as bytes
        """
        html_content = self.template.render(
            title=title,
            intro=intro,
            sections=sections,
            conclusion=conclusion,
            date=datetime.now().strftime("%B %d, %Y")
        )

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html_content, wait_until="networkidle")

            pdf_bytes = await page.pdf(
                format=self.format,
                margin={"top": "20mm", "bottom": "20mm", "left": "20mm", "right": "20mm"},
            )

            await browser.close()
            return pdf_bytes

    def run(self, title: str, intro: str, sections: List, conclusion: str) -> bytes:
        """
        Generate a PDF report and return the bytes.

        Args:
            title (str): The report title
            intro (str): Introduction text
            sections (list): List of sections, each with 'title' and 'content'
            conclusion (str): Conclusion text

        Returns:
            bytes: The generated PDF as bytes
        """
        logging.info("Generating report's pdf bytes...")
        return asyncio.run(self._generate_pdf_bytes(title, intro, sections, conclusion))
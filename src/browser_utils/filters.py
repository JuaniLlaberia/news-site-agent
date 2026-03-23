from bs4 import BeautifulSoup

class HTMLFilter:
    def __init__(self, html: str):
        self.original_html = html
        self.soup = BeautifulSoup(html, "html.parser")

    def keep_only(self,
                  tags: list[str] = None,
                  classes: list[str] = None,
                  include_parents: bool = False,
                  partial_match: bool = False,
                  include_only_href: bool = False):
        """
        Filter HTML to keep only specified tags and/or classes.
        Returns a new HTMLFilter instance for chaining operations.

        Args:
            tags: List of tag names to keep (e.g., ['div', 'a'])
            classes: List of class names to keep
            include_parents: If True, keep parent elements of matched elements
            partial_match: If True, match classes by substring (e.g., 'nav' matches 'nav-menu')
                          If False, only exact class matches (default)
        """
        keep_tags = set(tags or [])
        keep_classes = set(classes or [])

        # Find all elements we want to keep
        elements_to_keep = set()
        for tag in self.soup.find_all(True):
            # Check if tag name matches
            tag_matches = tag.name in keep_tags

            # Check if any class matches
            class_matches = False
            if tag.get("class") and keep_classes:
                if partial_match:
                    class_matches = any(
                        any(keep_class in class_name for keep_class in keep_classes)
                        for class_name in tag["class"]
                    )
                else:
                    class_matches = any(c in keep_classes for c in tag["class"])

            if tag_matches or class_matches:
                if include_parents:
                    elements_to_keep.add(tag.parent)
                else:
                    elements_to_keep.add(tag)

        # Create new soup with only kept elements
        new_soup = BeautifulSoup("", "html.parser")
        for element in elements_to_keep:
            if element.has_attr("href") and include_only_href:
                print(element.get("href"))

            new_soup.append(element.extract())

        return HTMLFilter(str(new_soup))

    def remove(self,
               tags: list[str] = None,
               classes: list[str] = None,
               partial_match: bool = False):
        """
        Remove specified tags and/or classes from HTML.
        Returns a new HTMLFilter instance for chaining operations.

        Args:
            tags: List of tag names to remove (e.g., ['script', 'iframe', 'style'])
            classes: List of class names to remove
            partial_match: If True, match classes by substring
                          If False, only exact class matches (default)
        """
        remove_tags = set(tags or [])
        remove_classes = set(classes or [])

        # Find all elements to remove
        elements_to_remove = []
        for tag in self.soup.find_all(True):
            should_remove = False

            # Check if tag name matches
            if tag.name in remove_tags:
                should_remove = True

            # Check if any class matches
            if not should_remove and tag.get("class") and remove_classes:
                if partial_match:
                    should_remove = any(
                        any(remove_class in class_name for remove_class in remove_classes)
                        for class_name in tag["class"]
                    )
                else:
                    should_remove = any(c in remove_classes for c in tag["class"])

            if should_remove:
                elements_to_remove.append(tag)

        # Remove the elements
        for element in elements_to_remove:
            element.decompose()

        return HTMLFilter(str(self.soup))

    def to_string(self) -> str:
        """Returns the current HTML as a string."""
        return str(self.soup)

    def __str__(self):
        """String representation returns the HTML."""
        return str(self.soup)
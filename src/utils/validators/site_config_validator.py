REQUIRED_FIELDS = ["site_name", "url", "url_dict", "title_selector", "content_selector", "rate_limit"]

def validate_site_config(site_config: dict[str, any]) -> tuple[bool, None | str]:
    """
    Validates the site config

    Args:
        site_config (dict[str, any]): Site configuration as a dictionary
    Returns:
        tuple[bool, None | str]: Tuple containing a bool to know whether the configuration is valid or not, and an error string in case it's not valid
    """
    missing_fields = []

    for key, val in site_config.items():
        if key in REQUIRED_FIELDS:
            if val is not None or (isinstance(val, list) and len(val) > 0):
                continue
            else:
                missing_fields.append(key)

    if len(missing_fields) > 0:
        return False, f"Some fields are missing in the site configuration: {missing_fields}"

    return True, None

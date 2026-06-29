"""Demo tool fixture for the CARE asciicast.

The recording shows `care catalog --tools <dir>` discovering
this file. CARL's `@carl_tool` decorator is intentionally NOT
applied so the demo runs without needing the (not-yet-shipped
in the installed wheel) attribute — the catalog still
enumerates the file by name.
"""


def normalise_temperature(value: float, unit: str = "C") -> float:
    """Round-trip a temperature into Celsius. Standalone so the
    catalog has something to enumerate."""
    if unit.upper() == "F":
        return (value - 32) * 5 / 9
    return value

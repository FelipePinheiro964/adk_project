
import pandas as pd
import requests
import io

from ibm_watsonx_orchestrate.agent_builder.tools import tool


@tool(
    name="process_csv",
    description=(
        "Reads a CSV file and displays, filters or sorts its records "
        "according to the user's request."
    )
)
def process_csv(
    file_path: str,
    operation: str = "read",
    column: str = "",
    order: str = "asc",
    limit: int = 20,
    filter_value: str = "",
    filter_operator: str = "equals"
) -> str:
    """
    Reads and processes a CSV file.

    Args:
        file_path: Signed URL of the uploaded CSV file.
        operation: Operation to perform: read, sort or filter.
        column: Column used for sorting or filtering.
        order: Sort direction: asc or desc.
        limit: Maximum number of rows to display.
        filter_value: Value used to filter records.
        filter_operator: equals, contains, greater_than or less_than.

    Returns:
        A text report containing the requested CSV records.
    """

    try:
        response = requests.get(file_path, timeout=30)
        response.raise_for_status()

        try:
            df = pd.read_csv(
                io.BytesIO(response.content),
                sep=None,
                engine="python",
                encoding="utf-8-sig"
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                io.BytesIO(response.content),
                sep=None,
                engine="python",
                encoding="latin-1"
            )

        df.columns = df.columns.str.strip()

        if df.empty:
            return "The CSV file contains no records."

        original_count = len(df)

        if operation == "sort":
            if column not in df.columns:
                return f"Column '{column}' was not found. Available columns: {list(df.columns)}"

            df = df.sort_values(
                by=column,
                ascending=(order.lower() == "asc"),
                na_position="last"
            )

        elif operation == "filter":
            if column not in df.columns:
                return f"Column '{column}' was not found. Available columns: {list(df.columns)}"

            if not filter_value:
                return "Please provide a value to filter."

            values = df[column]

            if filter_operator == "equals":
                mask = values.astype(str).str.casefold() == filter_value.casefold()

            elif filter_operator == "contains":
                mask = values.astype(str).str.contains(
                    filter_value,
                    case=False,
                    na=False,
                    regex=False
                )

            elif filter_operator in ["greater_than", "less_than"]:
                numeric_values = pd.to_numeric(values, errors="coerce")

                try:
                    target = float(filter_value)
                except ValueError:
                    return "This filter operator requires a numeric value."

                if filter_operator == "greater_than":
                    mask = numeric_values > target
                else:
                    mask = numeric_values < target

            else:
                return "Unsupported filter operator."

            df = df[mask]

        elif operation != "read":
            return "Unsupported operation. Use read, sort or filter."

        if limit < 1:
            return "The limit must be greater than zero."

        result = df.head(min(limit, 100))

        if result.empty:
            return "No records matched your request."

        return (
            f"Total original records: {original_count}\n"
            f"Records after processing: {len(df)}\n"
            f"Records displayed: {len(result)}\n\n"
            f"Columns: {', '.join(df.columns)}\n\n"
            f"{result.to_string(index=False)}"
        )

    except Exception as error:
        return f"Error processing CSV: {str(error)}"

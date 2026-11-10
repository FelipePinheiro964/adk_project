import io
import pandas as pd

from ibm_watsonx_orchestrate.agent_builder.tools import tool


def process_kpi(df: pd.DataFrame) -> str:
    linhas = [
        "KPI report",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Missing values: {int(df.isna().sum().sum())}",
        f"Duplicated rows: {int(df.duplicated().sum())}",
    ]

    numeric = df.select_dtypes("number")
    if numeric.empty:
        linhas.append("\nNo numeric columns found.")
    else:
        resumo = numeric.agg(["sum", "mean", "min", "max"]).T.round(2)
        linhas.append("\nNumeric columns (sum, mean, min, max):")
        linhas.append(resumo.to_string())

    return "\n".join(linhas)


@tool(
    name="process_csv",
    description="Reads, filters, sorts and generates KPI reports from CSV files uploaded by the user."
)
def process_csv(
    file_bytes: bytes,
    operation: str = "read",
    column: str = "",
    order: str = "asc",
    limit: int = 20,
    filter_value: str = "",
    filter_operator: str = "equals"
) -> str:
    """Processes an uploaded CSV file.
    Args:
        file_bytes: Binary content of the uploaded CSV.
        operation: read, sort, filter or kpis.
        column: Column used for sorting or filtering.
        order: asc or desc.
        limit: Maximum number of rows to display.
        filter_value: Value used for filtering.
        filter_operator: equals, contains, greater_than or less_than.
    Returns:
        str: Processed CSV records or a KPI report.
    """
    try:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )

        df.columns = df.columns.str.strip()

        original_count = len(df)

        if operation == "kpis":
            return _kpi_report(df)

        if operation == "sort":
            if column not in df.columns:
                return f"Column not found. Available: {list(df.columns)}"

            df = df.sort_values(
                by=column,
                ascending=order.lower() == "asc",
                na_position="last"
            )

        elif operation == "filter":
            if column not in df.columns:
                return f"Column not found. Available: {list(df.columns)}"

            values = df[column]

            if filter_operator == "equals":
                mask = values.astype(str).str.casefold() == filter_value.casefold()

            elif filter_operator == "contains":
                mask = values.astype(str).str.contains(
                    filter_value, case=False, na=False, regex=False
                )

            elif filter_operator in ("greater_than", "less_than"):
                numeric = pd.to_numeric(values, errors="coerce")
                target = float(filter_value)

                mask = (
                    numeric > target
                    if filter_operator == "greater_than"
                    else numeric < target
                )
            else:
                return "Unsupported filter operator."

            df = df[mask]

        elif operation != "read":
            return "Unsupported operation."

        if limit < 1:
            return "Limit must be greater than zero."

        result = df.head(min(limit, 100))

        if result.empty:
            return "No records found."

        return (
            f"Original records: {original_count}\n"
            f"Records after processing: {len(df)}\n"
            f"Displayed records: {len(result)}\n\n"
            f"{result.to_string(index=False)}"
        )

    except UnicodeDecodeError:
        return "Could not decode the CSV. Check its encoding."
    except Exception as error:
        return f"CSV processing error: {str(error)}"
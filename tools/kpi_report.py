import io
import pandas as pd

from ibm_watsonx_orchestrate.agent_builder.tools import tool


@tool(
    name="relatorio_kpis",
    description="Generates a KPI report (rows, nulls, duplicates and numeric summary) from a CSV file uploaded by the user."
)
def relatorio_kpis(file_bytes: bytes) -> str:
    """Generates a KPI report from an uploaded CSV file.
    Args:
        file_bytes: Binary content of the uploaded CSV.
    Returns:
        str: KPI report in text format.
    """
    try:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )

        df.columns = df.columns.str.strip()

        linhas = [
            "KPI report",
            f"Rows: {len(df)}",
            f"Columns: {len(df.columns)}",
            f"Missing values: {int(df.isna().sum().sum())}",
            f"Duplicated rows: {int(df.duplicated().sum())}",
        ]

        numeric = df.select_dtypes("number")
        if not numeric.empty:
            resumo = numeric.agg(["sum", "mean", "min", "max"]).T.round(2)
            linhas.append("\nNumeric columns (sum, mean, min, max):")
            linhas.append(resumo.to_string())
        else:
            linhas.append("\nNo numeric columns found.")

        return "\n".join(linhas)

    except UnicodeDecodeError:
        return "Could not decode the CSV. Check its encoding."
    except Exception as error:
        return f"CSV processing error: {str(error)}"
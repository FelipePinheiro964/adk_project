import csv
import io
import json
import warnings

import numpy as np
import pandas as pd
from ibm_watsonx_orchestrate.agent_builder.tools import tool, WXOFile

MAX_BYTES = 30 * 1024 * 1024   # limite de upload do Orchestrate
MAX_ROWS = 500_000             # máx. de linhas lidas
MAX_ROWS_SHOWN = 100           # máx. de linhas devolvidas (head/tail/sample)
MAX_COLS = 100                 # máx. de colunas em operações por coluna



def _py(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def _clean(obj):
    return json.loads(json.dumps(obj, default=_py))


def _num(x):
    return None if pd.isna(x) else round(float(x), 4)


def _records(df):
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


def _clamp(n, lo, hi):
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise ValueError("'n' deve ser um número inteiro.")
    return max(lo, min(hi, n))


def _need_col(df, column):
    if not column:
        raise ValueError("Informe o parâmetro 'column' com o nome da coluna.")
    if column in df.columns:
        return column
    for c in df.columns:
        if str(c).strip().lower() == column.strip().lower():
            return c
    raise ValueError(
        f"Coluna '{column}' não existe. Colunas: {[str(c) for c in df.columns[:MAX_COLS]]}"
    )


def _num_cols(df):
    return df.select_dtypes(include="number").columns.tolist()


def _load(wxo_file):
    notes = []
    try:
        name = WXOFile.get_file_name(wxo_file) or ""
        content = WXOFile.get_content(wxo_file)
    except Exception as e:
        raise ValueError(f"Não consegui acessar o arquivo enviado: {e}")

    if name and not name.lower().endswith((".csv", ".tsv", ".txt")):
        raise ValueError(f"'{name}' não parece ser um CSV.")
    if not content:
        raise ValueError("O arquivo está vazio.")
    if len(content) > MAX_BYTES:
        raise ValueError("O arquivo passa de 30 MB.")

    try:
        content.decode("utf-8-sig")
        enc = "utf-8-sig"
    except UnicodeDecodeError:
        enc = "latin-1"

    sample = content[:50_000].decode(enc, errors="replace")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","

    try:
        df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=enc,
                         nrows=MAX_ROWS + 1, low_memory=False)
    except Exception as e:
        raise ValueError(f"CSV inválido ou mal formatado: {e}")

    if df.empty or df.shape[1] == 0:
        raise ValueError("O CSV não tem dados.")
    if df.shape[1] == 1:
        notes.append(f"Só 1 coluna detectada (separador usado: '{sep}'). Verifique o separador.")
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)
        notes.append(f"Arquivo grande: analisadas só as primeiras {MAX_ROWS:,} linhas.")
    return df, notes

def _info(df, n, column):
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "total_nulls": int(df.isna().sum().sum()),
    }


def _columns(df, n, column):
    return {"columns": [str(c) for c in df.columns]}


def _dtypes(df, n, column):
    return {"dtypes": {str(c): str(t) for c, t in df.dtypes.items()}}


def _head(df, n, column):
    return {"rows": _records(df.head(_clamp(n, 1, MAX_ROWS_SHOWN)))}


def _tail(df, n, column):
    return {"rows": _records(df.tail(_clamp(n, 1, MAX_ROWS_SHOWN)))}


def _sample(df, n, column):
    k = min(_clamp(n, 1, MAX_ROWS_SHOWN), len(df))
    return {"rows": _records(df.sample(k, random_state=42))}


def _nulls(df, n, column):
    s = df.isna().sum()
    return {
        "total_nulls": int(s.sum()),
        "columns_without_nulls": int((s == 0).sum()),
        "columns_with_nulls": {
            str(c): {"count": int(v), "pct": round(v / len(df) * 100, 2)}
            for c, v in s.items() if v > 0
        },
    }


def _duplicates(df, n, column):
    subset = [_need_col(df, column)] if column else None
    extra = int(df.duplicated(subset=subset).sum())
    examples = df[df.duplicated(subset=subset, keep=False)].head(5)
    return {
        "scope": str(column) if column else "linha completa",
        "extra_duplicate_rows": extra,
        "pct": round(extra / len(df) * 100, 2),
        "examples": _records(examples),
    }


def _describe(df, n, column):
    cols = _num_cols(df)[:MAX_COLS]
    if not cols:
        raise ValueError("O dataset não tem colunas numéricas para estatísticas descritivas.")
    d = df[cols].describe().T.round(4)
    d.index.name = "column"
    return {"stats": _records(d.reset_index())}


def _nunique(df, n, column):
    return {"unique_values": {str(c): int(v) for c, v in df.iloc[:, :MAX_COLS].nunique().items()}}


def _value_counts(df, n, column):
    col = _need_col(df, column)
    vc = df[col].value_counts(dropna=False).head(_clamp(n, 1, 50))
    return {
        "column": str(col),
        "unique": int(df[col].nunique()),
        "top": [
            {"value": None if pd.isna(v) else v, "count": int(k), "pct": round(k / len(df) * 100, 2)}
            for v, k in vc.items()
        ],
    }


def _stats(df, n, column):
    if column:
        col = _need_col(df, column)
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            raise ValueError(
                f"A coluna '{col}' é {df[col].dtype}, não numérica: min/max/média/mediana não se aplicam."
            )
        cols = [col]
    else:
        cols = _num_cols(df)[:MAX_COLS]
        if not cols:
            raise ValueError("O dataset não tem colunas numéricas.")
    return {"stats": {
        str(c): {"min": _num(df[c].min()), "max": _num(df[c].max()),
                 "mean": _num(df[c].mean()), "median": _num(df[c].median())}
        for c in cols
    }}
    
    
def _find_dates(df):
    """Colunas de texto que parecem datas -> {coluna: Series datetime}."""
    found = {}
    for c in df.select_dtypes(include="object").columns[:MAX_COLS]:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        head = s.head(200)
        if head.str.match(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean() < 0.9:
            continue
        iso = head.str.match(r"^\d{4}-").mean() > 0.9
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=not iso)
        if parsed.notna().mean() >= 0.9:
            found[c] = parsed
    return found


def _kpis(df, n, column):
    total = len(df)
    dates = _find_dates(df)

    # qualidade dos dados
    nulls = (df.isna().mean() * 100)
    nulls = nulls[nulls > 0].sort_values(ascending=False).head(15)
    quality = {
        "duplicate_rows": int(df.duplicated().sum()),
        "columns_with_nulls_pct": {str(c): round(float(v), 2) for c, v in nulls.items()},
        "constant_columns": [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1],
    }

    # colunas numéricas (ids inteiros e únicos são separados, não viram KPI)
    ids, numeric, num_ok = [], {}, []
    for c in _num_cols(df)[:MAX_COLS]:
        s = df[c].dropna()
        if s.empty:
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique() == total:
            ids.append(str(c))
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        numeric[str(c)] = {
            "sum": _num(s.sum()), "mean": _num(s.mean()), "median": _num(s.median()),
            "std": _num(s.std()), "min": _num(s.min()), "max": _num(s.max()),
            "outliers_iqr": out, "outliers_pct": round(out / len(s) * 100, 2),
        }
        num_ok.append(c)

    # colunas categóricas
    categorical = {}
    for c in df.select_dtypes(exclude="number").columns[:MAX_COLS]:
        if c in dates:
            continue
        u = df[c].nunique(dropna=True)
        if u == 0:
            continue
        item = {"unique": int(u)}
        if u <= 50:
            vc = df[c].value_counts(dropna=True).head(5)
            item["top"] = [{"value": v, "pct": round(k / total * 100, 2)} for v, k in vc.items()]
        else:
            item["note"] = "muitos valores distintos (texto livre ou identificador)"
        categorical[str(c)] = item

    # datas e tendência (média da 2ª metade vs 1ª metade do período)
    time = {}
    for c, d in list(dates.items())[:3]:
        v = d.dropna().sort_values()
        info = {
            "min": v.iloc[0].isoformat(),
            "max": v.iloc[-1].isoformat(),
            "span_days": int((v.iloc[-1] - v.iloc[0]).days),
        }
        if len(v) >= 10 and num_ok:
            first, second = v.index[: len(v) // 2], v.index[len(v) // 2:]
            trend = {}
            for k in num_ok[:10]:
                a, b = df.loc[first, k].mean(), df.loc[second, k].mean()
                if pd.notna(a) and pd.notna(b) and a != 0:
                    trend[str(k)] = round(float((b - a) / abs(a) * 100), 2)
            info["mean_change_pct_second_half_vs_first"] = trend
        time[str(c)] = info

    # correlações fortes
    pairs = []
    cols = num_ok[:30]
    if len(cols) >= 2:
        corr = df[cols].corr()
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = corr.loc[a, b]
                if pd.notna(r) and abs(r) >= 0.6:
                    pairs.append({"a": str(a), "b": str(b), "corr": round(float(r), 3)})
        pairs = sorted(pairs, key=lambda p: -abs(p["corr"]))[:5]

    return {
        "size": {"rows": total, "columns": df.shape[1]},
        "quality": quality,
        "likely_id_columns": ids,
        "numeric": numeric,
        "categorical": categorical,
        "datetime": time,
        "strong_correlations": pairs,
    }
    
OPERATIONS = {
    "info": _info, "columns": _columns, "dtypes": _dtypes,
    "head": _head, "tail": _tail, "sample": _sample,
    "nulls": _nulls, "duplicates": _duplicates, "describe": _describe,
    "nunique": _nunique, "value_counts": _value_counts, "stats": _stats,
    "kpis": _kpis,
}


@tool
def process_csv(wxo_file: WXOFile, operation: str = "info", n: int = 5, column: str = "") -> dict:
    """Analisa o arquivo CSV enviado pelo usuário no chat. Use sempre o mesmo arquivo da conversa; nunca peça para reenviar.

    Operações (parâmetro operation):
    - info: linhas, colunas, tipos, memória e total de nulos (use ao receber o arquivo)
    - columns: nomes das colunas
    - dtypes: tipo de cada coluna
    - head / tail / sample: primeiras, últimas ou N linhas aleatórias (use n)
    - nulls: nulos por coluna e total
    - duplicates: linhas duplicadas (opcional: column)
    - describe: estatísticas descritivas das colunas numéricas
    - nunique: valores únicos por coluna
    - value_counts: frequência de valores de uma coluna (use column e n)
    - stats: mínimo, máximo, média e mediana (opcional: column)
    - kpis: perfil completo do dataset para gerar KPIs e insights

    Args:
        wxo_file (WXOFile): Arquivo CSV enviado pelo usuário.
        operation (str): Uma das operações listadas acima.
        n (int): Quantidade de linhas ou itens (head, tail, sample, value_counts).
        column (str): Nome da coluna (value_counts, stats, duplicates).

    Returns:
        dict: status ("ok" ou "error"), operation, rows, columns, notes e result com os dados da operação.
    """
    try:
        op = (operation or "").strip().lower()
        if op not in OPERATIONS:
            return {"status": "error",
                    "message": f"Operação '{operation}' inválida. Use: {list(OPERATIONS)}"}
        df, notes = _load(wxo_file)
        result = OPERATIONS[op](df, n, column)
        return _clean({"status": "ok", "operation": op, "rows": len(df),
                       "columns": df.shape[1], "notes": notes, "result": result})
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Erro inesperado: {type(e).__name__}: {e}"}
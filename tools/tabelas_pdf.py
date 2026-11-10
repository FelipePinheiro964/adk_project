import io
import re
import unicodedata
import zipfile

from ibm_watsonx_orchestrate.agent_builder.tools import tool


def _slug(texto, limite=40):
    texto = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", texto).strip("_").lower()[:limite]


def _num(v):
    v = re.sub(r"[R$%\s]", "", str(v or ""))
    if "," in v:
        v = v.replace(".", "").replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return None


@tool
def processar_pdf_tabelas(pdf_bytes: bytes, juntar: bool = False, nome_base: str = "") -> bytes:
    """Extrai tabelas de um PDF, nomeia os CSVs, une tabelas iguais se pedido e gera uma análise.

    Args:
        pdf_bytes (bytes): PDF enviado pelo usuário.
        juntar (bool): True para unir tabelas com as mesmas colunas em um único CSV.
        nome_base (str): Prefixo dos nomes dos arquivos (assunto do documento). Opcional.

    Returns:
        bytes: ZIP com os CSVs nomeados e o arquivo analise_geral.md.
    """
    import pandas as pd
    import pdfplumber

    tabelas, usados = [], set()
    palavras = pag_com_tab = 0
    prefixo = _slug(nome_base)

    def nome_unico(partes):
        base = "_".join(p for p in partes if p) or "tabela"
        nome, i = base, 2
        while nome in usados:
            nome, i = f"{base}_{i}", i + 1
        usados.add(nome)
        return nome

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        n_pag = len(pdf.pages)
        for num_pag, pagina in enumerate(pdf.pages, start=1):
            palavras += len((pagina.extract_text() or "").split())
            achou = False
            for num_tab, tab in enumerate(pagina.find_tables(), start=1):
                dados = tab.extract()
                if not dados or len(dados) < 2:
                    continue
                achou = True

                # título = última linha de texto acima da tabela
                topo, titulo = tab.bbox[1], ""
                if topo > 20:
                    area = pagina.crop((0, max(topo - 30, 0), pagina.width, topo))
                    linhas = (area.extract_text() or "").splitlines()
                    titulo = _slug(linhas[-1]) if linhas else ""

                cols, vistos = [], {}
                for i, c in enumerate(dados[0], 1):
                    c = re.sub(r"\s+", " ", (c or "").strip()) or f"coluna_{i}"
                    vistos[c] = vistos.get(c, 0) + 1
                    cols.append(c if vistos[c] == 1 else f"{c}_{vistos[c]}")

                tabelas.append({"pag": num_pag, "n": num_tab, "titulo": titulo,
                                "df": pd.DataFrame(dados[1:], columns=cols)})
            pag_com_tab += achou

    if not tabelas:
        raise ValueError("Nenhuma tabela encontrada no PDF.")

    # unir tabelas com as mesmas colunas
    if juntar:
        grupos = {}
        for t in tabelas:
            grupos.setdefault(tuple(c.lower() for c in t["df"].columns), []).append(t)
        finais = []
        for grupo in grupos.values():
            if len(grupo) == 1:
                finais.append(grupo[0])
                continue
            partes = []
            for t in grupo:
                d = t["df"].copy()
                d.columns = grupo[0]["df"].columns
                d["origem"] = f"pag{t['pag']}_tabela{t['n']}"
                partes.append(d)
            finais.append({"pag": grupo[0]["pag"], "n": 0, "titulo": grupo[0]["titulo"],
                           "df": pd.concat(partes, ignore_index=True), "unida": len(grupo)})
    else:
        finais = tabelas

    saida = io.BytesIO()
    with zipfile.ZipFile(saida, "w", zipfile.ZIP_DEFLATED) as zf:
        nomeados = []
        for t in finais:
            if t.get("unida"):
                nome = nome_unico([prefixo, t["titulo"] or "tabela", "unificada"])
            else:
                nome = nome_unico([prefixo, f"pag{t['pag']}", t["titulo"] or f"tabela{t['n']}"])
            nomeados.append((nome, t))
            zf.writestr(f"{nome}.csv", t["df"].to_csv(index=False).encode("utf-8-sig"))

        # análise geral
        rel = ["# Análise geral", "",
               f"- Páginas: {n_pag} (com tabelas: {pag_com_tab})",
               f"- Palavras no documento: {palavras}",
               f"- Tabelas encontradas: {len(tabelas)}",
               f"- CSVs gerados: {len(finais)}", ""]
        for nome, t in nomeados:
            df = t["df"]
            vazios = int(df.replace("", pd.NA).isna().sum().sum())
            origem = f"página {t['pag']}" + (f", {t['unida']} tabelas unidas" if t.get("unida") else "")
            rel += [f"## {nome}.csv", f"- Origem: {origem}",
                    f"- {len(df)} linhas x {len(df.columns)} colunas",
                    f"- Colunas: {', '.join(df.columns)}",
                    f"- Células vazias: {vazios}"]
            for c in df.columns:
                if c == "origem":
                    continue
                vals = [_num(v) for v in df[c] if str(v or "").strip()]
                ok = [v for v in vals if v is not None]
                if vals and len(ok) / len(vals) >= 0.8:
                    rel.append(f"- {c}: mín {min(ok):g}, máx {max(ok):g}, média {sum(ok)/len(ok):.2f}")
            rel.append("")
        zf.writestr("analise_geral.md", "\n".join(rel).encode("utf-8"))

    return saida.getvalue()
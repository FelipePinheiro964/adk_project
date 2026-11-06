
import io
import re
import zipfile

from ibm_watsonx_orchestrate.agent_builder.tools import tool


@tool
def extrair_tabelas_pdf(pdf_bytes: bytes) -> bytes:
    """Procura tabelas em um PDF e devolve um ZIP com um CSV por tabela.

    Args:
        pdf_bytes (bytes): Arquivo PDF enviado pelo usuário.

    Returns:
        bytes: Arquivo ZIP com um CSV para cada tabela encontrada.
    """
    import csv
    import pdfplumber

    def limpar(texto):
        texto = re.sub(r"[^\w\s-]", "", texto or "").strip()
        return re.sub(r"\s+", "_", texto)[:40]

    saida = io.BytesIO()
    usados = set()

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf, \
         zipfile.ZipFile(saida, "w", zipfile.ZIP_DEFLATED) as zf:

        for num_pag, pagina in enumerate(pdf.pages, start=1):
            for num_tab, tabela in enumerate(pagina.find_tables(), start=1):
                dados = tabela.extract()
                if not dados or len(dados) < 2:
                    continue

                # nome = linha de texto logo acima da tabela
                topo = tabela.bbox[1]
                titulo = ""
                if topo > 20:
                    area = pagina.crop((0, max(topo - 30, 0), pagina.width, topo))
                    linhas = (area.extract_text() or "").splitlines()
                    titulo = limpar(linhas[-1]) if linhas else ""

                base = f"pag{num_pag}_{titulo or f'tabela{num_tab}'}"
                nome, i = base, 2
                while nome in usados:
                    nome = f"{base}_{i}"
                    i += 1
                usados.add(nome)

                buf = io.StringIO()
                csv.writer(buf).writerows(
                    [[c or "" for c in linha] for linha in dados]
                )
                zf.writestr(f"{nome}.csv", buf.getvalue().encode("utf-8-sig"))

        if not usados:
            raise ValueError("Nenhuma tabela encontrada no PDF.")

    return saida.getvalue()
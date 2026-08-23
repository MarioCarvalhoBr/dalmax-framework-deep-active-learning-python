import argparse
import json
import os
import re

from PyPDF2 import PdfReader  # type: ignore


def extract_confusion_matrix(pdf_path, output_path):
    """
    Extrai a matriz de confusão de um arquivo PDF entre "True" e "Confusion Matrix",
    e salva os dados relevantes em um arquivo de texto.

    Args:
        pdf_path (str): Caminho do arquivo PDF de entrada.
        output_path (str): Caminho do arquivo de texto de saída.
    """
    try:
        # Ler o PDF
        reader = PdfReader(pdf_path)
        full_text = ""

        # Extrair texto de todas as páginas
        for page in reader.pages:
            full_text += page.extract_text()

        # Extrair conteúdo entre "True" e "Confusion Matrix"
        match = re.search(r"True(.*?)Confusion Matrix", full_text, re.DOTALL)
        if not match:
            print(f"Nenhum intervalo encontrado no arquivo {pdf_path}.")
            return

        # Filtrar apenas as linhas com números dentro do intervalo
        relevant_text = match.group(1)
        matrix_data = re.findall(r"^\d+(?:\s+\d+)+$", relevant_text, re.MULTILINE)

        if not matrix_data:
            print(f"Nenhuma matriz encontrada no intervalo no arquivo {pdf_path}.")
            return

        # Salvar os dados filtrados no arquivo de saída
        with open(output_path, "w") as output_file:
            output_file.write("\n".join(matrix_data))

        print(f"Matriz de confusão salva em: {output_path}")


        # SAlvar em um JSON
        json_path = output_path.replace(".txt", ".json")
        print(f"Salvando matriz de confusão em JSON: {json_path}")

        lista = []
        for data in matrix_data:
            lines = data.split("\n")
            for item in lines:
                items = item.split(" ")
                for i in items:
                    lista.append(int(i))
        # Criar um dicionário
        dicionario = {}
        dicionario["matrix_confusion"] = lista
        # Salvar em um arquivo JSON
        with open(json_path, "w") as json_file:
            json.dump(dicionario, json_file)

    except Exception as e:
        print(f"Erro ao processar o arquivo {pdf_path}: {e}")


def process_all_pdfs(folder_root):
    SEEDS = ["SEED_1", "SEED_2", "SEED_3"]
    for seed in SEEDS:
        NQS = os.listdir(os.path.join(folder_root, seed))
        for nq in NQS:
            print(f"Processando {nq}...")
            input_dir = os.path.join(folder_root, seed, nq)

            # Iterar sobre todos os arquivos PDF no diretório
            folders = os.listdir(input_dir)

            print(f"Processando {len(folders)} arquivos PDF em {input_dir}...")
            print(f"Folders: {folders}")
            for method_folder in folders:
                file_name = os.path.join(input_dir, method_folder, "confusion_matrix.pdf")
                # Verifica se o arquivo existe
                if os.path.exists(file_name):
                    pdf_path_output = file_name.replace(".pdf", ".txt")

                    extract_confusion_matrix(file_name, pdf_path_output)
                else:
                    print(f"Arquivo {file_name} não encontrado.")
                    raise ValueError(f"Arquivo {file_name} não encontrado.")



def main(args):
    folder_root = args.input_dir
    process_all_pdfs(folder_root)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="results/dalmax1/daninhas_full/", help="Input directory path.")
    parser.add_argument("--pattern", type=str, default="SEED*", help="Pattern to match (default is '*', matching all).")

    args = parser.parse_args()

    main(args)

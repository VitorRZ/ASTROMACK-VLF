# -*- coding: utf-8 -*-
"""
Gravador de dados binários, FITS e TXT para o sistema ASTROMACK

Criado em 22/05/2025
Autor: Vitor Rafael Zandarim
"""

import numpy as np
from astropy.table import Table
from astropy.io import fits
import os


def gerar_header_fits(
    data_obs: str,
    hora_obs: str,
    freq: int,
    station: str = "ROPK",
    local:  str = "-23.185230, -46.558557",
    samplerate: int = 96000,
    Rs: int = 200,
    metodo_amp: str = "Esse arquivo contem dados de Fase",
    metodo_fase: str = "Esse arquivo contem dados de Amplitude",
    gps: str = "Simulado",
    notas: str = "Captura continua de 24h com 1s/bloco"
):
    """
    Gera dicionário com metadados para cabeçalho FITS.
    """
    header = {
        "ORIGIN":    "ASTROMACK",
        "DATE-OBS":  data_obs,
        "TIME-LT":  hora_obs,
        "TIME-UT":  hora_obs,
        "STATION":   station,
        "LOCATE":   local,
        "FREQ":      freq,
        "MODUL":     "MSK",
        "SAMPLER":   samplerate,
        "SYMBOLR":   Rs,
        "BITRATE":   2 * Rs,
        "AMPL_SRC":  metodo_amp,
        "PHASESRC":  metodo_fase,
        "GPS":       gps,
        "NOTES":     notas
        
    }
    return header


def salvar_bin(dados, caminho, nome):
    """
    Salva dados em formato binário (.bin) como float64.
    """
    os.makedirs(caminho, exist_ok=True)
    np.array(dados, dtype=np.float64).tofile(os.path.join(caminho, nome + ".bin"))

def salvar_txt(dados, caminho, nome_arquivo, colunas=None, fmt="%.10f"):
    """
    Salva os dados em formato .txt com separador de tabulação.

    Parâmetros:
    - dados: array 1D ou 2D (lista ou NumPy array)
    - caminho: pasta de destino
    - nome_arquivo: nome do arquivo sem extensão
    - colunas: lista de nomes das colunas (opcional)
    - fmt: formato numérico (default: 10 casas decimais)
    """
    import numpy as np
    import os

    os.makedirs(caminho, exist_ok=True)
    caminho_completo = os.path.join(caminho, nome_arquivo + ".txt")

    dados = np.atleast_2d(dados)

    with open(caminho_completo, 'w') as f:
        if colunas:
            f.write("\t".join(colunas) + "\n")
        np.savetxt(f, dados, delimiter="\t", fmt=fmt)

    print(f"[TXT] Arquivo salvo em: {caminho_completo}")


def salvar_fits(caminho, nome_arquivo, dados, header1=None):
    """
    Salva os dados em formato FITS com cabeçalho opcional.
    """
    os.makedirs(caminho, exist_ok=True)
    tabela = Table(dados)

    header = fits.Header()
    if header1:
        for chave, valor in header1.items():
            header[chave] = str(valor)

    hdu = fits.BinTableHDU(data=tabela, header=header)
    caminho_fits = os.path.join(caminho, nome_arquivo + ".fits")
    hdu.writeto(caminho_fits, overwrite=True)
    print(f"Arquivo FITS salvo em '{caminho_fits}'.")

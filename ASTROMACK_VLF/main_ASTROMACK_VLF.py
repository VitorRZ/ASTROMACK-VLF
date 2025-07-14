# -*- coding: utf-8 -*-
"""
ASTROMACK VLF - Sistema de leitura de amplitude e fase de sinais VLF modulados em MSK

Autor: Vitor Rafael Zandarim
Data: 24/04/2025
"""

# =============================================================================
# IMPORTAÇÃO DE MÓDULOS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os

from Modulos.main_Demodulador_MSK2 import main_DMSK
from Modulos.Amplitude import Amplitude_Direta
from Modulos.Gravacao import salvar_txt, salvar_bin, salvar_fits, gerar_header_fits



# =============================================================================
# LOCALIZAÇÃO DE ZONA (MANUAL/CONSULTA)
# =============================================================================
from zoneinfo import available_timezones

def buscar_zonas_por_palavra(chave):
    """
    Busca zonas de tempo que contenham a palavra-chave fornecida (case-insensitive).
    """
    chave = chave.lower()
    zonas = sorted([z for z in available_timezones() if chave in z.lower()])
    return zonas

def local_captura():
    # O usuário digita um termo como 'sao', 'manaus', etc. É mais para consulta.
    entrada = input("Digite parte do nome da cidade ou região (ex: 'america', 'manaus', 'brazil', sao_paulo): ")
    resultados = buscar_zonas_por_palavra(entrada)
    
    if resultados:
        print("\nZonas encontradas:")
        for i, zona in enumerate(resultados):
            print(f"{i+1}. {zona}")
    else:
        print("Nenhuma zona encontrada com esse termo.")

# =============================================================================
# COVERSÂO DE HORARIO LT PARA UT (SEMI-AUTOMATICO)
# =============================================================================

from datetime import datetime
from zoneinfo import ZoneInfo

def obter_diferenca_UTC(data_str, hora_str, zona='America/Sao_Paulo'):
    """
    Retorna o deslocamento entre LT e UT para a data e horário fornecidos,
    considerando automaticamente o horário de verão com base na zona.

    Parâmetros:
        data_str (str): Data no formato 'DD-MM-AAAA' (ex: '10-01-2025').
        hora_str (str): Horário no formato 'HH:MM' (ex: '00:00').
        zona (str): Zona IANA (default = 'America/Sao_Paulo').

    Retorno:
        H (int): Valor de LT - UT (por ex: -3 ou -2)
    """
    try:
        # Converte para objeto datetime com o formato DD-MM-AAAA
        dt_local = datetime.strptime(f"{data_str} {hora_str}", "%d-%m-%Y %H:%M")
    except ValueError:
        raise ValueError("Formato de data inválido. Use 'DD-MM-AAAA' e hora 'HH:MM'.")

    # Aplica a zona e calcula o deslocamento UTC
    dt_zoned = dt_local.replace(tzinfo=ZoneInfo(zona))
    offset_horas = dt_zoned.utcoffset().total_seconds() / 3600

    print(f"[INFO] Offset UTC para {zona} em {data_str} {hora_str} foi de {int(offset_horas)} horas.")
    return int(offset_horas)



# =============================================================================
# PARÂMETROS DE ENTRADA
# =============================================================================

# Identificação do arquivo
Data = "10-01-2025"     # Ex.: 10-01-2025 ou 14-01-2025, etc. -
Hora_de_inicio_da_captura = "00:00"  # Ex.: "01:30" ou "23:59", etc. -
Nome_da_pasta = f'Captura dia {Data}'
Nome_do_arquivo_VLF = f'Captura {Data} 0h00 AM.mat' #Nome da captura Audacity
Nome_do_arquivo_GPS = None  # ou "GPS_simulado10-01-2025.bin"

# Parâmetros do sinal
Rs = 200                  # Taxa de símbolos (baud)
Rb = 2 * Rs               # Taxa de bits
Fc = 21400               # Frequência da portadora (Hz)
Taxa_de_amostragem = 96000  # Hz

# Flags de controle
simulacao = True
Amplitude_antes = False

# Normalização da hora (mesmo depois da captura):
H = -obter_diferenca_UTC(Data, Hora_de_inicio_da_captura, zona='America/Sao_Paulo')

# Parâmetros dos dados
data_obs=Data.replace("-", "-")     # já está no formato ISO
station= "ROPK"
local = "-23.185230, -46.558557"
metodo_fase="Demodulacao com |Fc|"
metodo_amp="Direta" if Amplitude_antes else "RMS + suavizacao"
gps="Simulado" if simulacao else ("Nenhum" if Nome_do_arquivo_GPS is None else "Real")

# Cabeçario dos dados de amplitude
header_amp = gerar_header_fits(
    data_obs=data_obs,
    hora_obs=Hora_de_inicio_da_captura,
    freq=Fc,
    station= station,
    local = local,
    samplerate = Taxa_de_amostragem,
    Rs=Rs,
    metodo_amp=metodo_amp,
    gps=gps
)

# Cabeçario dos dados de fase
header_fase = gerar_header_fits(
    data_obs=data_obs,
    hora_obs=Hora_de_inicio_da_captura,
    freq=Fc,
    station= station,
    local = local,
    samplerate = Taxa_de_amostragem,
    Rs=Rs,
    metodo_fase=metodo_fase,
    gps=gps
)

# =============================================================================
# DEFINIÇÃO DE CAMINHOS
# =============================================================================

diretorio_atual = os.getcwd()
diretorio_de_entrada = os.path.join(diretorio_atual, 'Capturas', Nome_da_pasta)
diretorio_de_pre_processamento = os.path.join(diretorio_atual, 'Pré-processamento')
diretorio_de_resultados = os.path.join(diretorio_atual, 'Resultado final')

caminho_do_arquivo_VLF = os.path.join(diretorio_de_entrada, Nome_do_arquivo_VLF)

# =============================================================================
# CLASSE DE LEITURA POR BLOCOS
# =============================================================================

class LeitorSinalVLF:
    def __init__(self, caminho, Fs=96000, tamanho_bloco=None):
        self.caminho = caminho
        self.Fs = Fs
        self.tamanho_bloco = tamanho_bloco if tamanho_bloco else Fs  # 1 segundo
        self.total_amostras = int(np.fromfile(caminho, dtype=np.float32).size)
        self.total_blocos = self.total_amostras // self.tamanho_bloco
        print(f"Arquivo preparado: {self.total_amostras:,} amostras, {self.total_blocos:,} blocos")

    def __iter__(self):
        return self.gerador_blocos()

    def gerador_blocos(self):
        with open(self.caminho, 'rb') as f:
            for _ in range(self.total_blocos):
                bloco = np.fromfile(f, dtype=np.float32, count=self.tamanho_bloco)
                if bloco.size < self.tamanho_bloco:
                    break
                yield bloco

# =============================================================================
# LEITURA E PROCESSAMENTO DO SINAL VLF
# =============================================================================

if Nome_do_arquivo_GPS is None and not simulacao:
    Sinal_VLF = LeitorSinalVLF(caminho_do_arquivo_VLF, Fs=Taxa_de_amostragem)

    if Amplitude_antes:
        Amplitude_db = np.array(Amplitude_Direta(Sinal_VLF, Taxa_de_amostragem, Rs, Fc))
        salvar_bin(Amplitude_db, diretorio_de_pre_processamento, f"Amplitude_db_Direta_{Data}")
        FE_DK2, FI_DK2, *_ = main_DMSK(Sinal_VLF, None, Taxa_de_amostragem, Rs, Fc)
    else:
        FE_DK2, FI_DK2, _, _, Amp = main_DMSK(Sinal_VLF, None, Taxa_de_amostragem, Rs, Fc)
        Amp = np.array(Amp)

elif Nome_do_arquivo_GPS is not None and not simulacao:
    Sinal_VLF = LeitorSinalVLF(caminho_do_arquivo_VLF, Fs=Taxa_de_amostragem)
    caminho_do_arquivo_GPS = os.path.join(diretorio_de_entrada, Nome_do_arquivo_GPS)
    Sinal_GPS = LeitorSinalVLF(caminho_do_arquivo_GPS, Fs=Taxa_de_amostragem)

    if Amplitude_antes:
        Amplitude_db = np.array(Amplitude_Direta(Sinal_VLF, Taxa_de_amostragem, Rs, Fc))
        salvar_bin(Amplitude_db, diretorio_de_pre_processamento, f"Amplitude_db_Direta_{Data}")
        FE_DK2, FI_DK2, *_ = main_DMSK(Sinal_VLF, Sinal_GPS, Taxa_de_amostragem, Rs, Fc)
    else:
        FE_DK2, FI_DK2, _, _, Amp = main_DMSK(Sinal_VLF, Sinal_GPS, Taxa_de_amostragem, Rs, Fc)
        Amp = np.array(Amp)

elif simulacao:
    from Modulos.Simulacao_GPS import gerar_pulso_GPS
    from tqdm import tqdm

    JITTER_RANGE_MS = 1000
    sinal_base = gerar_pulso_GPS(Taxa_de_amostragem, JITTER_RANGE_MS, 2*Taxa_de_amostragem)
    sinal_base2 = gerar_pulso_GPS(Taxa_de_amostragem, 0, 2*Taxa_de_amostragem)

    len_GPS = int(np.fromfile(caminho_do_arquivo_VLF, dtype=np.float32).size)
    total_segundos = len_GPS // Taxa_de_amostragem
    caminho_simulado = os.path.join(diretorio_de_entrada, f"GPS_simulado{Data}.bin")

    # Gera GPS simulado com jitter alternado
    with open(caminho_simulado, "wb") as f:
        C = 1
        for _ in tqdm(range(total_segundos), desc="Gerando sinal GPS simulado"):
            (sinal_base if C == 1 else sinal_base2).tofile(f)
            C *= -1

    caminho_do_arquivo_GPS = caminho_simulado
    Sinal_VLF = LeitorSinalVLF(caminho_do_arquivo_VLF, Fs=Taxa_de_amostragem)
    Sinal_GPS = LeitorSinalVLF(caminho_do_arquivo_GPS, Fs=Taxa_de_amostragem)

    if Amplitude_antes:
        Amplitude_db = np.array(Amplitude_Direta(Sinal_VLF, Taxa_de_amostragem, Rs, Fc))
        salvar_bin(Amplitude_db, diretorio_de_pre_processamento, f"Amplitude_db_Direta_{Data}")
        FE_DK2, FI_DK2, *_ = main_DMSK(Sinal_VLF, Sinal_GPS, Taxa_de_amostragem, Rs, Fc)
    else:
        FE_DK2, FI_DK2, _, _, Amp = main_DMSK(Sinal_VLF, Sinal_GPS, Taxa_de_amostragem, Rs, Fc)
        Amp = np.array(Amp)

# Conversão final dos arrays
FE_DK2 = np.array(FE_DK2)
FI_DK2 = np.array(FI_DK2)

salvar_bin(FE_DK2, diretorio_de_pre_processamento, f"FE_DK2_{Data}")
salvar_bin(FI_DK2, diretorio_de_pre_processamento, f"FI_DK2_{Data}")

# =============================================================================
# PÓS-PROCESSAMENTO DA AMPLITUDE
# =============================================================================

if not Amplitude_antes:
    epsilon = 1e-6
    P_referencia = 5e-6
    suavisacao = 60
    Amplitude = np.zeros(len(Amp) // (Rb * suavisacao))

    for k, i in enumerate(range(Rb * suavisacao, len(Amp), Rb * suavisacao)):
        Amplitude[k] = max(np.sqrt(np.mean(Amp[i - Rb * suavisacao:i]**2)), epsilon)

    Amplitude_db = -20 * np.log10(Amplitude / P_referencia)
    salvar_bin(Amplitude_db, diretorio_de_pre_processamento, f"Amplitude_db_{Data}")

# =============================================================================
# CÁLCULO E SALVAMENTO DA FASE
# =============================================================================

fase = (-np.unwrap(FE_DK2) * 360) / Fc
salvar_bin(fase, diretorio_de_resultados, f"Diferença_de_fase_{Data}")

# =============================================================================
# GRAVAÇÃO EM FITS E TXT (BACKUP)
# =============================================================================

# Amplitude

salvar_fits(
    caminho=diretorio_de_resultados,
    nome_arquivo=f"Amplitude_db_{Data}",
    dados={"FASE_D": Amplitude_db},
    header1=header_amp
)

# Fase

salvar_fits(
    caminho=diretorio_de_resultados,
    nome_arquivo=f"Diferença_de_fase_{Data}",
    dados={"AMP_D": fase},
    header1=header_fase
)

tempo_UT_Amp = np.linspace(0 + H, 24 + H, len(Amplitude_db))
tempo_UT_Fase = np.linspace(0 + H, 24 + H, len(fase))

dados_amp = np.column_stack((tempo_UT_Amp, Amplitude_db))
salvar_txt(dados_amp, diretorio_de_resultados, f"Amplitude_db_{Data}", colunas=["Tempo_UT", "Amplitude_dB"])

dados_fase = np.column_stack((tempo_UT_Fase, fase))
salvar_txt(dados_fase, diretorio_de_resultados, f"Fase_{Data}", colunas=["Tempo_UT", "Fase_deg"])

# =============================================================================
# PLOTAGEM FINAL (AMPLITUDE, FASE, COMPARAÇÃO)
# =============================================================================

# Amplitude
ColA = '#0093dcff'
plt.figure(figsize=(10, 6))
plt.plot(tempo_UT_Amp[1:], abs(Amplitude_db[1:]), ColA)
plt.title(f'Amplitude {Data}')
plt.ylabel("Amplitude [dB]")
plt.xlabel("Horas UT")
plt.grid()
plt.show()

# Fase
ColF = '#dd9300ff'
plt.figure(figsize=(10, 6))
plt.plot(tempo_UT_Fase, fase, ColF)
plt.title(f'Fase {Data}')
plt.ylabel("Fase [°]")
plt.xlabel("Horas UT")
plt.grid()
plt.show()

# Amplitude vs Fase
fig, ax1 = plt.subplots()
ax1.plot(tempo_UT_Amp[1:], abs(Amplitude_db[1:]), ColA, label=f'Amplitude {Data}')
ax1.set_ylabel('Amplitude [dB]', color=ColA)
ax1.tick_params(axis='y', labelcolor=ColA)

ax2 = ax1.twinx()
ax2.plot(tempo_UT_Fase, fase, ColF, label=f'Fase {Data}')
ax2.set_ylabel('Fase [°]', color=ColF)
ax2.tick_params(axis='y', labelcolor=ColF)

ax1.set_xlabel("Horas UT")
plt.title(f'Comparação do sinal VLF Amplitude Vs Fase - Dia {Data}')
ax1.legend(ax1.get_lines() + ax2.get_lines(), [l.get_label() for l in ax1.get_lines() + ax2.get_lines()])
plt.show()


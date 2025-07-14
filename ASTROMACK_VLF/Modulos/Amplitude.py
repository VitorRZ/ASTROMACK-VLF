import numpy as np
import scipy.signal as signal
from tqdm import tqdm

def filtro_passa_banda(freq_min, freq_max, fs, ordem=5):
    """
    Cria um filtro passa-banda Butterworth.

    Parâmetros:
        freq_min (float): Frequência mínima (Hz).
        freq_max (float): Frequência máxima (Hz).
        fs (float): Taxa de amostragem (Hz).
        ordem (int): Ordem do filtro.

    Retorno:
        b, a (ndarray): Coeficientes do filtro.
    """
    nyquist = 0.5 * fs
    low = freq_min / nyquist
    high = freq_max / nyquist
    return signal.butter(ordem, [low, high], btype='band')


def filtro_IIR(sinal, alpha=0.1):
    """
    Aplica filtro IIR (média móvel exponencial) ao sinal.

    Parâmetros:
        sinal (ndarray): Sinal de entrada.
        alpha (float): Fator de suavização (0 < alpha < 1).

    Retorno:
        ndarray: Sinal suavizado.
    """
    ema = np.zeros_like(sinal)
    ema[0] = sinal[0]
    for i in range(1, len(sinal)):
        ema[i] = alpha * sinal[i] + (1 - alpha) * ema[i - 1]
    return ema


def media_movel(sinal, comprimento):
    """
    Aplica média móvel simples ao sinal.

    Parâmetros:
        sinal (ndarray): Sinal de entrada.
        comprimento (int): Janela da média móvel.

    Retorno:
        ndarray: Sinal suavizado.
    """
    janela = np.ones(comprimento) / comprimento
    return np.convolve(sinal, janela, mode='same')


def Amplitude_Direta(Sinal_VLF, Taxa_de_amostragem, Rs, Fc, 
                     epsilon=1e-12, P_referencia=5e-6, suavizacao=True):
    """
    Calcula a amplitude RMS em dB de um sinal VLF por blocos, com ou sem suavização.

    Parâmetros:
        Sinal_VLF (iterável): Iterador de blocos do sinal.
        Taxa_de_amostragem (float): Taxa de amostragem (Hz).
        Rs (float): Taxa de símbolos (baud).
        Fc (float): Frequência da portadora (Hz).
        epsilon (float): Valor mínimo para evitar log de zero.
        P_referencia (float): Potência de referência para dB.
        suavizacao (bool): Aplica ou não média móvel final.

    Retorno:
        ndarray: Amplitudes em dB (suavizadas ou não).
    """
    largura_banda = Rs / 2
    Amp_dB = []

    for bloco in tqdm(Sinal_VLF, total=Sinal_VLF.total_blocos,
                      desc="Medindo Amplitude por blocos", unit="bloco"):
        
        # Definir a faixa do filtro
        freq_min = Fc - largura_banda
        freq_max = Fc + largura_banda

        # Filtragem passa-banda do bloco
        b, a = filtro_passa_banda(freq_min, freq_max, Taxa_de_amostragem)
        bloco_filtrado = signal.filtfilt(b, a, np.nan_to_num(bloco, nan=0.0))

        # Cálculo da amplitude RMS (dB)
        amplitude_rms = np.sqrt(np.mean(bloco_filtrado**2))
        amplitude_rms = max(amplitude_rms, epsilon)
        Amp_dB.append(-20 * np.log10(amplitude_rms / P_referencia))

    Amp_dB = np.array(Amp_dB)

    if suavizacao:
        # Aplica média móvel e remove bordas para evitar distorções
        Amp_suave = media_movel(Amp_dB, Rs // 2)
        return Amp_suave[Rs // 2 : -Rs // 4]  # retorno cortado para estabilidade
    else:
        return Amp_dB

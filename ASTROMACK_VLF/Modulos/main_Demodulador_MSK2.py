from tqdm import tqdm
import numpy as np
from .Demodulador_MSK2 import demodular_MSK2

def main_DMSK(Sinal_VLF, Sinal_GPS, Taxa_de_amostragem, Rs, Fc, Teste=1):
    """
    Função principal de demodulação MSK para leitura de fase e amplitude.

    Parâmetros:
        Sinal_VLF: iterador de blocos do sinal VLF (classe LeitorSinalVLF)
        Sinal_GPS: iterador do sinal GPS (ou None)
        Taxa_de_amostragem: taxa de amostragem do sinal (Hz)
        Rs: taxa de símbolos (baud)
        Fc: frequência da portadora (Hz)
        Teste: modo de teste (1 = padrão)

    Retorno:
        FE: fase esperada (referência)
        FI: fase integrada (resultado)
        bitss: fluxo de bits demodulados
        ASCII2: sequência ASCII detectada (opcional)
        Amp: vetor de amplitude por bloco
    """
    FE = []
    FI = []
    Amp = []
    bitss = []
    ASCII2 = []

    if Sinal_GPS is None:
        for bloco in tqdm(Sinal_VLF, total=Sinal_VLF.total_blocos, desc="Demodulando blocos", unit="bloco"):
            bits, ASCII_orig, fase_esperada, fase_integrada, Ampli, _, _ = demodular_MSK2(
                np.nan_to_num(bloco, nan=0.0),
                None,
                Fs=Taxa_de_amostragem,
                Rs=Rs,
                Fc=Fc,
                GPS=False,
                extrair_ascii=True,
                Teste=Teste
            )
            ASCII2.extend(ASCII_orig)
            Amp.extend(Ampli)
            FE.extend(fase_esperada)
            FI.extend(fase_integrada)
            bitss.extend(bits)

    else:
        from .Leitor_Sinal import Sincro_Amostras, comparador_de_fase_complexo, pll_sine_gen


        for bloco_VLF, bloco_GPS in tqdm(
            zip(Sinal_VLF, Sinal_GPS),
            total=min(Sinal_VLF.total_blocos, Sinal_GPS.total_blocos),
            desc="Demodulando com GPS", unit="bloco"
        ):
            GPS_senoidal = pll_sine_gen(bloco_GPS, Taxa_de_amostragem)
            Senoide_Amostra = Sincro_Amostras(Taxa_de_amostragem, len(bloco_GPS))
            _, Correcao_GPS_rad = comparador_de_fase_complexo(GPS_senoidal, Senoide_Amostra)

            bits, ASCII_orig, fase_esperada, fase_integrada, Ampli, _, _ = demodular_MSK2(
                np.nan_to_num(bloco_VLF, nan=0.0),
                np.nan_to_num(Correcao_GPS_rad, nan=0.0),
                Fs=Taxa_de_amostragem,
                Rs=Rs,
                Fc=Fc,
                GPS=True,
                extrair_ascii=True,
                Teste=Teste
            )
            ASCII2.extend(ASCII_orig)
            Amp.extend(Ampli)
            FE.extend(fase_esperada)
            FI.extend(fase_integrada)
            bitss.extend(bits)

    # Conversão final para arrays
    return np.array(FE), np.array(FI), bitss, np.int32(ASCII2), Amp

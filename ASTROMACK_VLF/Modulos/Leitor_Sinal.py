
import numpy as np

# -----------------------------------------------------------------------------
#  SINCRONIZAÇÂO DAS AMOSTRAS DO ARQUIVO
# -----------------------------------------------------------------------------

def Sincro_Amostras(Fs, total_samples=None):
    """
    Gera um sinal senoidal complexo de 1Hz local para referência local.
    """
    t = np.arange(total_samples) / Fs
    senoide_complexo = np.cos(2 * np.pi * t) + 1j * np.sin(2 * np.pi * t)
    return senoide_complexo

# -----------------------------------------------------------------------------
#  SUAVIÇÃO DE SINAL DE REFERENCIA DE GPS
# -----------------------------------------------------------------------------

def filtro_mola(fase, alpha=0.05):
    """
    Aplica suavização exponencial tipo "mola" ao vetor de fase.
    alpha pequeno = resposta mais lenta e suave.
    """
    fase_suavizada = np.zeros_like(fase)
    fase_suavizada[0] = fase[0]
    for i in range(1, len(fase)):
        fase_suavizada[i] = alpha * fase[i] + (1 - alpha) * fase_suavizada[i - 1]
    return fase_suavizada

# -----------------------------------------------------------------------------
#  VERIFICADOR DE DIFERENÇA DE FASE ENTRE SINAL LOCAL (AMOSTRAS) E SINAL DE GPS
# -----------------------------------------------------------------------------

def comparador_de_fase_complexo(sinal_GPS_complex, sinal_Amostras):
    """
    Compara dois sinais complexos e retorna o erro de fase em graus, já com unwrapping aplicado.
    """
    erro_complexo = sinal_GPS_complex * np.conj(sinal_Amostras)
    erro_fase_rad = np.angle(erro_complexo)
    erro_fase_rad = np.unwrap(erro_fase_rad)
    
    # Aplica a suavização tipo mola
    erro_fase_rad_suave = filtro_mola(erro_fase_rad, alpha=0.05)
    
    erro_fase_graus = erro_fase_rad_suave * 180 / np.pi
    return erro_fase_graus, erro_fase_rad_suave

# -----------------------------------------------------------------------------
#  CONVERSOR DE PULSOS DE GPS EM SINAL SENOIDAL COMPLEXO DE 1 HZ
# -----------------------------------------------------------------------------

def pll_sine_gen(pulso_GPS, sample_rate, Kp=0.005, Ki=0.00001):
    """
    Converte os pulsos de GPS em um sinal senoidal complexo de 1Hz.
    """
    t = np.arange(len(pulso_GPS)) / sample_rate
    senoide_real = np.zeros_like(t)
    senoide_complexo = np.zeros_like(t, dtype=np.complex64)

    freq = 1.0  # 1 Hz
    phase_error = 0
    integral = 0
    last_phase = 0

    pulse_indices = np.where(pulso_GPS > 0)[0]

    for i in range(len(pulse_indices) - 1):
        start = pulse_indices[i]
        end = pulse_indices[i+1] if i+1 < len(t) else len(t)

        phase_error = (last_phase - 2 * np.pi * freq * (start / sample_rate)) % (2 * np.pi)
        integral += phase_error
        correction = Kp * phase_error + Ki * integral
        last_phase += correction

        t_seg = t[start:end] - t[start]
        fase = 2 * np.pi * freq * t_seg + last_phase
        senoide_real[start:end] = np.sin(fase)
        senoide_complexo[start:end] = np.cos(fase) + 1j * np.sin(fase)

    return senoide_complexo
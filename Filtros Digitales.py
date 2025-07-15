import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, firwin, lfilter, freqz

# Parámetros generales
fs = 1000  # Frecuencia de muestreo
t = np.linspace(0, 1, fs, endpoint=False)

# Señales de prueba: varias frecuencias + ruido blanco
signal_clean = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t) + 0.2*np.sin(2*np.pi*250*t)
noise = np.random.normal(0, 0.4, t.shape)
signal_noisy = signal_clean + noise

# Gráfico de señal original y con ruido
plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.plot(t, signal_clean)
plt.title('Señal limpia')
plt.subplot(2, 1, 2)
plt.plot(t, signal_noisy)
plt.title('Señal con ruido')
plt.tight_layout()
plt.show()

# Función para mostrar respuesta en frecuencia
def plot_frequency_response(b, a, fs, title):
    w, h = freqz(b, a, worN=8000)
    plt.figure()
    plt.plot(0.5*fs*w/np.pi, 20*np.log10(abs(h)))
    plt.title(f'Respuesta en frecuencia: {title}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Ganancia [dB]')
    plt.grid()
    plt.show()

# --- Filtros digitales ---

# 1. Filtro Butterworth Pasa Bajos
b1, a1 = butter(N=4, Wn=100/(fs/2), btype='low')
plot_frequency_response(b1, a1, fs, 'Butterworth Pasa Bajos')
filtered_butter_low = lfilter(b1, a1, signal_noisy)

# 2. Filtro Chebyshev Pasa Altos
b2, a2 = cheby1(N=4, rp=1, Wn=150/(fs/2), btype='high')
plot_frequency_response(b2, a2, fs, 'Chebyshev Pasa Altos')
filtered_cheby_high = lfilter(b2, a2, signal_noisy)

# 3. Filtro FIR Pasa Banda con ventana
b3 = firwin(numtaps=101, cutoff=[80, 180], pass_zero=False, fs=fs)
a3 = 1  # FIR: sólo coeficientes b
plot_frequency_response(b3, a3, fs, 'FIR Pasa Banda (Ventana)')
filtered_fir_band = lfilter(b3, a3, signal_noisy)

# --- Comparación visual ---
def compare_signals(original, filtered, title):
    plt.figure(figsize=(10, 4))
    plt.plot(t, original, label='Original', alpha=0.5)
    plt.plot(t, filtered, label='Filtrada', linewidth=2)
    plt.title(f'Comparación: {title}')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.tight_layout()
    plt.show()

compare_signals(signal_noisy, filtered_butter_low, 'Butterworth Pasa Bajos')
compare_signals(signal_noisy, filtered_cheby_high, 'Chebyshev Pasa Altos')
compare_signals(signal_noisy, filtered_fir_band, 'FIR Pasa Banda')


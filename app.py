import gradio as gr
import matplotlib.pyplot as plt
import obspy
import io
import numpy as np
from PIL import Image
from scipy import signal

# -------------------------------------------------------------------
# Función para calcular H/V simplificado
# -------------------------------------------------------------------
def calcular_hv(streams):
    """
    Recibe una lista de 3 streams (N, E, Z) y devuelve
    frecuencia, amplitud H/V y figura.
    """
    # Extraer datos y frecuencia de muestreo
    datos = []
    fs = None
    for st in streams:
        tr = st[0]
        if fs is None:
            fs = int(tr.stats.sampling_rate)
        datos.append(tr.data)
    
    # Asegurar misma longitud (recortar al mínimo)
    min_len = min(len(d) for d in datos)
    datos = [d[:min_len] for d in datos]
    
    # Calcular espectros de amplitud con Welch
    f, Pxx_N = signal.welch(datos[0], fs=fs, nperseg=min(256, min_len//2))
    _, Pxx_E = signal.welch(datos[1], fs=fs, nperseg=min(256, min_len//2))
    _, Pxx_Z = signal.welch(datos[2], fs=fs, nperseg=min(256, min_len//2))
    
    # H/V: sqrt( (|N|^2 + |E|^2) / |Z|^2 )  -> promedio de amplitudes horizontales
    H = np.sqrt((Pxx_N + Pxx_E) / 2.0)
    V = np.sqrt(Pxx_Z)
    # Evitar división por cero
    V[V < 1e-10] = 1e-10
    hv = H / V
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(f, hv, color='crimson', linewidth=1.5)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Cociente H/V")
    ax.set_title("Cociente Espectral H/V")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)
    
    # Buscar pico principal (frecuencia fundamental)
    idx_peak = np.argmax(hv[(f >= 0.2) & (f <= 20)])  # restringir a rango útil
    f_peak = f[(f >= 0.2) & (f <= 20)][idx_peak]
    hv_peak = hv[(f >= 0.2) & (f <= 20)][idx_peak]
    ax.axvline(f_peak, color='navy', linestyle='--', alpha=0.7, label=f'Pico: {f_peak:.2f} Hz')
    ax.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img, f_peak, hv_peak

# -------------------------------------------------------------------
# Función principal que maneja los archivos y análisis
# -------------------------------------------------------------------
def procesar_archivo(archivos, tipo_analisis):
    if not archivos:
        return None, "Ningún archivo seleccionado."
    
    try:
        # Si solo es un archivo, manejarlo como antes
        if len(archivos) == 1:
            archivo = archivos[0]
            nombre = archivo.lower()
            if not nombre.endswith('.sac'):
                return None, "Solo archivos SAC por ahora."
            stream = obspy.read(archivo)
            traza = stream[0]
            tiempo = traza.times()
            amplitud = traza.data
            fs = int(traza.stats.sampling_rate)
            info = f"SAC: {traza.stats.channel}, {fs} Hz, {traza.stats.npts} pts"
            
            if tipo_analisis in ("Serie Temporal", "Espectrograma"):
                # Diezmar si es necesario
                if len(tiempo) > 10000:
                    paso = len(tiempo) // 10000
                    tiempo = tiempo[::paso]
                    amplitud = amplitud[::paso]
                    fs = fs // paso
                
                fig, ax = plt.subplots(figsize=(8, 4))
                if tipo_analisis == "Serie Temporal":
                    ax.plot(tiempo, amplitud, linewidth=0.5, color='navy')
                    ax.set_xlabel("Tiempo (s)")
                    ax.set_ylabel("Amplitud")
                    ax.set_title(f"Serie Temporal - {archivo}")
                    ax.grid(True, linestyle='--', alpha=0.5)
                else:  # Espectrograma
                    nperseg = min(256, len(amplitud)//4)
                    noverlap = nperseg // 2
                    f, t_espectro, Sxx = signal.spectrogram(amplitud, fs=fs, nperseg=nperseg, noverlap=noverlap)
                    im = ax.pcolormesh(t_espectro, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
                    ax.set_xlabel("Tiempo (s)")
                    ax.set_ylabel("Frecuencia (Hz)")
                    ax.set_title(f"Espectrograma - {archivo}")
                    plt.colorbar(im, ax=ax, label='dB')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img = Image.open(buf)
                plt.close(fig)
                return img, info
            
            else:
                return None, "Para H/V necesitas 3 archivos SAC (N, E, Z)."
        
        # --- H/V con múltiples archivos ---
        elif len(archivos) == 3:
            # Leer los tres streams
            streams = [obspy.read(f) for f in archivos]
            img_hv, f_peak, hv_peak = calcular_hv(streams)
            info = (f"H/V calculado con 3 componentes.\n"
                    f"Frecuencia fundamental estimada: {f_peak:.2f} Hz\n"
                    f"Amplitud del pico: {hv_peak:.2f}")
            return img_hv, info
        
        else:
            return None, "Carga 1 archivo para Serie Temporal/Espectrograma o 3 para H/V."
    
    except Exception as e:
        return None, f"Error al procesar: {str(e)}"

# -------------------------------------------------------------------
# Función simulada del asistente IA (luego conectaremos OpenAI)
# -------------------------------------------------------------------
def interpretar_hv(f_peak, hv_peak):
    if f_peak is None:
        return "No hay datos H/V disponibles. Procesa primero 3 archivos SAC."
    # Clasificación básica según frecuencia
    if f_peak < 1.0:
        tipo_suelo = "blando (posiblemente arcillas o rellenos)"
    elif f_peak < 5.0:
        tipo_suelo = "intermedio (arenas densas o gravas)"
    else:
        tipo_suelo = "rígido (roca o suelo muy compacto)"
    
    interpretacion = (
        f"**Interpretación automática (simulada):**\n"
        f"El pico H/V a **{f_peak:.2f} Hz** sugiere un suelo **{tipo_suelo}**.\n"
        f"Amplitud del pico: **{hv_peak:.2f}**.\n\n"
        f"**Recomendación:** Para construcciones, considera la norma sísmica local. "
        f"Este tipo de suelo puede amplificar las ondas sísmicas. "
        f"Consulta a un ingeniero geofísico para un análisis detallado."
    )
    return interpretacion

# -------------------------------------------------------------------
# Interfaz Gradio
# -------------------------------------------------------------------
with gr.Blocks(title="Procesamiento Sísmico - Prototipo") as demo:
    gr.Markdown("# Prototipo de Procesamiento Sísmico")
    gr.Markdown("Sube **1 archivo SAC** para Serie Temporal/Espectrograma, o **3 archivos SAC (N, E, Z)** para H/V.")
    
    with gr.Row():
        archivos_input = gr.File(file_count="multiple", type="filepath", label="Archivos SAC")
        tipo_input = gr.Radio(["Serie Temporal", "Espectrograma", "H/V"], 
                              label="Tipo de análisis", value="Serie Temporal")
    
    with gr.Row():
        boton_procesar = gr.Button("Procesar")
    
    with gr.Row():
        imagen_output = gr.Image(type="pil", label="Gráfico")
        info_output = gr.Textbox(label="Información", lines=3)
    
    # Sección del asistente IA
    gr.Markdown("---")
    gr.Markdown("### Asistente IA (simulado)")
    with gr.Row():
        f_peak_state = gr.State(None)
        hv_peak_state = gr.State(None)
        boton_interpretar = gr.Button("Interpretar con IA")
        interpretacion_output = gr.Textbox(label="Resultado de la IA", lines=5)
    
    # Eventos
    def procesar_y_guardar_estado(archivos, tipo):
        resultado = procesar_archivo(archivos, tipo)
        img, info = resultado
        # Intentar extraer f_peak y hv_peak del texto de info
        f_peak = None
        hv_peak = None
        if info and "H/V" in info:
            try:
                lines = info.split('\n')
                f_peak = float(lines[1].split(':')[1].strip().split()[0])
                hv_peak = float(lines[2].split(':')[1].strip())
            except:
                pass
        return img, info, f_peak, hv_peak
    
    boton_procesar.click(
        fn=procesar_y_guardar_estado,
        inputs=[archivos_input, tipo_input],
        outputs=[imagen_output, info_output, f_peak_state, hv_peak_state]
    )
    
    boton_interpretar.click(
        fn=interpretar_hv,
        inputs=[f_peak_state, hv_peak_state],
        outputs=interpretacion_output
    )

demo.launch()
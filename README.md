COTAGOR Geophysics – Democratizando el procesamiento sísmico

Este proyecto nace de la frustración de un estudiante que no podía procesar sus datos sin un doctorado. Ahora intento devolverle a la comunidad una herramienta sencilla y abierta.

Con esta aplicación web puedes:
- Seleccionar un método (H/V, espectrograma, serie temporal).
- Subir tus archivos SAC (de 1 o 3 componentes).
- Obtener gráficos profesionales en segundos.
- Recibir una interpretación en español gracias a la IA (Gemini).

¿Tienes ideas, ganas de ayudar o quieres añadir MASW? ¡Pull requests, issues y sugerencias son más que bienvenidos!

Cómo probarlo

Necesitas Python 3.10 y conda. Ejecuta:

\`\`\`bash
conda create -n geo python=3.10 -y
conda activate geo
conda install -c conda-forge obspy pandas matplotlib -y
pip install gradio google-genai
\`\`\`

Luego clona el repo y lanza la app:

\`\`\`bash
git clone https://github.com/[TuUsuario]/thiel_proto.git
cd thiel_proto
python app.py
\`\`\`

Abre tu navegador en `http://127.0.0.1:7860`.

Cómo contribuir
¡Me encantaría recibir tu ayuda! Haz un fork, crea una rama y envía un pull request. También puedes abrir un issue si encuentras errores o tienes ideas.

Construido con ilusión desde Arequipa, Perú.
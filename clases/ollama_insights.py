import os
import streamlit as st
from dotenv import load_dotenv
import requests
import json
from typing import List, Dict, Optional, Any
import pandas as pd

# Cargar variables de entorno
load_dotenv()

class LlamaInsightGenerator:
    """
    Generador de insights usando Llama a través de Ollama (local y gratuito)
    """
    
    def __init__(self, model_name: Optional[str] = None, ollama_url: str = "http://localhost:11434"):
        """
        Inicializa el generador para conectarse a Ollama
        
        Args:
            model_name: Nombre del modelo en Ollama (ej: 'llama3.2', 'gemma3:4b')
                        Si no se proporciona, se usa el de .env o 'llama3.2' por defecto.
            ollama_url: URL base de la API de Ollama (por defecto http://localhost:11434)
        """
        self.ollama_url = ollama_url.rstrip('/')
        self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'llama3.2')
        self.max_tokens = int(os.getenv('OLLAMA_MAX_TOKENS', 50000))
        self.temperature = float(os.getenv('OLLAMA_TEMPERATURE', 0.7))
        self.disponible = False
        
        # Verificar que Ollama esté corriendo y el modelo disponible
        try:
            # Verificar conectividad con Ollama
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                modelos_disponibles = [m['name'] for m in response.json().get('models', [])]
                
                # Buscar si el modelo solicitado está disponible (exacto o como prefijo)
                modelo_encontrado = None
                for m in modelos_disponibles:
                    if m == self.model_name or m.startswith(self.model_name + ':'):
                        modelo_encontrado = m
                        break
                
                if modelo_encontrado:
                    self.model_name = modelo_encontrado  # Usar el nombre completo
                    self.disponible = True
                    st.success(f"✅ Ollama conectado, usando modelo '{self.model_name}'")
                else:
                    st.warning(f"""
                    ⚠️ El modelo '{self.model_name}' no está disponible localmente.
                    
                    Para descargarlo, abre una terminal y ejecuta:
                    ollama pull {self.model_name}
                    
                    Modelos disponibles: {', '.join(modelos_disponibles) if modelos_disponibles else 'ninguno'}
                    """)
            else:
                st.error("""
                ❌ No se pudo conectar con Ollama.
                
                Asegúrate de que Ollama esté corriendo:
                1. Descarga Ollama desde https://ollama.com/
                2. Ejecuta la aplicación o el servicio
                3. Verifica con 'ollama list' en la terminal
                """)
        except requests.exceptions.ConnectionError:
            st.error("""
            ❌ No se pudo conectar con Ollama.
            
            Asegúrate de que Ollama esté instalado y ejecutándose:
            - Windows: Inicia la aplicación Ollama
            - macOS/Linux: Ejecuta 'ollama serve' en la terminal
            """)
        except Exception as e:
            st.warning(f"⚠️ Error inesperado al conectar con Ollama: {str(e)}")
            
    def generar_lista_evidencias(self, df, target, tipos, p_fallo, resultados_condicionales):
        """
        Genera una lista de evidencias (hallazgos) a partir de los datos y análisis.
        """
        evidencias = []
        modelo_metrics = st.session_state.get('metricas_manual', None)
        # 1. Probabilidad a priori
        evidencias.append(f"La probabilidad a priori de {target} es {p_fallo:.3f}.")
        
        # 2. Estadísticas descriptivas de variables numéricas respecto a target
        num_cols = [col for col in tipos['numerica'] if col != target]
        for col in num_cols:
            media_global = df[col].mean()
            media_clase1 = df[df[target]==1][col].mean()
            media_clase0 = df[df[target]==0][col].mean()
            evidencias.append(f"Media de {col}: global={media_global:.2f}, cuando {target}=1: {media_clase1:.2f}, cuando {target}=0: {media_clase0:.2f}.")
        
        # 3. Distribución de variables categóricas
        cat_cols = [col for col in tipos['categorica'] if col != target] + tipos['binaria']
        for col in cat_cols:
            # Proporción de target=1 por cada categoría
            prop = df.groupby(col)[target].mean().sort_values(ascending=False)
            for cat, prob in prop.items():
                evidencias.append(f"Cuando {col} = {cat}, la probabilidad de {target}=1 es {prob:.3f}.")
        
        # 4. Resultados de probabilidades condicionales (si se proporcionan)
        if resultados_condicionales:
            for res in resultados_condicionales:
                evid = res['Evidencia']
                cond = res['Condición']
                p_bayes = list(res.values())[4]
                evidencias.append(f"Para la evidencia '{evid} {cond}'")
                if p_bayes:
                    evidencias.append(f"Según Bayes, la probabilidad sería {p_bayes:.3f}.")
        
        # 5. Métricas del modelo (si se proporcionan)
        if modelo_metrics:
            acc = modelo_metrics.get('accuracy')
            sens = modelo_metrics.get('sensibilidad')
            espec = modelo_metrics.get('especificidad')
            if acc is not None:
                evidencias.append(f"El modelo Naive Bayes tiene una exactitud (accuracy) de {acc:.3f}.")
                evidencias.append(f"Sensibilidad (tasa de detección de {target}=1): {sens:.3f}.")
                evidencias.append(f"Especificidad (tasa de detección de {target}=0): {espec:.3f}.")
            # Matriz de confusión
            cm = modelo_metrics.get('cm')
            if cm is not None:
                vn, fp, fn, vp = cm.ravel()
                evidencias.append(f"Matriz de confusión: Verdaderos Negativos={vn}, Falsos Positivos={fp}, Falsos Negativos={fn}, Verdaderos Positivos={vp}.")
        
        return evidencias

    
    def generar_insights(self, 
                        p_fallo: float,
                        target: str,
                        df: Optional[pd.DataFrame] = None,
                        evidencias_impacto: Optional[List[Dict]] = None,
                        tipos: Optional[Dict] = None,
                        contexto_adicional: Optional[Dict] = None) -> List[str]:
        """
        Genera insights usando Llama a través de Ollama basados en los datos de la aplicación
        
        Args:
            p_fallo: Probabilidad a priori de fallo
            target: Nombre de la variable objetivo
            df: DataFrame con los datos
            evidencias_impacto: Lista de evidencias con impacto
            tipos: Diccionario con tipos de columnas
            contexto_adicional: Información adicional
            
        Returns:
            Lista de insights generados
        """
        
        if not self.disponible:
            return self._generar_insights_basicos(p_fallo, target, evidencias_impacto, df, tipos)
        
        # Construir prompt con los datos disponibles
        prompt = self._construir_prompt_completo(
            p_fallo=p_fallo,
            target=target,
            df=df,
            evidencias_impacto=evidencias_impacto,
            tipos=tipos,
            contexto_adicional=contexto_adicional
        )
        
        try:
            # Preparar la solicitud a Ollama
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_p": 0.95,
                    "top_k": 40,
                }
            }
            
            # Llamar a Ollama
            with st.spinner(f'{self.model_name} analizando los datos...'):
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=60  # Timeout de 60 segundos para respuestas largas
                )
            
            if response.status_code == 200:
                result = response.json()
                respuesta_texto = result.get('response', '')
                if respuesta_texto:
                    insights = self._procesar_respuesta(respuesta_texto)
                    return insights if insights else self._generar_insights_basicos(p_fallo, target, evidencias_impacto, df, tipos)
                else:
                    return self._generar_insights_basicos(p_fallo, target, evidencias_impacto, df, tipos)
            else:
                st.warning(f"⚠️ Error en Ollama (código {response.status_code}): {response.text}")
                return self._generar_insights_basicos(p_fallo, target, evidencias_impacto, df, tipos)
                
        except requests.exceptions.Timeout:
            st.warning("⚠️ La generación de insights está tomando demasiado tiempo. Usando insights básicos.")
            return self._generar_insights_basicos(p_fallo, target, evidencias_impacto, df, tipos)
        except Exception as e:
            st.warning(f"⚠️ Error generando insights con Llama: {str(e)}")
            return self._generar_insights_basicos(p_fallo, target, evidencias_impacto, df, tipos)
    
    def _construir_prompt_completo(self, p_fallo, target, df, evidencias_impacto, tipos, contexto_adicional):
        """Construye un prompt detallado con todos los datos disponibles"""
        # (Este método es idéntico al original, solo asegurarse de que esté bien indentado)
        prompt = f"""Eres un analista de datos experto en análisis de riesgos y fallos. 
Analiza los siguientes datos y genera 3-5 insights CLAVE y ACCIONABLES en español.

CONTEXTO DEL ANÁLISIS:
- Variable objetivo (lo que queremos predecir): '{target}' (fallo/éxito)
- Probabilidad base de fallo (sin condiciones): {p_fallo:.3f} ({p_fallo*100:.1f}%)

"""
        # Información del dataset
        if df is not None:
            prompt += f"\nDATOS DEL DATASET:\n"
            prompt += f"- Total de registros: {len(df)}\n"
            prompt += f"- Total de columnas: {len(df.columns)}\n"
            
            if tipos:
                if tipos.get('numerica'):
                    prompt += f"- Variables numéricas: {len(tipos['numerica'])} columnas\n"
                if tipos.get('categorica'):
                    prompt += f"- Variables categóricas: {len(tipos['categorica'])} columnas\n"
                if tipos.get('binaria'):
                    prompt += f"- Variables binarias: {len(tipos['binaria'])} columnas\n"
        
        # Evidencias con mayor impacto
        if evidencias_impacto and len(evidencias_impacto) > 0:
            prompt += f"\nFACTORES QUE MÁS INFLUYEN EN EL RIESGO:\n"
            
            # Ordenar por impacto y tomar top 5
            df_ev = pd.DataFrame(evidencias_impacto)
            if not df_ev.empty and 'P(Fallo|Evidencia) empírica' in df_ev.columns:
                df_ev['impacto'] = df_ev['P(Fallo|Evidencia) empírica'] - p_fallo
                df_ev = df_ev.sort_values('impacto', ascending=False)
                
                for _, row in df_ev.head(5).iterrows():
                    prob = row.get('P(Fallo|Evidencia) empírica', 0)
                    impacto = prob - p_fallo
                    direccion = "AUMENTA" if impacto > 0 else "REDUCE"
                    prompt += f"- {row.get('Evidencia', 'N/A')} {row.get('Condición', '')}: "
                    prompt += f"P(fallo)={prob:.3f} ({direccion} riesgo en {abs(impacto):.3f})\n"
        
        # Correlaciones con variables numéricas
        if df is not None and target in df.columns and tipos and tipos.get('numerica'):
            try:
                num_cols = [col for col in tipos['numerica'] if col in df.columns and col != target]
                if num_cols:
                    correlaciones = df[num_cols + [target]].corr()[target].drop(target).sort_values(ascending=False)
                    prompt += f"\nVARIABLES NUMÉRICAS MÁS CORRELACIONADAS:\n"
                    for var, corr in correlaciones.head(3).items():
                        prompt += f"- {var}: correlación {corr:.3f}\n"
            except:
                pass
        
        # Instrucciones específicas
        prompt += """
        
INSTRUCCIONES PARA LOS INSIGHTS:
1. Cada insight debe ser UNA línea comenzando con un bullet point (-)
2. Sé específico y cuantifica cuando sea posible (usa los valores numéricos)
3. Enfócate en hallazgos ACCIONABLES (qué podemos hacer con esta información)
4. Identifica patrones interesantes o relaciones inesperadas
5. Menciona tanto factores que AUMENTAN como REDUCEN el riesgo
6. Máximo 5 insights

Ejemplo de buen insight:
- "La presencia de 'historial_previo' aumenta la probabilidad de fallo del 25% al 68%, sugiriendo que clientes con experiencia previa necesitan supervisión especial"

Genera los insights ahora:"""
        
        return prompt
    
    def _procesar_respuesta(self, respuesta: str) -> List[str]:
        """Procesa la respuesta del modelo y extrae los insights"""
        lineas = respuesta.strip().split('\n')
        insights = []
        
        for linea in lineas:
            linea = linea.strip()
            # Buscar líneas que parezcan insights (con bullet points)
            if linea and (linea.startswith('-') or linea.startswith('•') or linea.startswith('*')):
                # Limpiar el formato
                insight = linea.lstrip('-•* ').strip()
                if insight:  # Solo agregar si no está vacío
                    insights.append(f"- {insight}")
            elif linea and len(insights) < 5 and not linea.startswith('#') and not linea.startswith('```'):
                # Si no tiene bullet pero parece un insight, agregarlo
                if any(keyword in linea.lower() for keyword in ['probabilidad', 'riesgo', 'aumenta', 'disminuye', 'factor', 'variable']):
                    insights.append(f"- {linea}")
        
        # Si no se encontraron insights con el formato esperado, tomar las primeras líneas no vacías
        if not insights:
            for linea in lineas[:5]:
                if linea.strip() and not linea.startswith('```'):
                    insights.append(f"- {linea.strip()}")
        
        return insights[:5]  # Máximo 5 insights
    
    def _generar_insights_basicos(self, p_fallo, target, evidencias_impacto, df, tipos):
        """Genera insights básicos cuando el modelo no está disponible"""
        insights = []
        
        # Insight 1: Probabilidad base
        insights.append(f"- La probabilidad base de {target} es {p_fallo:.3f} ({p_fallo*100:.1f}%)")
        
        # Insight 2: Evidencia de mayor impacto
        if evidencias_impacto and len(evidencias_impacto) > 0:
            df_ev = pd.DataFrame(evidencias_impacto)
            if not df_ev.empty and 'P(Fallo|Evidencia) empírica' in df_ev.columns:
                max_idx = df_ev['P(Fallo|Evidencia) empírica'].idxmax()
                max_ev = df_ev.loc[max_idx]
                prob = max_ev['P(Fallo|Evidencia) empírica']
                impacto = prob - p_fallo
                direccion = "aumenta" if impacto > 0 else "disminuye"
                insights.append(f"- '{max_ev.get('Evidencia', 'N/A')}' {direccion} el riesgo: P(fallo)={prob:.3f} (Δ={impacto:+.3f})")
        
        # Insight 3: Correlación más fuerte
        if df is not None and target in df.columns and tipos and tipos.get('numerica'):
            try:
                num_cols = [col for col in tipos['numerica'] if col in df.columns and col != target]
                if num_cols:
                    correlaciones = df[num_cols + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)
                    if len(correlaciones) > 0:
                        max_corr_var = correlaciones.index[0]
                        max_corr_val = df[max_corr_var].corr(df[target])
                        insights.append(f"- '{max_corr_var}' es la variable numérica más relacionada con {target} (correlación: {max_corr_val:.3f})")
            except:
                pass
        
        # Insight 4: Tamaño del dataset
        if df is not None:
            insights.append(f"- Análisis basado en {len(df)} registros y {len(df.columns)} variables")
        
        return insights
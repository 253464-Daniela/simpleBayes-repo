import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict, Optional, Any
import pandas as pd

# Cargar variables de entorno
load_dotenv()

class GeminiInsightGenerator:
    """
    Generador de insights usando Gemini API (gratuito)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el generador con Gemini API
        
        Args:
            api_key: API key de Google AI Studio
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            st.error("""
            ⚠️ No se encontró API key de Gemini. 
            
            Para obtener una gratis:
            1. Ve a https://aistudio.google.com/
            2. Inicia sesión con tu cuenta Google
            3. Haz clic en "Get API key"
            4. Crea una nueva API key
            5. Cópiala y pégala en el archivo .env
            
            Mientras tanto, usando insights básicos.
            """)
            self.disponible = False
            return
        
        # Configurar Gemini
        try:
            genai.configure(api_key=self.api_key)
            self.model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite')
            self.model = genai.GenerativeModel(self.model_name)
            self.max_tokens = int(os.getenv('GEMINI_MAX_TOKENS', 500))
            self.temperature = float(os.getenv('GEMINI_TEMPERATURE', 0.7))
            self.disponible = True
            
            # Probar conexión
            test_response = self.model.generate_content("test", generation_config={"max_output_tokens": 5})
            st.success("✅ Gemini API conectada correctamente")
            
        except Exception as e:
            st.warning(f"⚠️ Error conectando con Gemini: {str(e)}. Usando insights básicos.")
            self.disponible = False
    
    def generar_insights(self, 
                        p_fallo: float,
                        target: str,
                        df: Optional[pd.DataFrame] = None,
                        evidencias_impacto: Optional[List[Dict]] = None,
                        tipos: Optional[Dict] = None,
                        contexto_adicional: Optional[Dict] = None) -> List[str]:
        """
        Genera insights usando Gemini basados en los datos de la aplicación
        
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
            # Configurar generación
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": 0.95,
                "top_k": 40,
            }
            
            # Llamar a Gemini
            with st.spinner('🤔 Gemini analizando los datos...'):
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            
            # Procesar respuesta
            if response and response.text:
                insights = self._procesar_respuesta(response.text)
                return insights if insights else self._generar_insights_basicos(p_fallo, target, evidencias_impacto, df, tipos)
            else:
                return self._generar_insights_basicos(p_fallo, target, evidencias_impacto, df, tipos)
                
        except Exception as e:
            st.warning(f"⚠️ Error generando insights con Gemini: {str(e)}")
            return self._generar_insights_basicos(p_fallo, target, evidencias_impacto, df, tipos)
    
    def _construir_prompt_completo(self, p_fallo, target, df, evidencias_impacto, tipos, contexto_adicional):
        """Construye un prompt detallado con todos los datos disponibles"""
        
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
        """Procesa la respuesta de Gemini y extrae los insights"""
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
        """Genera insights básicos cuando Gemini no está disponible"""
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
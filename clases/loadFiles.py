import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, ConfusionMatrixDisplay
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime

class LoadFiles:
    
    def validar_archivo(archivo):
        """
        Valida que el archivo sea válido antes de procesarlo
        Retorna: (es_valido, mensaje_error, dataframe)
        """
        if archivo is None:
            return False, "No se ha seleccionado ningún archivo", None
        
        # Verificar nombre del archivo
        if not archivo.name:
            return False, "El archivo no tiene nombre", None
        
        # Verificar extensión
        extensiones_permitidas = ['csv', 'xlsx', 'xls']
        file_extension = archivo.name.split('.')[-1].lower()
        
        if file_extension not in extensiones_permitidas:
            return False, f"Extensión no permitida. Usa: {', '.join(extensiones_permitidas)}", None
        
        # Verificar tamaño (opcional, límite de 200MB)
        if archivo.size > 200 * 1024 * 1024:  # 200MB
            return False, "El archivo es demasiado grande (máximo 200MB)", None
        
        try:
            # Intentar leer según extensión
            if file_extension == 'csv':
                # Probar diferentes codificaciones
                try:
                    df = pd.read_csv(archivo, encoding='utf-8')
                except UnicodeDecodeError:
                    archivo.seek(0)  # Reiniciar el puntero del archivo
                    df = pd.read_csv(archivo, encoding='latin1')
            else:
                df = pd.read_excel(archivo)
            
            # Validaciones del DataFrame
            if df.empty:
                return False, "El archivo no contiene datos", None
            
            if len(df.columns) == 0:
                return False, "El archivo no tiene columnas", None
                
            return True, "Archivo válido", df
            
        except pd.errors.EmptyDataError:
            return False, "El archivo está vacío", None
        except pd.errors.ParserError:
            return False, "Error al parsear el archivo. Verifica el formato", None
        except Exception as e:
            return False, f"Error inesperado: {str(e)}", None
    
    def detectar_tipos(df):
        """
        Detecta automáticamente el tipo de cada columna:
        - fecha: si puede convertirse a datetime (al menos 80% exitoso y años entre 1900-2100)
        - binaria: si tiene solo dos valores (0/1, sí/no, verdadero/falso, etc.)
        - numérica: si es de tipo numérico (int/float) y no es binaria
        - categórica: el resto
        """
        tipos = {'fecha': [], 'numerica': [], 'categorica': [], 'binaria': []}
        
        for col in df.columns:
            # Obtener valores únicos no nulos
            unique_vals = df[col].dropna().unique()
            
            # --- Detección de binaria (prioritaria) ---
            # Caso numérico: si solo contiene 0 y 1 (o 0.0 y 1.0)
            if pd.api.types.is_numeric_dtype(df[col]):
                if set(unique_vals).issubset({0, 1}) or set(unique_vals).issubset({0.0, 1.0}):
                    tipos['binaria'].append(col)
                    continue
            
            # Caso texto: normalizamos y verificamos si todos los valores pertenecen a un conjunto binario
            else:
                # Convertir a string, minúsculas y quitar espacios
                unique_str = set(str(v).strip().lower() for v in unique_vals if pd.notna(v))
                binarios_posibles = {'si', 'sí', 'no', 'true', 'false', 'yes', 'no', 'verdadero', 'falso', '1', '0'}
                if unique_str.issubset(binarios_posibles):
                    tipos['binaria'].append(col)
                    continue
            
        # --- Detección de numérica ---
            if pd.api.types.is_numeric_dtype(df[col]):
                # Si llegamos aquí, no es binaria ni fecha, entonces es numérica
                tipos['numerica'].append(col)
                continue
            else:
            # --- Detección de fecha ---
                try:
                    # Convertir con coerce y contar aciertos
                    fecha_convertida = pd.to_datetime(df[col], errors='coerce')
                    proporcion_valida = fecha_convertida.notna().sum() / len(df)
                    # Además, comprobar que los años sean razonables (entre 1900 y 2100)
                    años_validos = True
                    if proporcion_valida > 0:
                        años = fecha_convertida.dropna().dt.year
                        if ((años < 1900) | (años > 2100)).any():
                            años_validos = False
                    if proporcion_valida >= 0.8 and años_validos:
                        tipos['fecha'].append(col)
                        continue
                except:
                    pass
            # --- Si no es nada de lo anterior, es categórica ---
            tipos['categorica'].append(col)
        
        return tipos
    
    def mapear_binaria(series):
        """
        Convierte una serie con valores binarios (texto o numérico) a 0/1 entero.
        Soporta: sí/sí, no, true/false, yes/no, verdadero/falso, 1/0.
        """
        s = series.astype(str).str.lower().str.strip()
        mapeo = {
            'si': 1, 'sí': 1, 'yes': 1, 'true': 1, 'verdadero': 1, '1': 1,
            'no': 0, 'false': 0, 'falso': 0, '0': 0
        }
        return s.map(mapeo).fillna(0).astype(int)

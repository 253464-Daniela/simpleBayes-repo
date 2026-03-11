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

# ------------------------------------------------------------
# Clasificador Naive Bayes simple (combinado)
# ------------------------------------------------------------
class SimpleNaiveBayes:
    def __init__(self):
        self.gnb = GaussianNB()
        self.cnb = CategoricalNB()
        self.num_cols = []
        self.cat_cols = []
        self.label_encoders = {}
        
    def prob_condicional(df, target_col, evid_col, evid_type, threshold=None, valor_cat=None):
        """
        Calcula P(Fallo|Evidencia) y P(Evidencia|Fallo) según el tipo de evidencia.
        - Para numérica: evidencia = valor > threshold
        - Para categórica: evidencia = valor == valor_cat
        """
        if evid_type == 'numerica':
            if threshold is None:
                return None, None
            mask_evid = df[evid_col] > threshold
            p_fallo_dado_evid = df[mask_evid][target_col].mean() if mask_evid.sum() > 0 else 0
            mask_fallo = df[target_col] == 1
            p_evid_dado_fallo = (mask_evid & mask_fallo).sum() / mask_fallo.sum() if mask_fallo.sum() > 0 else 0
            return p_fallo_dado_evid, p_evid_dado_fallo
        elif evid_type == 'categorica':
            if valor_cat is None:
                return None, None
            mask_evid = df[evid_col] == valor_cat
            p_fallo_dado_evid = df[mask_evid][target_col].mean() if mask_evid.sum() > 0 else 0
            mask_fallo = df[target_col] == 1
            p_evid_dado_fallo = (mask_evid & mask_fallo).sum() / mask_fallo.sum() if mask_fallo.sum() > 0 else 0
            return p_fallo_dado_evid, p_evid_dado_fallo


    def bayes_theorem(p_A, p_B_given_A, p_B):
        """Aplica el teorema de Bayes: P(A|B) = P(B|A)*P(A)/P(B)"""
        if p_B == 0:
            return 0
        return (p_B_given_A * p_A) / p_B
        
    def fit(self, X_num, X_cat, y):
        self.num_cols = X_num.columns.tolist() if X_num is not None else []
        self.cat_cols = X_cat.columns.tolist() if X_cat is not None else []
        
        # Verificar que hay al menos 2 clases en y
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            st.warning(f"Solo se detectó una clase ({unique_classes[0]}) en los datos de entrenamiento")
        
        if self.num_cols:
            self.gnb.fit(X_num, y)
        if self.cat_cols:
            X_cat_encoded = pd.DataFrame()
            for col in self.cat_cols:
                le = LabelEncoder()
                X_cat_encoded[col] = le.fit_transform(X_cat[col].astype(str))
                self.label_encoders[col] = le
            self.cnb.fit(X_cat_encoded, y)
    
    def predict(self, X_num, X_cat):
        probas = np.zeros(len(X_num) if X_num is not None else len(X_cat))
        n_models = 0
        
        if self.num_cols and X_num is not None:
            proba_num = self.gnb.predict_proba(X_num)
            # Verificar si hay al menos 2 clases
            if proba_num.shape[1] > 1:
                probas += proba_num[:, 1]
                n_models += 1
            else:
                # Si solo hay una clase, la probabilidad de la clase 1 es 0
                # (asumiendo que la clase presente es la 0)
                st.warning("GaussianNB: Solo se detectó una clase en los datos de entrenamiento")
        
        if self.cat_cols and X_cat is not None:
            X_cat_encoded = pd.DataFrame()
            for col in self.cat_cols:
                le = self.label_encoders[col]
                X_cat_encoded[col] = le.transform(X_cat[col].astype(str))
            
            proba_cat = self.cnb.predict_proba(X_cat_encoded)
            # Verificar si hay al menos 2 clases
            if proba_cat.shape[1] > 1:
                probas += proba_cat[:, 1]
                n_models += 1
            else:
                st.warning("CategoricalNB: Solo se detectó una clase en los datos de entrenamiento")
        
        # Evitar división por cero
        if n_models > 0:
            probas = probas / n_models
        else:
            # Si no hay modelos válidos, retornar todos como clase mayoritaria
            probas = np.zeros(len(probas))
        
        return (probas > 0.5).astype(int), probas


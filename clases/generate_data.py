import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generar_datos():
    # Fijar semilla para reproducibilidad
    np.random.seed(42)

    # Generar fechas para 200 días
    fechas = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(200)]

    # Variables de entrada con relaciones complejas
    n = 200

    # Temperatura con patrón estacional y ruido
    temperatura = 25 + 5 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 2, n)

    # Humedad correlacionada negativamente con temperatura (patrón real)
    humedad = 70 - 0.5 * (temperatura - 25) + np.random.normal(0, 5, n)
    humedad = np.clip(humedad, 30, 95)  # Limitar a rangos realistas

    # Presión atmosférica con variaciones
    presion = 1010 + np.random.normal(0, 5, n) + 2 * np.sin(np.linspace(0, 2*np.pi, n))

    # Variables categóricas con distribuciones realistas
    tipo_mantenimiento = np.random.choice(['bajo', 'medio', 'alto'], n, p=[0.4, 0.35, 0.25])
    turno = np.random.choice(['mañana', 'tarde', 'noche'], n, p=[0.4, 0.35, 0.25])

    # Variable binaria con probabilidad basada en otras variables
    prob_alarma = 0.3 + 0.3 * (temperatura > 28) + 0.2 * (np.array(tipo_mantenimiento) == 'bajo') - 0.1 * (presion > 1015)
    prob_alarma = np.clip(prob_alarma, 0.1, 0.8)  # Limitar probabilidades
    alarma_previa = np.random.binomial(1, prob_alarma)

    # Variable objetivo (fallo) con relación no determinística
    # Probabilidad base
    prob_fallo = 0.1

    # Factores que aumentan probabilidad
    prob_fallo += 0.25 * (temperatura > 29)  # Temperatura alta
    prob_fallo += 0.15 * (np.array(tipo_mantenimiento) == 'bajo')  # Mantenimiento bajo
    prob_fallo += 0.20 * (alarma_previa == 1)  # Alarma previa
    prob_fallo += 0.10 * (humedad < 50)  # Humedad baja
    prob_fallo += 0.10 * (turno == 'noche')  # Turno noche
    prob_fallo -= 0.05 * (presion > 1015)  # Presión alta reduce riesgo

    # Añadir interacciones
    prob_fallo += 0.15 * ((temperatura > 28) & (alarma_previa == 1))
    prob_fallo += 0.10 * ((np.array(tipo_mantenimiento) == 'bajo') & (humedad < 55))

    # Asegurar que esté en rango [0, 1]
    prob_fallo = np.clip(prob_fallo, 0.05, 0.85)

    # Generar fallo con probabilidad calculada
    fallo = np.random.binomial(1, prob_fallo)

    # Crear DataFrame
    df = pd.DataFrame({
        'fecha': fechas,
        'temperatura': np.round(temperatura, 1),
        'humedad': np.round(humedad, 1),
        'presion': np.round(presion, 1),
        'tipo_mantenimiento': tipo_mantenimiento,
        'turno': turno,
        'alarma_previa': ['Sí' if x == 1 else 'No' for x in alarma_previa],
        'fallo': fallo
    })

    # Mostrar información
    print("=" * 50)
    print("DATASET GENERADO - NO PERFECTAMENTE SEPARABLE")
    print("=" * 50)
    print(f"\nTotal registros: {len(df)}")
    print(f"Período: {df['fecha'].min().date()} a {df['fecha'].max().date()}")
    print("\nDistribución de fallos:")
    print(df['fallo'].value_counts())
    print(f"Probabilidad a priori de fallo: {df['fallo'].mean():.3f}")

    print("\n" + "=" * 50)
    print("RELACIONES NO DETERMINÍSTICAS:")
    print("=" * 50)

    # Mostrar que no hay relaciones perfectas
    print("\n1. Misma alarma_previa, diferentes fallos:")
    ejemplo_alarma_si = df[df['alarma_previa'] == 'Sí'].head(8)
    print(ejemplo_alarma_si[['alarma_previa', 'fallo', 'temperatura', 'tipo_mantenimiento']].to_string(index=False))

    print("\n2. Misma temperatura, diferentes fallos:")
    temp_alta = df[(df['temperatura'] > 29) & (df['temperatura'] < 30)].head(8)
    print(temp_alta[['temperatura', 'fallo', 'alarma_previa', 'tipo_mantenimiento']].to_string(index=False))

    print("\n3. Mismo mantenimiento, diferentes resultados:")
    mantenimiento_bajo = df[df['tipo_mantenimiento'] == 'bajo'].head(8)
    print(mantenimiento_bajo[['tipo_mantenimiento', 'fallo', 'temperatura', 'humedad']].to_string(index=False))

    # Verificar correlaciones
    print("\n" + "=" * 50)
    print("CORRELACIONES (ninguna perfecta):")
    print("=" * 50)
    corr_matrix = df[['temperatura', 'humedad', 'presion', 'fallo']].corr()
    print(corr_matrix['fallo'].sort_values(ascending=False))

    # Guardar a CSV
    df.to_csv('dataset_no_perfecto.csv', index=False)
    print("\n✅ Dataset guardado como 'dataset_no_perfecto.csv'")

    # Mostrar estadísticas básicas
    print("\n" + "=" * 50)
    print("ESTADÍSTICAS BÁSICAS:")
    print("=" * 50)
    print(df.describe())

    # Mostrar primeras filas
    print("\n" + "=" * 50)
    print("PRIMERAS 10 FILAS:")
    print("=" * 50)
    print(df.head(10).to_string())
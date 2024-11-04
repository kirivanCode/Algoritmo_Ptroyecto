import random
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import requests
from faker import Faker
import joblib
import os
from datetime import datetime, time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

if not os.path.exists('models'):
    os.makedirs('models')

# Configuración inicial donde hago la conexion con la api zzzzz
fake = Faker()
BASE_URL = "http://localhost:8000/api"

class ScheduleOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_model = None
        self.is_fitted = False
        self.feature_names = [
            'experiencia', 'calificacion_alumno', 'alumnos', 
            'bloques', 'horarios_disponibles', 'capacidad_salon',
            'conflictos_horario', 'carga_profesor'
        ]
        # Intentar cargar el modelo guardado al inicializar
        self.load_model()
        self.performance_history = []
        self.pattern_database = {}
        self.slot_duration = 45  # duración en minutos de cada slot
        self.model_params = {
            'test_size': 0.2,
           
        }
        self.HORAS_PERMITIDAS = [
            "06:00", "07:30", "09:00", 
            "10:30",  "12:00", "13:30", 
            "15:00",  "16:30",  "18:30", "20:15"
        ]
        # otros parámetros que pueda necesitar
        

    def save_model(self):
        """Guarda el modelo y el scaler en archivos"""
        if self.is_fitted:
            try:
                # Crear un directorio modelos si no existe
                if not os.path.exists('models'):
                    os.makedirs('models')
                
                # Guardar el modelo y el scaler
                joblib.dump(self.best_model, 'models/best_model.joblib')
                joblib.dump(self.scaler, 'models/scaler.joblib')
                # Guardar el estado de entrenamiento
                joblib.dump(self.is_fitted, 'models/is_fitted.joblib')
                return True
            except Exception as e:
                st.error(f"Error al guardar el modelo: {str(e)}")
                return False
            

    def load_model(self):
            """Carga el modelo y el scaler desde archivos"""
            try:
                if os.path.exists('models/best_model.joblib') and \
                os.path.exists('models/scaler.joblib') and \
                os.path.exists('models/is_fitted.joblib'):
                    
                    self.best_model = joblib.load('models/best_model.joblib')
                    self.scaler = joblib.load('models/scaler.joblib')
                    self.is_fitted = joblib.load('models/is_fitted.joblib')
                    return True
                return False
            except Exception as e:
                st.error(f"Error al cargar el modelo: {str(e)}")
                return False 
    
    

    
    @st.cache_data
    def get_data(_self, endpoint):  # odtengo los datos
        try:
            response = requests.get(f"{BASE_URL}/{endpoint}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Error al obtener datos de {endpoint}: {str(e)}")
            return None
        
    #preparo los datos

    def prepare_features(self, df_profesores, df_materias, df_salones, df_horarios, df_profesor_materia):
        features = []
        labels = []
        conflicts = []

        # Cargo el dicionario para el modelo por medio del id del profesor
        carga_profesor = df_profesor_materia.groupby('profesor_id').size().to_dict()
        
        for _, prof_mat in df_profesor_materia.iterrows():
            profesor = df_profesores[df_profesores['id'] == prof_mat['profesor_id']].iloc[0]
            materia = df_materias[df_materias['id'] == prof_mat['materia_id']].iloc[0]
            
            # calculo si hay conflictos de ids entre los datos
            horarios_prof = df_horarios[df_horarios['profesor_id'] == prof_mat['profesor_id']]
            conflictos = self.calcular_conflictos(horarios_prof)
            
            # Para cada salón disponible
            for _, salon in df_salones.iterrows():
                if salon['capacidad_alumnos'] >= materia['alumnos']:
                    feature = [
                        prof_mat['experiencia'],
                        prof_mat['calificacion_alumno'],
                        materia['alumnos'],
                        materia['bloques'],
                        len(horarios_prof),
                        salon['capacidad_alumnos'],
                        conflictos,
                        carga_profesor.get(prof_mat['profesor_id'], 0)
                    ]
                    
                    features.append(feature)
                    labels.append(1)  # Combinación válida
                    conflicts.append(conflictos)

        # Generar ejemplos negativos más realistas, o como el amor de ella
        negative_examples = self.generate_negative_examples(
            df_profesores, df_materias, df_salones, df_horarios, 
            df_profesor_materia, len(features)
        )
        
        features.extend(negative_examples[0])
        labels.extend(negative_examples[1])
        conflicts.extend(negative_examples[2])

        return np.array(features), np.array(labels), conflicts

   

    def hay_solapamiento(self, inicio1, fin1, inicio2, fin2):
        """Verifica si hay solapamiento entre dos rangos de tiempo"""
        if isinstance(inicio1, str):
            inicio1 = self.parse_time(inicio1)
        if isinstance(fin1, str):
            fin1 = self.parse_time(fin1)
        if isinstance(inicio2, str):
            inicio2 = self.parse_time(inicio2)
        if isinstance(fin2, str):
            fin2 = self.parse_time(fin2)

        return (inicio1 < fin2 and fin1 > inicio2)
    
    def calcular_conflictos(self, horarios_prof):
        if horarios_prof.empty:
            return 0
        
        conflictos = 0
        horarios_list = horarios_prof.values.tolist()
        
        for i in range(len(horarios_list)):
            for j in range(i + 1, len(horarios_list)):
                if self.hay_solapamiento(
                    horarios_list[i][2], horarios_list[i][3],  # hora_inicio, hora_fin del primer horario
                    horarios_list[j][2], horarios_list[j][3]   # hora_inicio, hora_fin del segundo horario
                ):
                    conflictos += 1
        
        return conflictos

    def generate_negative_examples(self, df_profesores, df_materias, df_salones, 
                                 df_horarios, df_profesor_materia, num_samples):
        features = []
        labels = []
        conflicts = []
        
        for _ in range(num_samples):
            profesor = df_profesores.sample(1).iloc[0]
            materia = df_materias.sample(1).iloc[0]
            salon = df_salones.sample(1).iloc[0]
            
            # Verificar si es una combinación inválida
            if df_profesor_materia[(df_profesor_materia['profesor_id'] == profesor['id']) & 
                                 (df_profesor_materia['materia_id'] == materia['id'])].empty:
                
                horarios_prof = df_horarios[df_horarios['profesor_id'] == profesor['id']]
                conflictos = self.calcular_conflictos(horarios_prof)
                
                feature = [
                    np.random.randint(1, 5),  # experiencia aleatoria, no se es mas random
                    np.random.randint(1, 5),  # calificación xd
                    materia['alumnos'],
                    materia['bloques'],
                    len(horarios_prof),
                    salon['capacidad_alumnos'],
                    conflictos,
                    len(df_profesor_materia[df_profesor_materia['profesor_id'] == profesor['id']])
                ]
                
                features.append(feature)
                labels.append(0)
                conflicts.append(conflictos)
        
        return features, labels, conflicts

    def train_model(self, X, y, model_params):
    # Escalar los datos

    
            X_scaled = self.scaler.fit_transform(X)
    
            # Split de datos usando los parámetros
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, 
                 test_size=0.2,  # Valor directo
                random_state=model_params['random_state']
            )
            
            if model_params['model_type'] == 'knn':
                model = KNeighborsClassifier()
                param_grid = {
                    'n_neighbors': [model_params['n_neighbors']],
                    'weights': [model_params['weights']],
                    'metric': [model_params['metric']]
                }
            else:  # RandomForest
                model = RandomForestClassifier(random_state=model_params['random_state'])
                param_grid = {
                    'n_estimators': [model_params['n_estimators']],
                    'max_depth': [model_params['max_depth']],
                    'min_samples_split': [model_params['min_samples_split']],
                    'min_samples_leaf': [model_params['min_samples_leaf']]
                }
            
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=model_params['cv_folds'],
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # entrenamiento de los diferentes estimados, como la composicion y el resto
            
            self.best_model = grid_search.best_estimator_
            self.is_fitted = True
            
            # Guardo el modelo entrenado
            if self.save_model():
                st.success("Modelo guardado exitosamente")
            
            # Evaluación del modelo
            y_pred = self.best_model.predict(X_test)
            
            results = {
                'best_params': grid_search.best_params_,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            if isinstance(self.best_model, RandomForestClassifier):
                results['feature_importance'] = dict(zip(
                    self.feature_names, 
                    self.best_model.feature_importances_
        ))
    
            return results

    def generate_schedule(self, df_profesores, df_materias, df_salones, 
                         df_horarios, df_profesor_materia, optimization_params):
        # Verificar si el modelo y el scaler están entrenados
        if not self.is_fitted:
            st.error("El modelo no ha sido entrenado. Por favor, entrene el modelo primero.")
            return {
                "status": "ERROR",
                "horario_generado": [],
                "warnings": [],
                "errors": ["El modelo debe ser entrenado antes de generar el horario"]
            }

        horario_generado = []
        warnings = []
        errors = []
        
        # Ordenar materias segun la prioridad
        df_materias_sorted = df_materias.sort_values(
            ['alumnos', 'bloques'], 
            ascending=[False, False]
        )
        
        for _, materia in df_materias_sorted.iterrows():
            if materia['alumnos'] < optimization_params['min_alumnos']:
                warnings.append(f"Materia {materia['nombre']} no tiene suficientes alumnos")
                continue
                
            clases_asignadas = self.asignar_clases(
                materia, df_profesores, df_salones, df_horarios,
                df_profesor_materia, optimization_params, horario_generado
            )
            
            #toca meter una parte donde se pueda medir
            if clases_asignadas < materia['bloques']:
                warnings.append(
                    f"No se pudieron asignar todos los horarios para {materia['nombre']}"
                )
        
        return {
            "status": "OPTIMAL" if len(warnings) == 0 else "FEASIBLE",
            "horario_generado": horario_generado,
            "warnings": warnings,
            "errors": errors
        }

    def asignar_clases(self, materia, df_profesores, df_salones, df_horarios,
                   df_profesor_materia, optimization_params, horario_generado):
        clases_asignadas = 0
        while clases_asignadas < materia['bloques'] * 2:  # Multiplicamos por 2 para crear dos bloques de 45 min
            mejor_asignacion = self.encontrar_mejor_asignacion(
                materia, df_profesores, df_salones, df_horarios,
                df_profesor_materia, optimization_params, horario_generado
            )

            if mejor_asignacion is None:
                break

            profesor, salon, horario, score = mejor_asignacion

            # Crear dos clases de 45 minutos
            for i in range(2):
                hora_inicio = (datetime.combine(datetime.min, horario['hora_inicio']) + timedelta(minutes=45*i)).time()
                hora_fin = (datetime.combine(datetime.min, hora_inicio) + timedelta(minutes=45)).time()

                clase = {
                    'grupo': fake.unique.bothify(text='??##', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                    'dia_semana': horario['dia'],
                    'hora_inicio': hora_inicio.strftime('%H:%M'),  # Ajusta el formato según sea necesario
                    'hora_fin': hora_fin.strftime('%H:%M'),  # Ajusta el formato según sea necesario
                    'alumnos': int(materia['alumnos']),
                    'materia_id': int(materia['id']),
                    'salon_id': int(salon['id']),
                    'profesor_id': int(profesor['id'])
                }

                horario_generado.append(clase)
                clases_asignadas += 1

            # Actualizar disponibilidad
            df_horarios = df_horarios[
                ~((df_horarios['profesor_id'] == profesor['id']) &
                (df_horarios['dia'] == horario['dia']) &
                (df_horarios['hora_inicio'].apply(self.parse_time) <= horario['hora_inicio']) &
                (df_horarios['hora_fin'].apply(self.parse_time) >= horario['hora_fin']))
            ]

        return clases_asignadas // 2  # Devolvemos el número de bloques de 90 minutos asignados


    def encontrar_mejor_asignacion(self, materia, df_profesores, df_salones,
                             df_horarios, df_profesor_materia,
                             optimization_params, horario_generado):
        mejor_score = -1
        mejor_asignacion = None
        

        for _, profesor in df_profesores.iterrows():
            if self.get_carga_actual(profesor['id'], horario_generado) >= optimization_params['max_carga_profesor']:
                continue

            prof_mat = df_profesor_materia[
                (df_profesor_materia['profesor_id'] == profesor['id']) &
                (df_profesor_materia['materia_id'] == materia['id'])
            ]

            if prof_mat.empty:
                continue

            horarios_disponibles = df_horarios[
            df_horarios['profesor_id'] == profesor['id']
            ]   

            for _, horario in horarios_disponibles.iterrows():
                hora_inicio_disponible = self.parse_time(str(horario['hora_inicio']))
                hora_fin_disponible = self.parse_time(str(horario['hora_fin']))

                # Generar múltiples horas de inicio aleatorias
                horas_inicio_posibles = []
                for hora in self.HORAS_PERMITIDAS:
                    hora_inicio = self.parse_time(hora)
                    if hora_inicio >= hora_inicio_disponible and (
                        datetime.combine(datetime.today(), hora_inicio) +
                        timedelta(minutes=90)
                    ).time() <= hora_fin_disponible:
                        horas_inicio_posibles.append(hora_inicio)

                if not horas_inicio_posibles:
                    continue

                # Seleccionar una hora de inicio aleatoria
                hora_inicio_aleatoria = random.choice(horas_inicio_posibles)
                hora_fin_aleatoria = (
                    datetime.combine(datetime.today(), hora_inicio_aleatoria) +
                    timedelta(minutes=90)
                ).time()

                nuevo_horario = {
                    'dia': horario['dia'],
                    'hora_inicio': hora_inicio_aleatoria,
                    'hora_fin': hora_fin_aleatoria
                }


                # El resto de la función permanece igual
                if self.hay_conflicto_horario(profesor['id'], nuevo_horario, horario_generado):
                    continue

                for _, salon in df_salones.iterrows():
                    if salon['capacidad_alumnos'] < materia['alumnos']:
                        continue

                    if self.salon_ocupado(salon['id'], nuevo_horario, horario_generado):
                        continue

                    features = [
                        prof_mat['experiencia'].iloc[0],
                        prof_mat['calificacion_alumno'].iloc[0],
                        materia['alumnos'],
                        materia['bloques'],
                        len(horarios_disponibles),
                        salon['capacidad_alumnos'],
                        self.calcular_conflictos(horarios_disponibles),
                        self.get_carga_actual(profesor['id'], horario_generado)
                    ]

                    features_scaled = self.scaler.transform([features])
                    score = self.best_model.predict_proba(features_scaled)[0][1]

                    if score > mejor_score:
                        mejor_score = score
                        mejor_asignacion = (profesor, salon, nuevo_horario, score)

        return mejor_asignacion
    
    def parse_time(self, time_str):
        """Convierte una cadena de tiempo en un objeto time"""
        if isinstance(time_str, (datetime, time)):
            return time_str.time() if isinstance(time_str, datetime) else time_str
        try:
            return datetime.strptime(time_str, '%H:%M:%S').time()
        except ValueError:
            return datetime.strptime(time_str, '%H:%M').time()


    def get_carga_actual(self, profesor_id, horario_generado):
        return len([
            clase for clase in horario_generado 
            if clase['profesor_id'] == profesor_id
        ])

    def hay_conflicto_horario(self, profesor_id, horario_nuevo, horario_generado):
        for clase in horario_generado:
            if (clase['profesor_id'] == profesor_id and
                clase['dia_semana'] == horario_nuevo['dia'] and
                self.hay_solapamiento(
                    self.parse_time(clase['hora_inicio']),
                    self.parse_time(clase['hora_fin']),
                    horario_nuevo['hora_inicio'],
                    horario_nuevo['hora_fin']
                )):
                return True
        return False

    def salon_ocupado(self, salon_id, horario_nuevo, horario_generado):
        for clase in horario_generado:
            if (clase['salon_id'] == salon_id and
                clase['dia_semana'] == horario_nuevo['dia'] and
                self.hay_solapamiento(
                    clase['hora_inicio'], clase['hora_fin'],
                    horario_nuevo['hora_inicio'], horario_nuevo['hora_fin']
                )):
                return True
        return False

def main():
    st.title('Generador Optimizado de Horarios de las clases UTS')

    optimizer = ScheduleOptimizer()

    if optimizer.load_model():
        st.success("Modelo cargado exitosamente")

    # Sidebar para parámetros
    st.sidebar.header('Parámetros de Optimización')

    # Añadir sección para parámetros adaptativos
    st.sidebar.subheader('Parámetros Adaptativos')
    adaptive_params = {
        'slot_duration': st.sidebar.slider(
            'Duración del slot (minutos)',
            min_value=30,
            max_value=120,
            value=45,
            step=15,
            help='Duración de cada bloque de tiempo'
        ),
        'min_alumnos': st.sidebar.number_input(
            'Mínimo de alumnos por clase',
            min_value=1,
            value=10,
            help='Número mínimo de alumnos para abrir una clase'
        ),
        'max_carga_profesor': st.sidebar.number_input(
            'Máxima carga por profesor',
            min_value=1,
            value=20,
            help='Número máximo de horas que puede dar un profesor'
        ),
        'optimization_level': st.sidebar.select_slider(
            'Nivel de optimización',
            options=['Bajo', 'Medio', 'Alto'],
            value='Medio',
            help='Nivel de agresividad en la optimización'
        ),
        'conflict_tolerance': st.sidebar.slider(
            'Tolerancia a conflictos',
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            help='Nivel de tolerancia a conflictos en el horario'
        ),
        'enable_pattern_detection': st.sidebar.checkbox(
            'Habilitar detección de patrones',
            value=True,
            help='Permite detectar y aprender de patrones en los horarios'
        ),
        'auto_correction': st.sidebar.checkbox(
            'Habilitar auto-corrección',
            value=True,
            help='Permite que el sistema corrija automáticamente conflictos'
        ),
        'learning_rate': st.sidebar.slider(
            'Tasa de aprendizaje',
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            help='Velocidad de adaptación del modelo'
        )
    }

    st.sidebar.subheader('Configuración del Modelo')
    model_params = {
            'model_type': st.sidebar.selectbox(
            'Tipo de Modelo',
        ['knn', 'random_forest'],
        help='Seleccione el algoritmo de aprendizaje automático'
    )
}
            # Parámetros específicos según el tipo de modelo seleccionado
    if model_params['model_type'] == 'knn':
            model_params.update({
                'n_neighbors': st.sidebar.slider(
                    'Número de vecinos (K)',
                    min_value=1,
                    max_value=20,
                    value=5,
                    help='Número de vecinos a considerar en KNN'
                ),
                'weights': st.sidebar.selectbox(
                    'Ponderación',
                    ['uniform', 'distance'],
                    help='Método de ponderación de los vecinos'
                ),
                'metric': st.sidebar.selectbox(
                    'Métrica de distancia',
                    ['euclidean', 'manhattan'],
                    help='Métrica para calcular la distancia entre puntos'
                )
            })
    else:  # Random Forest
            model_params.update({
                'n_estimators': st.sidebar.slider(
                    'Número de árboles',
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=50,
                    help='Número de árboles en el bosque'
                ),
                'max_depth': st.sidebar.slider(
                    'Profundidad máxima',
                    min_value=1,
                    max_value=50,
                    value=10,
                    help='Profundidad máxima de los árboles'
                ),
                'min_samples_split': st.sidebar.slider(
                    'Muestras mínimas para división',
                    min_value=2,
                    max_value=10,
                    value=2,
                    help='Número mínimo de muestras requeridas para dividir un nodo'
                ),
                'min_samples_leaf': st.sidebar.slider(
                    'Muestras mínimas en hojas',
                    min_value=1,
                    max_value=10,
                    value=1,
                    help='Número mínimo de muestras requeridas en un nodo hoja'
                )
            })
    # Parámetros generales de entrenamiento
    st.sidebar.subheader('Parámetros de Entrenamiento')
    model_params.update({
    'test_size': st.sidebar.slider(
        'Tamaño del conjunto de prueba',
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        help='Proporción de datos para prueba'
    ),
    'random_state': st.sidebar.number_input(
        'Semilla aleatoria',
        min_value=0,
        value=42,
        help='Semilla para reproducibilidad'
    ),
    'cv_folds': st.sidebar.slider(
        'Número de folds para validación cruzada',
        min_value=2,
        max_value=10,
        value=5,
        help='Número de particiones para validación cruzada'
    )
})

    # Parámetros de optimización
    st.sidebar.subheader('Restricciones y Límites')
    optimization_params = {
        'min_alumnos': st.sidebar.number_input(
            'Mínimo de alumnos por clase',
            min_value=1,
            value=10,
            help='Número mínimo de alumnos para abrir una clase',
            key='min_alumnos'  # Añade esta línea
        ),
        'max_carga_profesor': st.sidebar.number_input(
            'Máxima carga por profesor',
            min_value=1,
            value=20,
            help='Número máximo de horas que puede dar un profesor',
            key='max_carga_profesor'  # Añade esta línea
        ),
        'min_experiencia': st.sidebar.number_input(
            'Experiencia mínima requerida',
            min_value=0,
            value=1,
            help='Años mínimos de experiencia requeridos',
            key='min_experiencia'  # Añade esta línea
        ),
        'min_calificacion': st.sidebar.number_input(
            'Calificación mínima del profesor',
            min_value=1,
            max_value=5,
            value=3,
            help='Calificación mínima aceptable del profesor',
            key='min_calificacion'  # Añade esta línea
        )
    }
    # Cargar datos
    with st.spinner('Cargando datos...'):
        data = {
            'profesores': optimizer.get_data('profesores'),
            'materias': optimizer.get_data('materias'),
            'salones': optimizer.get_data('salones'),
            'horarios_disponibles': optimizer.get_data('horarios_disponibles'),
            'profesor_materia': optimizer.get_data('profesor_materia')
        }

    if all(data.values()):
        st.success("✅ Datos cargados correctamente")
        
        # Convertir a DataFrames de items
        dfs = {k: pd.DataFrame(v) for k, v in data.items()}
        
        # Mostrar resumen de datos
        with st.expander("📊 Resumen de Datos"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Profesores", len(dfs['profesores']))
            with col2:
                st.metric("Materias", len(dfs['materias']))
            with col3:
                st.metric("Salones", len(dfs['salones']))

        # Preparar características y entrenar modelo
        if st.button('🎯 Entrenar Modelo'):
            with st.spinner('Preparando datos y entrenando modelo...'):
                # Preparar características
                X, y, conflicts = optimizer.prepare_features(
                    dfs['profesores'],
                    dfs['materias'],
                    dfs['salones'],
                    dfs['horarios_disponibles'],
                    dfs['profesor_materia']
                )
                
                results = optimizer.train_model(X, y, model_params)
                st.success('✅ Modelo entrenado y guardado exitosamente')
                
                # Mostrar métricas del modelo
                st.subheader('📈 Métricas del Modelo')
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Mejores parámetros:", results['best_params'])
                    st.write("Matriz de confusión:")
                    st.write(pd.DataFrame(
                        results['confusion_matrix'],
                        columns=['Pred 0', 'Pred 1'],
                        index=['Real 0', 'Real 1']
                    ))
                
                with col2:
                    report_df = pd.DataFrame(results['classification_report']).transpose()
                    st.write("Reporte de clasificación:")
                    st.write(report_df)
                
                # Mostrar importancia de características si es Random Forest, importante
                if 'feature_importance' in results:
                    st.subheader('🎯 Importancia de Características')
                    fig = px.bar(
                        x=list(results['feature_importance'].keys()),
                        y=list(results['feature_importance'].values()),
                        title='Importancia de Características'
                    )
                    st.plotly_chart(fig)
                

        # Generar horario
        if st.button('📅 Generar Horario Optimizado'):
            with st.spinner('Generando horario con optimización adaptativa...'):
                resultado = optimizer.generate_schedule(
                            dfs['profesores'],
                            dfs['materias'],
                            dfs['salones'],
                            dfs['horarios_disponibles'],
                            dfs['profesor_materia'],
                            {**optimization_params, **adaptive_params}  # Incluye adaptive_params aquí
                        )


            if resultado["status"] in ["OPTIMAL", "FEASIBLE"]:
                st.success(f'✅ Horario generado ({resultado["status"]})')

                    # Convertir horario a DataFrame para mejor visualización
                df_horario = pd.DataFrame(resultado["horario_generado"])

                    # posting v:
                for _, clase in df_horario.iterrows():
                    clase_data = {
                        'grupo': clase['grupo'],
                        'dia_semana': clase['dia_semana'],
                        'hora_inicio': str(clase['hora_inicio']),
                        'hora_fin': str(clase['hora_fin']),
                        'alumnos': int(clase['alumnos']),
                        'materia_id': int(clase['materia_id']),
                        'salon_id': int(clase['salon_id']),
                        'profesor_id': int(clase['profesor_id'])
                }
                        # Para depuración
                    try:
                        response = requests.post(
                            f"{BASE_URL}/clases",
                            json=clase_data,
                            headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                        )
                        response.raise_for_status()  # Lanza una excepción si hay error en la respuesta
                        st.success("✅ Horario enviado exitosamente a la API")
                    except requests.RequestException as e:
                        st.error(f"❌ Error al enviar el horario a la API: {str(e)}")
                        st.write("Detalles del error:", response.text if 'response' in locals() else "No hay respuesta")
                    
    
            # Aquí puedes agregar el código para enviar el horario a la API
            #try:
                #response = requests.post(f"{BASE_URL}/clases", json=df_horario.to_dict(orient='records'))
                #response.raise_for_status()  # Lanza un error si la respuesta fue un error
                #st.success("✅ Horario enviado a la API exitosamente")
            #except requests.RequestException as e:
                #st.error(f"❌ Error al enviar el horario a la API: {str(e)}")
                
                # Añadir nombres de profesores y materias
                df_horario = df_horario.merge(
                    dfs['profesores'][['id', 'nombre']],
                    left_on='profesor_id',
                    right_on='id',
                    suffixes=('', '_profesor')
                )
                df_horario = df_horario.merge(
                    dfs['materias'][['id', 'nombre']],
                    left_on='materia_id',
                    right_on='id',
                    suffixes=('', '_materia')
                )
                
                # Visualización del horario
                st.subheader('📊 Horario Generado')
                
                # Crear vista por día
                dias = df_horario['dia_semana'].unique()
                for dia in dias:
                    with st.expander(f"Horario {dia}"):
                        df_dia = df_horario[df_horario['dia_semana'] == dia].sort_values('hora_inicio')
                        st.write(df_dia[['grupo', 'hora_inicio', 'hora_fin', 'nombre_materia', 'nombre', 'alumnos']])
                
                # Mostrar estadísticas
                st.subheader('📈 Estadísticas del Horario')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de clases", len(df_horario))
                with col2:
                    st.metric("Profesores asignados", df_horario['profesor_id'].nunique())
                with col3:
                    st.metric("Materias programadas", df_horario['materia_id'].nunique())
                
                # Mostrar advertencias si las hay
                if resultado["warnings"]:
                    st.warning("⚠️ Advertencias:")
                    for warning in resultado["warnings"]:
                        st.write(warning)
                
                # Opción para descargar el horario
                csv = df_horario.to_csv(index=False)
                st.download_button(
                    "⬇️ Descargar Horario (CSV)",
                    csv,
                    "horario.csv",
                    "text/csv",
                    key='download-csv'
                )
            else:
                st.error('❌ No fue posible generar el horario')
                if resultado["errors"]:
                    for error in resultado["errors"]:
                        st.write(error)
    else:
        st.error('❌ No se pudieron cargar todos los datos necesarios. Por favor, verifica la conexión con la API.')

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import RandomizedSearchCV

class AdaptiveScheduleOptimizer(ScheduleOptimizer):
    def __init__(self):
        super().__init__()
        self.performance_history = []
        self.pattern_database = {}
        self.slot_duration = 45  # duración en minutos de cada slot
        self.model_params = {
            'test_size': 0.2,
            # otros parámetros...
        }
        

    def split_time_block(self, hora_inicio, hora_fin, duration_minutes=45, flexibility_minutes=15):
        """
        Divide un bloque de tiempo en slots con inicios flexibles
        
        Args:
            hora_inicio: Hora de inicio del bloque disponible
            hora_fin: Hora de fin del bloque disponible
            duration_minutes: Duración de cada clase
            flexibility_minutes: Minutos de flexibilidad para el inicio de clase
        """
        try:
            inicio = datetime.strptime(str(hora_inicio), '%H:%M:%S')
        except ValueError:
            try:
                inicio = datetime.strptime(str(hora_inicio), '%H:%M')
            except ValueError:
                return []

        try:
            fin = datetime.strptime(str(hora_fin), '%H:%M:%S')
        except ValueError:
            try:
                fin = datetime.strptime(str(hora_fin), '%H:%M')
            except ValueError:
                return []

        inicio = inicio.replace(year=2024, month=1, day=1)
        fin = fin.replace(year=2024, month=1, day=1)

        delta = fin - inicio
        total_minutes = delta.total_seconds() / 60
        
        # Calculamos cuántos slots posibles hay considerando la flexibilidad
        possible_starts = int((total_minutes - duration_minutes) // flexibility_minutes) + 1
        
        time_slots = []
        for i in range(possible_starts):
            # Calculamos un inicio flexible para cada slot
            flexible_inicio = inicio + timedelta(minutes=i * flexibility_minutes)
            flexible_fin = flexible_inicio + timedelta(minutes=duration_minutes)
            
            # Verificamos que el slot completo esté dentro del horario disponible
            if flexible_fin <= fin:
                time_slots.append((
                    flexible_inicio.strftime('%H:%M:%S'),
                    flexible_fin.strftime('%H:%M:%S')
                ))

                for i in range(possible_starts):
                    flexible_inicio = inicio + timedelta(minutes=i * flexibility_minutes)
                    flexible_fin = flexible_inicio + timedelta(minutes=duration_minutes)

                    if flexible_fin <= fin:
                        time_slots.append((
                            flexible_inicio.strftime('%H:%M:%S'),
                            flexible_fin.strftime('%H:%M:%S')
                        ))

        # Opcionalmente, podemos aleatorizar los slots disponibles
        # para evitar que siempre se elijan los mismos horarios
        random.shuffle(time_slots)
        
        return time_slots
    
    def select_best_slot(self, available_slots):
        """
        Selecciona el mejor slot de tiempo basado en la preferencia de horario medio.
        """
        if not available_slots:
            return None

        # Convertir los slots a minutos desde medianoche
        def time_to_minutes(time_str):
            h, m, s = map(int, time_str.split(':'))
            return h * 60 + m

        # Calcular el punto medio del rango de tiempo disponible
        start_times = [time_to_minutes(slot[0]) for slot in available_slots]
        end_times = [time_to_minutes(slot[1]) for slot in available_slots]
        mid_point = (min(start_times) + max(end_times)) // 2

        # Encontrar el slot más cercano al punto medio
        best_slot = min(available_slots, key=lambda slot: abs(time_to_minutes(slot[0]) - mid_point))

        return best_slot


    def process_available_schedules(self, df_horarios):
        new_slots = []

        for _, row in df_horarios.iterrows():
            # Obtener todos los slots posibles
            slots = self.split_time_block(
                row['hora_inicio'],
                row['hora_fin'],
                duration_minutes=90,  # duración de clase (dos bloques de 45 minutos)
                flexibility_minutes=15  # flexibilidad de inicio
            )

            # Seleccionar el mejor slot
            best_slot = self.select_best_slot(slots)

            if best_slot:
                new_slots.append({
                    'profesor_id': row['profesor_id'],
                    'dia': row['dia'],
                    'hora_inicio': best_slot[0],
                    'hora_fin': best_slot[1],
                    'probabilidad': 1.0  # Máxima probabilidad para el mejor slot
                })

        return pd.DataFrame(new_slots)

    def prepare_features_with_slots(self, df_profesores, df_materias, df_salones, df_horarios, df_profesor_materia):
        """
        Versión adaptativa de prepare_features que incluye procesamiento de slots
        """
        df_horarios_slots = self.process_available_schedules(df_horarios)
        features, labels, conflicts = [], [], []
        carga_profesor = df_profesor_materia.groupby('profesor_id').size().to_dict()

        for _, prof_mat in df_profesor_materia.iterrows():
            profesor = df_profesores[df_profesores['id'] == prof_mat['profesor_id']].iloc[0]
            materia = df_materias[df_materias['id'] == prof_mat['materia_id']].iloc[0]
            horarios_prof = df_horarios_slots[df_horarios_slots['profesor_id'] == prof_mat['profesor_id']]
            conflictos = self.calcular_conflictos(horarios_prof)

            for _, salon in df_salones.iterrows():
                if salon['capacidad_alumnos'] >= materia['alumnos']:
                    feature = [
                        prof_mat['experiencia'],
                        prof_mat['calificacion_alumno'],
                        materia['alumnos'],
                        materia['bloques'],
                        len(horarios_prof),
                        salon['capacidad_alumnos'],
                        conflictos,
                        carga_profesor.get(prof_mat['profesor_id'], 0)
                    ]
                    features.append(feature)
                    labels.append(1)
                    conflicts.append(conflictos)

        negative_examples = self.generate_negative_examples(
            df_profesores, df_materias, df_salones, df_horarios_slots, df_profesor_materia, len(features)
        )
        features.extend(negative_examples[0])
        labels.extend(negative_examples[1])
        conflicts.extend(negative_examples[2])

        # Almacenar para uso en adaptación
        self.last_features = np.array(features)
        self.last_labels = np.array(labels)
        self.last_conflicts = conflicts

        return self.last_features, self.last_labels, self.last_conflicts

    def generate_schedule(self, *args, **kwargs):
        """
        Versión mejorada de generate_schedule que utiliza slots de tiempo flexibles
        """
        df_horarios = kwargs.get('df_horarios', args[3])
        df_horarios_slots = self.process_available_schedules(df_horarios)
        
        # Ordenar slots por probabilidad para favorecer ciertos horarios
        df_horarios_slots = df_horarios_slots.sort_values(
            by='probabilidad', 
            ascending=False
        )
        
        kwargs['df_horarios'] = df_horarios_slots
        
        initial_schedule = super().generate_schedule(*args, **kwargs)
        
        if initial_schedule["status"] in ["OPTIMAL", "FEASIBLE"]:
            optimized_schedule = self.auto_correct_schedule(initial_schedule)
            self.feedback_loop(optimized_schedule)
            self.detect_patterns()
            return optimized_schedule

        return initial_schedule

    def calcular_conflictos(self, horarios_prof):
        """
        Calcula conflictos en los horarios de un profesor
        """
        # Implementa la lógica de detección de conflictos
        # Por ejemplo, superposición de horarios
        conflictos = 0
        horarios_ordenados = horarios_prof.sort_values(['dia', 'hora_inicio'])
        
        for i in range(len(horarios_ordenados) - 1):
            actual = horarios_ordenados.iloc[i]
            siguiente = horarios_ordenados.iloc[i + 1]
            
            if actual['dia'] == siguiente['dia']:
                if actual['hora_fin'] > siguiente['hora_inicio']:
                    conflictos += 1
                    
        return conflictos

    def update_model(self, new_data, actual_performance):
        X_new, y_new = self.prepare_features(new_data)
        X_combined = np.vstack([self.X_historical, X_new])
        y_combined = np.concatenate([self.y_historical, y_new])
        self.train_model(X_combined, y_combined)
        self.adjust_weights(actual_performance)

    def auto_optimize_parameters(self):
        param_space = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        auto_search = RandomizedSearchCV(
            self.best_model,
            param_space,
            n_iter=20,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        auto_search.fit(self.X_historical, self.y_historical)
        return auto_search.best_params_

    def feedback_loop(self, schedule_results):
        success_rate = self.evaluate_schedule_success(schedule_results)
        conflict_rate = self.analyze_conflicts(schedule_results)
        if success_rate < 0.8:
            new_params = self.auto_optimize_parameters()
            self.update_model_parameters(new_params)
        self.performance_history.append({
            'success_rate': success_rate,
            'conflict_rate': conflict_rate,
            'timestamp': datetime.now()
        })

    def detect_patterns(self):
        patterns = {
            'popular_times': self.analyze_popular_times(),
            'common_conflicts': self.identify_common_conflicts(),
            'successful_combinations': self.find_successful_combinations()
        }
        self.adjust_strategy(patterns)

    def auto_correct_schedule(self, schedule):
        conflicts = self.detect_conflicts(schedule)
        for conflict in conflicts:
            alternative_solution = self.find_alternative_solution(conflict)
            if alternative_solution:
                schedule = self.apply_solution(schedule, alternative_solution)
        return schedule

    def monitor_performance(self):
        metrics = {
            'satisfaction_rate': self.calculate_satisfaction(),
            'resource_utilization': self.analyze_resource_usage(),
            'scheduling_efficiency': self.measure_efficiency()
        }
        if metrics['satisfaction_rate'] < self.threshold:
            self.trigger_optimization()

    def generate_schedule(self, *args, **kwargs):
        initial_schedule = super().generate_schedule(*args, **kwargs)
        optimized_schedule = self.auto_correct_schedule(initial_schedule)
        self.feedback_loop(optimized_schedule)
        self.detect_patterns()
        return optimized_schedule

    def alert_system(self):
        def check_conditions():
            alerts = []
            if self.performance_metrics['success_rate'] < 0.7:
                alerts.append("Bajo rendimiento detectado")
            if self.resource_usage > 0.9:
                alerts.append("Alto uso de recursos")
            return alerts
        
        alerts = check_conditions()
        if alerts:
            self.trigger_automatic_adjustments(alerts)

    def distribute_schedule_for_max_capacity(self, subject_id, available_schedule, max_class_capacity):
        professors = self.get_professors_for_subject(subject_id)
        remaining_capacity = max_class_capacity
        schedule_distribution = []

        for time_slot in available_schedule:
            for professor in professors:
                if remaining_capacity <= 0:
                    break
                if self.check_professor_availability(professor, time_slot):
                    students_to_assign = min(professor.max_class_size, remaining_capacity)
                    remaining_capacity -= students_to_assign
                    schedule_distribution.append({
                        'professor_id': professor.id,
                        'subject_id': subject_id,
                        'time_slot': time_slot,
                        'students_assigned': students_to_assign
                    })
                    if remaining_capacity <= 0:
                        break
        return schedule_distribution

    def get_professors_for_subject(self, subject_id):
        # Implementa la lógica para obtener profesores para una materia
        pass

    def check_professor_availability(self, professor, time_slot):
        # Implementa la lógica para verificar la disponibilidad del profesor
        pass

#optimzacion de procesos
class AdaptiveScheduleOptimizer(ScheduleOptimizer):
    

    def distribute_schedule_for_max_capacity(self, subject_id, available_schedule, max_class_capacity):
        """
        Distributes available schedule slots for a given subject among professors to maximize class capacity.
        """
        # Fetch relevant professors and their availability for this subject
        professors = self.get_professors_for_subject(subject_id)
        remaining_capacity = max_class_capacity  # Initialize with max capacity for the classroom
        schedule_distribution = []  # List to hold schedule assignments

        for time_slot in available_schedule:
            # Iterate over professors and allocate slots as long as there's remaining capacity
            for professor in professors:
                if remaining_capacity <= 0:
                    break  # Stop when max capacity is met for this subject

                # Check if the professor is available for the current time slot
                if self.check_professor_availability(professor, time_slot):
                    # Assign students up to the professor's max class size or remaining capacity
                    students_to_assign = min(professor.max_class_size, remaining_capacity)
                    remaining_capacity -= students_to_assign
                    
                    # Append the distribution details for this time slot and professor
                    schedule_distribution.append({
                        'professor_id': professor.id,
                        'subject_id': subject_id,
                        'time_slot': time_slot,
                        'students_assigned': students_to_assign
                    })

                    # If there is no remaining capacity, break out of loop
                    if remaining_capacity <= 0:
                        break

        # Return the final distribution that maximizes class capacity per slot
        return schedule_distribution

    # Helper methods to fetch professors for a given subject and check availability
    def get_professors_for_subject(self, subject_id):
        """
        Mock method to retrieve a list of professors who teach the given subject.
        This would ideally pull from the profesor_materia relationship in the database.
        """
        # Example placeholder code
        return [professor for professor in self.professors if professor.subject_id == subject_id]

    def check_professor_availability(self, professor, time_slot):
        """
        Mock method to check if a professor is available for a given time slot.
        This would typically check against the horarios_disponibles table.
        """
        # Example placeholder code
        return True  # AYUDA

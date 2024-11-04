import streamlit as st
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import requests
from faker import Faker

# Inicializar Faker
fake = Faker()

# URL base para las solicitudes a la API
BASE_URL = "http://localhost:8000/api"

# Función para obtener los datos desde la API
def get_data(endpoint):
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error al obtener datos de {endpoint}: {str(e)}")
        return None


min_alumnos = 15
# Función para enviar los datos a la API
def post_data(endpoint, data):
    try:
        response = requests.post(f"{BASE_URL}/{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error al enviar datos a {endpoint}: {str(e)}")
        return None

# Función para generar un acrónimo único para el grupo
def generar_acronimo():
    return fake.unique.bothify(text='??##', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Función para preprocesar los datos
def preprocesar_datos(profesores, materias, salones, horarios_disponibles, profesor_materia):
    df_profesores = pd.DataFrame(profesores)
    df_materias = pd.DataFrame(materias)
    df_salones = pd.DataFrame(salones)
    df_horarios_disponibles = pd.DataFrame(horarios_disponibles)
    df_profesor_materia = pd.DataFrame(profesor_materia)
    
    return df_profesores, df_materias, df_salones, df_horarios_disponibles, df_profesor_materia

# Función para generar el horario y hacer el POST a la API
def generar_horario(profesores, materias, salones, horarios_disponibles, profesor_materia):
    df_profesores, df_materias, df_salones, df_horarios_disponibles, df_profesor_materia = preprocesar_datos(profesores, materias, salones, horarios_disponibles, profesor_materia)
    
    st.write("Datos preprocesados:")
    st.write(f"Profesores: {len(df_profesores)}")
    st.write(f"Materias: {len(df_materias)}")
    st.write(f"Salones: {len(df_salones)}")
    st.write(f"Horarios disponibles: {len(df_horarios_disponibles)}")
    st.write(f"Relaciones profesor-materia: {len(df_profesor_materia)}")

     # Add these debug prints
    st.write("df_materias index:", df_materias.index.tolist())
    st.write("df_profesor_materia materia_id values:", df_profesor_materia['materia_id'].unique().tolist())
    
    model = cp_model.CpModel()
    
    # Variables
    clases = {}
    for i, horario in df_horarios_disponibles.iterrows():
        for j, salon in df_salones.iterrows():
            for k, prof_mat in df_profesor_materia.iterrows():
                clave = (i, j, k)
                clases[clave] = model.NewBoolVar(f'clase_h{i}_s{j}_pm{k}')
    
    st.write(f"Variables creadas: {len(clases)}")
    # Restricciones para la generacion de la clase
    restricciones_aplicadas = 0

    # 1. Un profesor no puede dar más de una clase al mismo tiempo
    for i, horario in df_horarios_disponibles.iterrows():
        for profesor_id in df_profesores['id']:
            model.Add(sum(clases[(i, j, k)] 
                          for j in df_salones.index 
                          for k in df_profesor_materia[df_profesor_materia['profesor_id'] == profesor_id].index) <= 1)
            restricciones_aplicadas += 1

    # 2. Un salón no puede tener más de una clase al mismo tiempo
    for i, horario in df_horarios_disponibles.iterrows():
        for j in df_salones.index:
            model.Add(sum(clases[(i, j, k)] for k in df_profesor_materia.index) <= 1)
            restricciones_aplicadas += 1

    # 3. Respetar la disponibilidad de los profesores
    for i, horario in df_horarios_disponibles.iterrows():
        for k, prof_mat in df_profesor_materia.iterrows():
            if prof_mat['profesor_id'] != horario['profesor_id']:
                for j in df_salones.index:
                    model.Add(clases[(i, j, k)] == 0)
                    restricciones_aplicadas += 1
    
   # 4. No exceder la capacidad del salón (modified)
    for j, salon in df_salones.iterrows():
        for i in df_horarios_disponibles.index:
            constraint_terms = []
            for k, prof_mat in df_profesor_materia.iterrows():
                materia_id = prof_mat['materia_id']
                if materia_id in df_materias.index:
                    alumnos = df_materias.loc[materia_id, 'alumnos']
                    constraint_terms.append(clases[(i, j, k)] * alumnos)
                else:
                    st.warning(f"Materia ID {materia_id} not found in df_materias")
            if constraint_terms:
                model.Add(sum(constraint_terms) <= salon['capacidad_alumnos'])
                restricciones_aplicadas += 1
            else:
                st.warning(f"No valid terms for constraint in salon {j}")

    # 5. Priorizar profesores con mayor experiencia y calificación
    for k, prof_mat in df_profesor_materia.iterrows():
        score = prof_mat['experiencia'] + prof_mat['calificacion_alumno']
        for i in df_horarios_disponibles.index:
            for j in df_salones.index:
                model.Add(clases[(i, j, k)] * score >= 0)  # Esto aumentará el valor de la función objetivo para profesores con mayor puntaje
    
    # 6. Función objetivo: maximizar clases asignadas y puntaje de profesores
    model.Maximize(sum(clases.values()) + 
                   sum(clases[(i, j, k)] * (df_profesor_materia.loc[k, 'experiencia'] + df_profesor_materia.loc[k, 'calificacion_alumno'])
                       for i in df_horarios_disponibles.index
                       for j in df_salones.index
                       for k in df_profesor_materia.index))
    
    # 7. Nueva restricción: No crear clase si no hay suficientes alumnos
    for i in df_horarios_disponibles.index:
        for j in df_salones.index:
            for k, prof_mat in df_profesor_materia.iterrows():
                materia_id = prof_mat['materia_id']
                alumnos = df_materias.loc[materia_id, 'alumnos']
                model.Add(clases[(i, j, k)] == 0).OnlyEnforceIf(model.NewBoolVar(f'not_enough_students_{i}_{j}_{k}'))
                model.Add(alumnos >= min_alumnos).OnlyEnforceIf(clases[(i, j, k)])
    
    # Resolver el modelo
    solver = cp_model.CpSolver()
    st.write("Resolviendo el modelo...")
    status = solver.Solve(model)
    st.write(f"Se aplicaron {restricciones_aplicadas} restricciones.")
    
    st.write(f"Estado de la solución: {solver.StatusName(status)}")
    result = {
        "status": solver.StatusName(status),
        "horario_generado": [],
        "warnings": [],
        "errors": []
    }

    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for i, horario in df_horarios_disponibles.iterrows():
            for j, salon in df_salones.iterrows():
                for k, prof_mat in df_profesor_materia.iterrows():
                    if solver.BooleanValue(clases[(i, j, k)]):
                        try:
                            materia = df_materias.loc[df_materias['id'] == prof_mat['materia_id']].iloc[0]
                            profesor = df_profesores.loc[df_profesores['id'] == prof_mat['profesor_id']].iloc[0]
                            
                            if materia['alumnos'] >= min_alumnos:
                                clase_data = {
                                    'grupo': generar_acronimo(),
                                    'dia_semana': horario['dia'],
                                    'hora_inicio': str(horario['hora_inicio']),
                                    'hora_fin': str(horario['hora_fin']),
                                    'alumnos': int(materia['alumnos']),
                                    'materia_id': int(materia['id']),
                                    'salon_id': int(salon['id']),
                                    'profesor_id': int(profesor['id'])
                                }
                                result["horario_generado"].append(clase_data)
                                
                                # Enviar la clase generada a la API
                                post_data('clases', clase_data)
                            else:
                                result["son muy pocos alumnos, no se sube"].append(f"La materia {materia['nombre']} no tiene suficientes alumnos ({materia['alumnos']}) para crear una clase.")
                        except IndexError:
                            result["errors"].append(f"Error al acceder a los datos de materia o profesor para la combinación: materia_id={prof_mat['materia_id']}, profesor_id={prof_mat['profesor_id']}")
                        except Exception as e:
                            result["errors"].append(f"Error inesperado al procesar una clase: {str(e)}")
        
        if not result["horario_generado"]:
            result["warnings"].append("No se pudo generar ninguna clase que cumpla con todas las restricciones.")
    else:
        result["errors"].append("No se pudo encontrar una solucion")
    
    return result


# Función auxiliar para preprocesar los datos
def preprocesar_datos(profesores, materias, salones, horarios_disponibles, profesor_materia):
    df_profesores = pd.DataFrame(profesores)
    df_materias = pd.DataFrame(materias)
    df_salones = pd.DataFrame(salones)
    df_horarios_disponibles = pd.DataFrame(horarios_disponibles)
    df_profesor_materia = pd.DataFrame(profesor_materia)
    
    return df_profesores, df_materias, df_salones, df_horarios_disponibles, df_profesor_materia


# Aplicación Streamlit
def main():
    st.title('Generador de Horarios UTS con restricciones')
    
    # Obtener los datos
    with st.spinner('Cargando datos...'):
        profesores = get_data('profesores')
        materias = get_data('materias')
        salones = get_data('salones')
        horarios_disponibles = get_data('horarios_disponibles')
        profesor_materia = get_data('profesor_materia')
    
    if all([profesores, materias, salones, horarios_disponibles, profesor_materia]):
        st.success("Todos los datos se cargaron correctamente")
        
        if st.button('Generar Horario para los profesores'):
            with st.spinner('Generando horario...'):
                horario_df = generar_horario(profesores, materias, salones, horarios_disponibles, profesor_materia)
            
            if horario_df is not None:
                st.success('Horario generado con éxito')
                st.write(horario_df)
            else:
                st.error('No fue posible generar el horario')
    else:
        st.error('No se pudieron cargar todos los datos necesarios. Por favor, verifica la conexión con la API.')

if __name__ == "__main__":
    main()
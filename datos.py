import requests
from faker import Faker
from datetime import datetime, timedelta

# Create a Faker instance
fake = Faker('es_ES')

# API URL
base_url = 'http://localhost:8000/api'

# Allowed schedules (blocks of two consecutive)
horarios_inicio = ["06:00", "06:45", "07:30", "08:15", "09:00", "09:45", "10:30", "11:15", 
                   "12:00", "12:45", "13:30", "14:15", "15:00", "15:45", "16:30", "17:15", "18:30", "20:15"]

# List of subjects
materias_por_semestre = {
    "Primer Semestre": [
        "Cálculo Diferencial", "Matemáticas Discretas", "Cultura Física", "Herramientas Digitales", 
        "Procesos de Lectura y Escritura", "Pensamiento Algorítmico", "Álgebra Superior"
    ],
    "Segundo Semestre": [
        "Sistemas Digitales", "Estructura de Computadores", "Mecánica", "Cálculo Integral", 
        "Fundamentos de POO", "Diseño de Bases de Datos", "Optativa I"
    ],
    "Tercer Semestre": [
        "Planeación de Sistemas Informáticos", "Programación Orientada a Objetos", "Sistemas Operativos", 
        "Programación de Dispositivos", "Epistemología", "Motores de Bases de Datos", "Electromagnetismo"
    ],
    "Cuarto Semestre": [
        "Redes", "Programación Web", "Laboratorio de Física", "Electiva de Profundización I", 
        "Optativa II", "Inglés I", "Estructura de Datos"
    ],
    "Quinto Semestre": [
        "Aplicaciones Móviles", "Administración de Servidores", "Programación en Java", 
        "Ética", "Electiva de Profundización II", "Metodología de la Investigación I", "Inglés II"
    ],
    "Sexto Semestre": [
        "Cálculo Multivariable", "Introducción a la Ingeniería", "Desarrollo de Aplicaciones Empresariales", 
        "Selección y Evaluación de Tecnología", "Seguridad de las Tecnologías de la Información", 
        "Nuevas Tecnologías de Desarrollo", "Electiva de Profundización III"
    ]
}

def create_professors(num):
    professor_ids = []
    for _ in range(num):
        data = {
            'tipo_cedula': fake.random_element(elements=('Cédula de Ciudadanía', 'Cédula de Extrangería', 'Targeta de Identidad', 'Registro Civil', 'Pasaporte')),
            'cedula': fake.unique.random_number(digits=10),
            'nombre': fake.name(),
            'tipo_contrato': fake.random_element(elements=('Cátedra', 'Planta', 'Tiempo Completo')),
            'estado': fake.random_element(elements=('Activo', 'Inactivo', 'En proceso')),
            'image_path': fake.image_url(),
        }
        response = requests.post(f'{base_url}/profesores', json=data)
        if response.status_code == 201:
            professor_ids.append(response.json()['id'])
        print(f'Professor created: {response.status_code}')
    return professor_ids

def create_subjects():
    subject_ids = []
    
    for semester, subjects in materias_por_semestre.items():
        for subject in subjects:
            # Crear una materia por asignatura con un mínimo de 50 alumnos
            alumnos = fake.random_int(min=50, max=100)
            
            data = {
                'codigo': fake.word(),
                'nombre': subject,
                'alumnos': alumnos,
                'bloques': fake.random_int(min=1, max=3),
            }
            response = requests.post(f'{base_url}/materias', json=data)
            if response.status_code == 201:
                subject_id = response.json()['id']
                subject_ids.append(subject_id)
                print(f'Subject created: {subject} with ID {subject_id} and {data["alumnos"]} students')
            else:
                print(f"Error creating subject: {response.status_code}")
    
    return subject_ids

def create_classrooms(num):
    classroom_ids = []
    for _ in range(num):
        data = {
            'codigo': fake.word(),
            'capacidad_alumnos': fake.random_int(min=20, max=100),
            'tipo': fake.random_element(elements=('Teórico', 'Laboratorio')),
        }
        response = requests.post(f'{base_url}/salones', json=data)
        if response.status_code == 201:
            classroom_ids.append(response.json()['id'])
        print(f'Classroom created: {response.status_code}')
    return classroom_ids

def create_available_schedules(professor_ids, max_blocks_per_professor):
    # Definir bloques de horarios según tipo de contrato
    horarios_por_contrato = {
        'Cátedra': [
            {'inicio': "18:30", 'fin': "22:00"}
        ],
        'Planta': [
            {'inicio': "06:00", 'fin': "12:00"}
        ],
        'Tiempo Completo': [
            {'inicio': "06:00", 'fin': "12:00"},
            {'inicio': "13:30", 'fin': "18:00"}
        ]
    }

    for professor_id in professor_ids:
        # Obtener el tipo de contrato del profesor
        response = requests.get(f'{base_url}/profesores/{professor_id}')
        if response.status_code == 200:
            tipo_contrato = response.json()['tipo_contrato']
            bloques_disponibles = horarios_por_contrato[tipo_contrato]
        else:
            continue

        # Crear disponibilidad para cada día de la semana
        dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
        
        for bloque in bloques_disponibles:
            for dia in dias_semana:
                data = {
                    'dia': dia,
                    'hora_inicio': bloque['inicio'],
                    'hora_fin': bloque['fin'],
                    'profesor_id': professor_id,
                }
                response = requests.post(f'{base_url}/horarios_disponibles', json=data)
                print(f'Available schedule created: {response.status_code} - {dia} {bloque["inicio"]} to {bloque["fin"]}')

def create_professor_subject(professor_ids, subject_ids):
    if not subject_ids:
        print("No subjects were created, aborting professor-subject assignment.")
        return
    
    subjects_per_professor = {professor_id: 0 for professor_id in professor_ids}
    assignments_created = []
    subjects_assigned = set()

    # Primero, asegurarse de que cada materia tenga al menos un profesor
    for subject_id in subject_ids:
        available_professors = [pid for pid, count in subjects_per_professor.items() if count < 4]
        
        if not available_professors:
            print("All professors have 4 subjects assigned. Cannot assign all subjects.")
            break
        
        professor_id = fake.random_element(available_professors)
        
        data = {
            'profesor_id': professor_id,
            'materia_id': subject_id,
            'experiencia': fake.random_int(min=1, max=10),
            'calificacion_alumno': fake.random_int(min=1, max=5),
        }
        response = requests.post(f'{base_url}/profesor_materia', json=data)
        
        if response.status_code == 201:
            subjects_per_professor[professor_id] += 1
            assignments_created.append((professor_id, subject_id))
            subjects_assigned.add(subject_id)
            print(f'Professor-Subject created: {response.status_code} for professor {professor_id} and subject {subject_id}')
        else:
            print(f'Error creating Professor-Subject: {response.status_code}')

    # Luego, asignar materias adicionales a profesores hasta que todos tengan 4 o no haya más profesores disponibles
    while True:
        available_professors = [pid for pid, count in subjects_per_professor.items() if count < 4]
        if not available_professors:
            break
        
        for professor_id in available_professors:
            subject_id = fake.random_element(subject_ids)
            if (professor_id, subject_id) not in assignments_created:
                data = {
                    'profesor_id': professor_id,
                    'materia_id': subject_id,
                    'experiencia': fake.random_int(min=1, max=10),
                    'calificacion_alumno': fake.random_int(min=1, max=5),
                }
                response = requests.post(f'{base_url}/profesor_materia', json=data)
                
                if response.status_code == 201:
                    subjects_per_professor[professor_id] += 1
                    assignments_created.append((professor_id, subject_id))
                    subjects_assigned.add(subject_id)
                    print(f'Additional Professor-Subject created: {response.status_code} for professor {professor_id} and subject {subject_id}')
                else:
                    print(f'Error creating additional Professor-Subject: {response.status_code}')

    print(f"Total professor-subject assignments created: {len(assignments_created)}")
    print(f"Total subjects assigned: {len(subjects_assigned)} out of {len(subject_ids)}")
    if len(subjects_assigned) < len(subject_ids):
        print("Warning: Not all subjects were assigned. You may need more professors.")

# Generate data
num_subjects = 10
num_professors = 45
num_classrooms = 45
num_schedules_per_professor = 2
num_professor_subject = num_professors * 4  # Cambiado para permitir 4 materias por profesor

# Call functions to generate data
subject_ids = create_subjects()
professor_ids = create_professors(num_professors)
classroom_ids = create_classrooms(num_classrooms)
create_available_schedules(professor_ids, num_schedules_per_professor)
create_professor_subject(professor_ids, subject_ids)
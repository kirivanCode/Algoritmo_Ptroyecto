# Algoritmo_Ptroyecto
Parte del proyecto de grado para la creacion de las clases de las UTS



Documentación del Sistema de Optimización de Horarios Adaptativos
Descripción General
Este sistema es una solución avanzada para la generación y optimización de las clases de la UTS. Utiliza técnicas de aprendizaje automático y optimización adaptativa para crear horarios eficientes que satisfacen múltiples restricciones y preferencias.

Componentes Principales
1. Interfaz de Usuario (Streamlit)
Muestra resúmenes de datos (profesores, materias, salones)
Permite entrenar modelos y generar horarios optimizados
Visualiza resultados y métricas de rendimiento
2. Preparación de Datos
Procesa información de profesores, materias, salones y horarios disponibles
Prepara características para el modelo de optimización
3. Modelo de Aprendizaje Automático
Entrena un modelo (posiblemente Random Forest) para predecir asignaciones de horarios óptimas
Incluye funciones para auto-optimización de parámetros
4. Generador de Horarios
Utiliza el modelo entrenado para crear horarios escolares
Implementa lógica para manejar restricciones y preferencias
5. Optimización Adaptativa
Ajusta parámetros y estrategias basándose en rendimiento histórico
Detecta patrones para mejorar futuras generaciones de horarios
6. Manejo de Slots de Tiempo Flexibles
Divide bloques de tiempo en slots más pequeños con flexibilidad
Optimiza la utilización del tiempo disponible
7. Sistema de Auto-corrección y Retroalimentación
Detecta y corrige conflictos en horarios generados
Aprende de resultados anteriores para mejorar futuros horarios
8. Visualización de Resultados
Muestra horarios generados en formato tabular y gráfico
Presenta métricas de rendimiento y estadísticas
9. Integración con API
Envía horarios generados a una API externa
Maneja errores y confirmaciones de envío
10. Distribución de Capacidad Máxima
Optimiza la asignación de profesores y horarios para maximizar la capacidad de las clases
Funciones Clave
split_time_block(): Divide bloques de tiempo en slots flexibles
select_best_slot(): Selecciona el mejor slot de tiempo basado en preferencias
process_available_schedules(): Procesa horarios disponibles en slots optimizados
prepare_features_with_slots(): Prepara características para el modelo incluyendo slots de tiempo
generate_schedule(): Genera horarios optimizados
auto_correct_schedule(): Corrige automáticamente conflictos en horarios
feedback_loop(): Implementa un ciclo de retroalimentación para mejora continua
distribute_schedule_for_max_capacity(): Distribuye horarios para maximizar la capacidad de las clases


Notas Técnicas
El sistema utiliza Streamlit para la interfaz de usuario
Se implementa un enfoque adaptativo para la optimización continua
La integración con API externa permite la persistencia de datos generados

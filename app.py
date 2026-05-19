import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.image("logo.png", width=100)
st.sidebar.image("Logo DMC.png") 
st.sidebar.caption("App version 1.0")

# 1. Configuracion principal de la pagina
st.set_page_config(page_title="EDA Bank Marketing", layout="wide")

#ADICIONAL UI/UX
estilo_propio = """
        <style>
        /* Ocultar marca de Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Personalización de las pestañas (Tabs) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f1f3f6;
            border-radius: 4px 4px 0 0;
            padding: 10px 20px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0A2E3B; /* Color oscuro profesional */
            color: white !important;
        }
        </style>
        """
st.markdown(estilo_propio, unsafe_allow_html=True)

# PROGRAMACION ORIENTADA A OBJETOS (POO)
class DataAnalyzer:
    def __init__(self):
        # Inicializamos el DataFrame como None
        self.df = None

# Carga del archivo CSV.
    def cargar_datos(self, ruta_archivo):
        try:
            # Se utiliza sep=';' dado el formato del dataset BankMarketing
            self.df = pd.read_csv(ruta_archivo, sep=';')
            return True, "Dataset cargado exitosamente."
        except FileNotFoundError:
            return False, "Error: El archivo no fue encontrado. Verifique que 'BankMarketing.csv' este en la misma ruta."
        except Exception as e:
            return False, f"Error inesperado al cargar los datos: {e}"

#DEF 1
    def obtener_info_general(self):
        """Genera un DataFrame con información de tipos y valores nulos."""
        info_df = pd.DataFrame({
            'Tipo de Dato': self.df.dtypes.astype(str),
            'Valores No Nulos': self.df.notnull().sum(),
            'Valores Nulos': self.df.isnull().sum()
        })
        return info_df

#DEF 2
    def clasificar_variables(self):
        """Clasifica las variables en numéricas y categóricas."""
        numericas = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categoricas = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        return numericas, categoricas

#DEF 3
    def obtener_estadisticas(self):
        """Retorna las estadísticas descriptivas de las variables numéricas."""
        return self.df.describe()

#DEF 4
    def plot_faltantes(self):
        """Genera un gráfico de barras con los valores nulos si los hay."""
        fig, ax = plt.subplots(figsize=(10, 4))
        faltantes = self.df.isnull().sum()
        
        # Filtramos solo las columnas que tienen nulos
        faltantes = faltantes[faltantes > 0] 
        
        if faltantes.empty:
            ax.text(0.5, 0.5, 'No se encontraron valores nulos (NaN) explícitos en el dataset.', 
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax.axis('off')
        else:
            sns.barplot(x=faltantes.index, y=faltantes.values, ax=ax, palette='viridis')
            ax.set_title('Valores Nulos por Variable')
            ax.set_ylabel('Cantidad')
            plt.xticks(rotation=45)
        return fig

#DEF 5
    def plot_distribucion_numerica(self, columna):
        """Genera un histograma con la curva de densidad para variables numéricas."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(self.df[columna], kde=True, ax=ax, color='skyblue')
        ax.set_title(f'Distribución de {columna}')
        ax.set_ylabel('Frecuencia')
        return fig

#DEF 6
    def plot_distribucion_categorica(self, columna):
        """Genera un gráfico de barras de conteo para variables categóricas."""
        fig, ax = plt.subplots(figsize=(8, 4))
        # Ordenamos los valores de mayor a menor frecuencia
        orden = self.df[columna].value_counts().index
        sns.countplot(data=self.df, x=columna, ax=ax, palette='Set2', order=orden)
        ax.set_title(f'Frecuencia de {columna}')
        ax.set_ylabel('Cantidad')
        plt.xticks(rotation=45)
        return fig
    
#DEF 7
    def plot_bivariado_num_cat(self, col_num, col_cat):
        """Genera un Boxplot para analizar una variable numérica segmentada por una categórica."""
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=self.df, x=col_cat, y=col_num, ax=ax, palette='Set3')
        ax.set_title(f'Distribución de {col_num} según {col_cat}')
        plt.xticks(rotation=45)
        return fig

#DEF 8
    def plot_bivariado_cat_cat(self, col_cat1, col_cat2):
        """Genera un gráfico de barras agrupadas para comparar dos variables categóricas."""
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=self.df, x=col_cat1, hue=col_cat2, ax=ax, palette='Paired')
        ax.set_title(f'Relación entre {col_cat1} y {col_cat2}')
        plt.xticks(rotation=45)
        return fig

#DEF 9
    def plot_dinamico(self, col_x, col_y):
        """Genera un gráfico de dispersión (Scatterplot) entre dos variables elegidas."""
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=self.df, x=col_x, y=col_y, ax=ax, alpha=0.6, color='coral')
        ax.set_title(f'Dispersión: {col_x} vs {col_y}')
        return fig

# MODULOS DE LA APLICACION
def modulo_home():
    st.title("Análisis Exploratorio de Datos: Bank Marketing")
    st.markdown("---")

    st.subheader("Objetivo del Análisis")
    st.write(
        "El presente proyecto constituye una herramienta interactiva diseñada para realizar un Análisis "
        "Exploratorio de Datos (EDA) sobre el dataset de campañas de marketing de una institución financiera. "
        "El objetivo principal es descubrir relaciones y comportamientos relevantes entre las variables, "
        "permitiendo entender los factores que influyen en la aceptación de los productos ofrecidos."
    )

    st.subheader("Datos del Autor")
    st.write("- **Nombre completo:** Kevin Yago Castrejón Sosa")
    st.write("- **Curso / Especialización:** Especialización en Python for Analytics")
    st.write("- **Año:** 2026")

    st.subheader("Contexto del Dataset")
    st.write(
        "El conjunto de datos 'BankMarketing.csv' documenta los resultados de una campaña de marketing directo. "
        "Recientemente, la efectividad de las campañas de la institución cayó del 12% al 8%, impactando el "
        "rendimiento comercial. A través del análisis de las variables demográficas, financieras y de contacto "
        "disponibles, se buscará extraer insights de alto valor."
    )

    st.subheader("Tecnologías Utilizadas")
    st.write("- **Lenguaje:** Python")
    st.write("- **Manipulación y Análisis numérico:** Pandas, NumPy")
    st.write("- **Visualización de Datos:** Matplotlib, Seaborn")
    st.write("- **Despliegue Web:** Streamlit")


# MODULO CARGA
def modulo_carga():
    st.title("Módulo 2: Carga y Exploración Inicial del Dataset")
    st.markdown("---")

    # Mensaje informativo de UX
    st.info("Para este caso de estudio, el dataset 'BankMarketing.csv' ya se encuentra preconfigurado en el repositorio interno del proyecto. Haz clic en el botón para extraer los datos e iniciar el análisis.")
    
    # Boton para disparar la accion de carga
    if st.button("Cargar Dataset Local (BankMarketing.csv)"):
        # Se llama al metodo del objeto instanciado en session_state
        exito, mensaje = st.session_state.analyzer.cargar_datos("BankMarketing.csv")
        
        if exito:
            st.success(mensaje)
        else:
            st.error(mensaje)

    # Validamos si el DataFrame ya esta cargado en la instancia
    df = st.session_state.analyzer.df
    if df is not None:
        st.subheader("1. Dimensiones del Dataset")
        # Obtencion de filas y columnas con .shape
        filas, columnas = df.shape
        st.write(f"- **Número de filas:** {filas}")
        st.write(f"- **Número de columnas:** {columnas}")

        st.subheader("2. Tipos de Datos por Variable")
        # Convertimos dtypes a string para una mejor visualizacion en Streamlit
        df_tipos = df.dtypes.astype(str).reset_index()
        df_tipos.columns = ["Variable", "Tipo de Dato"]
        st.dataframe(df_tipos, use_container_width=True)

        st.subheader("3. Vista Previa de los Datos")
        # Uso de columnas de Streamlit para mostrar head() y tail() lado a lado
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Primeras 5 filas (head):**")
            st.dataframe(df.head())
            
        with col2:
            st.write("**Últimas 5 filas (tail):**")
            st.dataframe(df.tail())

#TRES PRIMEROS ITEMS
def modulo_eda():
    st.title("Módulo 3: Análisis Exploratorio de Datos (EDA)")
    st.markdown("---")

    # Validación de seguridad: no ejecutar si no hay datos
    if st.session_state.analyzer.df is None:
        st.warning("Por favor, cargue el dataset en el Módulo 2 antes de continuar.")
        return

    df = st.session_state.analyzer.df

#--------
#Mostrar KPIs referencial con st.metric
    st.markdown("### KPIs principales")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    total_filas = df.shape[0]
    tasa_conversion = (df['y'] == 'yes').mean() * 100
    edad_promedio = df['age'].mean()
    
    col_kpi1.metric(label="Total de Clientes Analizados", value=f"{total_filas:,}")
    col_kpi2.metric(label="Tasa de Conversión Global", value=f"{tasa_conversion:.2f}%")
    col_kpi3.metric(label="Edad Promedio", value=f"{edad_promedio:.1f} años")
    st.markdown("---")
#----------

    # Creación de las 10 pestañas requeridas
    nombres_tabs = [
        "1. Info General", "2. Clasificación", "3. Estadísticas",
        "4. Faltantes", "5. Dist. Numéricas", "6. Dist. Categóricas",
        "7. Bivariado (Num vs Cat)", "8. Bivariado (Cat vs Cat)", 
        "9. Dinámico", "10. Hallazgos"
    ]
    tabs = st.tabs(nombres_tabs)

    # Ítem 1: Información general del dataset
    with tabs[0]:
        st.subheader("Ítem 1: Información General del Dataset")
        st.write("Resumen de tipos de datos y conteo de valores nulos:")
        info_df = st.session_state.analyzer.obtener_info_general()
        st.dataframe(info_df, use_container_width=True)

    # Ítem 2: Clasificación de variables
    with tabs[1]:
        st.subheader("Ítem 2: Clasificación de Variables")
        num, cat = st.session_state.analyzer.clasificar_variables()
        
        # Uso de columnas para mostrar listas en paralelo
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Variables Numéricas ({len(num)}):**")
            st.write(", ".join(num))
        with col2:
            st.write(f"**Variables Categóricas ({len(cat)}):**")
            st.write(", ".join(cat))

    # Ítem 3: Estadísticas descriptivas
    with tabs[2]:
        st.subheader("Ítem 3: Estadísticas Descriptivas")
        st.write("Medidas de tendencia central y dispersión para variables numéricas:")
        st.dataframe(st.session_state.analyzer.obtener_estadisticas(), use_container_width=True)
        st.write("**Interpretación básica:** La tabla superior nos permite observar las medias, medianas (50%) y desviaciones estándar (std) para comprender la distribución y la escala de cada variable numérica del dataset.")

    # Ítem 4: Valores Faltantes
    with tabs[3]:
        st.subheader("Ítem 4: Análisis de Valores Faltantes")
        st.write("Visualización de datos nulos en el dataset.")
        fig_faltantes = st.session_state.analyzer.plot_faltantes()
        st.pyplot(fig_faltantes)
        st.write("**Nota:** En algunos datasets como Bank Marketing, los faltantes vienen ocultos bajo la etiqueta 'unknown' en lugar de NaN.")

    # Ítem 5: Distribución de Variables Numéricas
    with tabs[4]:
        st.subheader("Ítem 5: Distribución de Variables Numéricas")
        num, _ = st.session_state.analyzer.clasificar_variables()
        
        # Widget para seleccionar la variable de forma dinámica
        var_num = st.selectbox("Seleccione una variable numérica para analizar:", num)
        if var_num:
            fig_num = st.session_state.analyzer.plot_distribucion_numerica(var_num)
            st.pyplot(fig_num)

    # Ítem 6: Distribución de Variables Categóricas
    with tabs[5]:
        st.subheader("Ítem 6: Distribución de Variables Categóricas")
        _, cat = st.session_state.analyzer.clasificar_variables()
        
        var_cat = st.selectbox("Seleccione una variable categórica para analizar:", cat)
        if var_cat:
            fig_cat = st.session_state.analyzer.plot_distribucion_categorica(var_cat)
            st.pyplot(fig_cat)

    # Ítem 7: Análisis Bivariado (Numérico vs Categórico)
    with tabs[6]:
        st.subheader("Ítem 7: Bivariado (Numérico vs Categórico)")
        st.write("Analiza la distribución de una variable numérica segmentada por grupos categóricos.")
        
        col_num_sel = st.selectbox("Seleccione la variable numérica:", num, key="biv_num")
        col_cat_sel = st.selectbox("Seleccione la variable categórica:", cat, key="biv_cat")
        
        if col_num_sel and col_cat_sel:
            fig_biv1 = st.session_state.analyzer.plot_bivariado_num_cat(col_num_sel, col_cat_sel)
            st.pyplot(fig_biv1)

    # Ítem 8: Análisis Bivariado (Categórico vs Categórico)
    with tabs[7]:
        st.subheader("Ítem 8: Bivariado (Categórico vs Categórico)")
        st.write("Analiza cómo se distribuyen las frecuencias entre dos variables categóricas.")
        
        col_cat1_sel = st.selectbox("Variable Principal (Eje X):", cat, key="biv_cat1")
        col_cat2_sel = st.selectbox("Variable Secundaria (Agrupación/Color):", cat, key="biv_cat2")
        
        if col_cat1_sel and col_cat2_sel:
            # Validacion para evitar cruzar la misma variable
            if col_cat1_sel == col_cat2_sel:
                st.warning("Seleccione variables categóricas diferentes para un mejor análisis.")
            else:
                fig_biv2 = st.session_state.analyzer.plot_bivariado_cat_cat(col_cat1_sel, col_cat2_sel)
                st.pyplot(fig_biv2)

    # Ítem 9: Gráfico Dinámico
    with tabs[8]:
        st.subheader("Ítem 9: Análisis Dinámico (Dispersión)")
        st.write("Explora la posible correlación entre dos variables numéricas.")
        
        col_x = st.selectbox("Variable Eje X:", num, key="din_x")
        col_y = st.selectbox("Variable Eje Y:", num, key="din_y")
        
        if col_x and col_y:
            fig_din = st.session_state.analyzer.plot_dinamico(col_x, col_y)
            st.pyplot(fig_din)

    # Ítem 10: Hallazgos principales
    with tabs[9]:
        st.subheader("Ítem 10: Hallazgos y Patrones Observados")
        st.markdown(
            """
            En base a la exploración inicial de los datos (EDA), podemos concluir lo siguiente:
            
            - **Desequilibrio de clases:** La variable objetivo ('y', que indica si aceptó o no la campaña) suele presentar una gran cantidad de 'no' frente a los 'yes'.
            - **Valores ocultos:** Existen variables categóricas (como 'job' o 'education') que contienen el valor 'unknown', lo cual debe ser tratado en la etapa de limpieza.
            - **Variables económicas:** Atributos como 'euribor3m' o 'emp.var.rate' muestran correlaciones interesantes que influyen en la decisión financiera del cliente.
            - **Distribución de edad:** Se pueden observar picos de aceptación en clientes muy jóvenes o en edad de jubilación, según los gráficos bivariados.
            """
        )


def modulo_conclusiones():
    st.title("Módulo 4: Conclusiones y Recomendaciones")
    st.markdown("---")

    st.subheader("Resumen de Hallazgos Principales")
    st.write("A partir del Análisis Exploratorio de Datos (EDA) realizado, se destacan los siguientes puntos:")
    st.markdown(
        """
        * **Tasa de conversión baja:** Existe un claro desequilibrio en la variable objetivo ('y'), confirmando que la gran mayoría de los clientes contactados no adquieren el producto, lo que explica la caída de la efectividad al 8%.
        * **Impacto macroeconómico:** Variables como la tasa de variación del empleo ('emp.var.rate') y el Euribor ('euribor3m') muestran patrones que sugieren que el contexto económico del cliente influye fuertemente en su decisión.
        * **Calidad de la base de datos:** Se detectó una cantidad considerable de valores etiquetados como 'unknown' en variables categóricas clave (como educación y trabajo), lo que representa un área de mejora en la captura de datos.
        * **Duración del contacto:** La variable 'duration' suele tener una relación directa con el éxito de la campaña; sin embargo, no es un buen predictor previo, ya que solo se conoce una vez finalizada la llamada.
        """
    )

    st.markdown("---")

    st.subheader("Recomendaciones Accionables")
    st.write("Basados en los datos observados, se proponen las siguientes acciones para futuras campañas:")
    st.markdown(
        """
        1. **Segmentación inteligente:** En lugar de realizar campañas masivas, se sugiere enfocar los esfuerzos en los perfiles demográficos (edad y tipo de trabajo) que mostraron una mayor proporción de respuestas positivas en el análisis bivariado, optimizando así los recursos de los ejecutivos.
        2. **Mejora en la captura de datos:** Implementar validaciones en los sistemas del banco para evitar registros en blanco o 'unknown' al momento de actualizar la información del cliente, permitiendo perfilamientos más exactos en el futuro.
        3. **Monitoreo del contexto económico:** Alinear el lanzamiento de campañas agresivas de créditos con los momentos donde los indicadores macroeconómicos (como el índice de confianza del consumidor) muestren tendencias favorables, aumentando la probabilidad de aceptación.
        """
    )

# FUNCION PRINCIPAL Y ENRUTAMIENTO
def main():
    # Instanciamos la clase DataAnalyzer y la guardamos en session_state para que no se pierda al interactuar con la pagina
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    st.sidebar.title("Menú de Navegación")
    
    opciones = [
        "Módulo 1: Home",
        "Módulo 2: Carga del Dataset",
        "Módulo 3: Análisis EDA",
        "Módulo 4: Conclusiones"
    ]
    
    eleccion = st.sidebar.radio("Seleccione un módulo:", opciones)

    # Enrutamiento logico
    if eleccion == "Módulo 1: Home":
        modulo_home()
    elif eleccion == "Módulo 2: Carga del Dataset":
        modulo_carga()
    elif eleccion == "Módulo 3: Análisis EDA":
        modulo_eda()
    elif eleccion == "Módulo 4: Conclusiones":
        modulo_conclusiones()

if __name__ == "__main__":
    main()
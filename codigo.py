{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "335da2a9",
   "metadata": {},
   "source": [
    "# Imports, constants and functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af610a01",
   "metadata": {
    "notebookRunGroups": {
     "groupValue": "2"
    }
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import nrrd\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import sklearn as sk\n",
    "import seaborn as sns\n",
    "import radiomics as pr\n",
    "import SimpleITK as sitk\n",
    "import matplotlib.pyplot as plt\n",
    "from radiomics import featureextractor\n",
    "from sklearn.linear_model import lasso_path\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from statsmodels.stats.outliers_influence import variance_inflation_factor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4e107a63",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Constantes\n",
    "INPUT_PATH = '/Users/veramegias/Documents/Universidad/Cuarto/TFG/Segmentaciones'\n",
    "IMAGES_PATH = 'images'\n",
    "OUTPUT_PATH = 'outputs'\n",
    "RECOMPUTE = False\n",
    "\n",
    "columns_file_text = f'{OUTPUT_PATH}/columnas_df_features.txt'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a7c8c568",
   "metadata": {},
   "outputs": [],
   "source": [
    "def transform_dataframe(df):\n",
    "    \"\"\"\n",
    "    Transforma un DataFrame con celdas de diferentes clases a numeros o strings.\n",
    "    \n",
    "    Para columnas que contienen listas o tuplas:\n",
    "    - Si la lista o tupla tiene un único elemento, convierte el valor en un número o string según corresponda.\n",
    "    - Si la lista o tupla tiene múltiples elementos, expande la columna en varias columnas, \n",
    "      una por cada elemento de la lista o tupla. Las nuevas columnas se nombran usando el nombre \n",
    "      original seguido por un sufijo `_1`, `_2`, etc.\n",
    "\n",
    "    Para columnas que contienen diccionarios:\n",
    "    - Cada clave del diccionario se convierte en una nueva columna.\n",
    "    - Si un valor del diccionario es un array/lista o tupla:\n",
    "        - Si tiene un único elemento, se convierte en un valor único.\n",
    "        - Si tiene múltiples elementos, genera columnas adicionales con sufijos `_1`, `_2`, etc.\n",
    "    - Las nuevas columnas se nombran usando el nombre original seguido por `_{key}` y, si es necesario, \n",
    "      un sufijo adicional para los arrays o tuplas.\n",
    "    - Elimina la columna original una vez procesada.\n",
    "\n",
    "    Args:\n",
    "        df (pd.DataFrame): DataFrame original.\n",
    "    Returns:\n",
    "        df (pd.DataFrame): DataFrame transformado.\n",
    "    \"\"\"\n",
    "    # Crear una copia para no modificar el original\n",
    "    transformed_df = df.copy()\n",
    "\n",
    "    # Iterar sobre las columnas\n",
    "    for col in transformed_df.columns:\n",
    "        # Identificar las celdas que son listas, tuplas o arrays\n",
    "        if transformed_df[col].apply(lambda x: isinstance(x, (list, tuple))).any():\n",
    "            # Expandir los valores si hay listas/tuplas con más de un elemento\n",
    "            expanded = transformed_df[col].apply(lambda x: list(x) if isinstance(x, (list, tuple)) else [x])\n",
    "            \n",
    "            # Verificar la longitud máxima de las listas/tuplas\n",
    "            max_len = expanded.apply(len).max()\n",
    "            \n",
    "            if max_len > 1:\n",
    "                # Crear nuevas columnas para listas/tuplas con múltiples elementos\n",
    "                for i in range(max_len):\n",
    "                    transformed_df[f\"{col}_{i+1}\"] = expanded.apply(lambda x: x[i] if i < len(x) else None)\n",
    "                \n",
    "                # Eliminar la columna original\n",
    "                transformed_df.drop(columns=[col], inplace=True)\n",
    "            else:\n",
    "                # Convertir listas/tuplas con un único elemento en valores (número o string)\n",
    "                transformed_df[col] = expanded.apply(lambda x: x[0] if len(x) == 1 else x)\n",
    "        \n",
    "        # Identificar las celdas que son diccionarios\n",
    "        elif transformed_df[col].apply(lambda x: isinstance(x, dict)).any():\n",
    "            # Expandir las claves del diccionario en nuevas columnas\n",
    "            dict_expansion = transformed_df[col].apply(lambda x: x if isinstance(x, dict) else {})\n",
    "            keys = set(k for d in dict_expansion for k in d.keys())\n",
    "            \n",
    "            for key in keys:\n",
    "                # Extraer los valores de la clave específica\n",
    "                key_values = dict_expansion.apply(lambda x: x.get(key, None))\n",
    "                \n",
    "                # Si los valores son arrays, listas o tuplas, manejarlos como tal\n",
    "                if key_values.apply(lambda x: isinstance(x, (list, tuple))).any():\n",
    "                    # Expandir los arrays/tuplas en columnas adicionales\n",
    "                    expanded = key_values.apply(lambda x: list(x) if isinstance(x, (list, tuple)) else [x])\n",
    "                    max_len = expanded.apply(len).max()\n",
    "                    \n",
    "                    for i in range(max_len):\n",
    "                        transformed_df[f\"{col}_{key}_{i+1}\"] = expanded.apply(lambda x: x[i] if i < len(x) else None)\n",
    "                else:\n",
    "                    # Si no son listas/tuplas, mantener el valor tal cual\n",
    "                    transformed_df[f\"{col}_{key}\"] = key_values\n",
    "            \n",
    "            # Eliminar la columna original\n",
    "            transformed_df.drop(columns=[col], inplace=True)\n",
    "\n",
    "    return transformed_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0aad380b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def same_sizes(image1, image2):\n",
    "    \"\"\"\n",
    "    Comprueba que la imagen 1 y la imagen 1 tienen las mismas dimensiones.\n",
    "    Args:\n",
    "        image1 (SimpleITK.Image): Imagen 1.\n",
    "        image2 (SimpleITK.Image): Imagen 2.\n",
    "    Returns:\n",
    "        boolean: Si el tamaño coincide\n",
    "    \"\"\"\n",
    "    return image1.GetSize() == image2.GetSize()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02c08045",
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_features(image_file_path, mask_file_path):\n",
    "    \"\"\"\n",
    "    Extrae características radiómicas de una imagen y su máscara utilizando PyRadiomics.\n",
    "    Args:\n",
    "        image_file_path (str): Ruta al archivo NRRD que contiene la imagen a analizar.\n",
    "        mask_file_path (str): Ruta al archivo NRRD que contiene la máscara asociada a la imagen.\n",
    "    Returns:\n",
    "        features (dict): Características radiómicas extraídas.\n",
    "    Extra:\n",
    "        Comprueba que el tamaño de las imagenes sea compatible.\n",
    "    \"\"\"\n",
    "    # Cargar la imagen desde el archivo NRRD\n",
    "    image_data, _ = nrrd.read(image_file_path)\n",
    "    image = sitk.GetImageFromArray(image_data)\n",
    "    \n",
    "    # Cargar la máscara desde el archivo NRRD\n",
    "    mask_data, _ = nrrd.read(mask_file_path)\n",
    "    mask = sitk.GetImageFromArray(mask_data)\n",
    "    \n",
    "    # Crear un extractor de características de PyRadiomics\n",
    "    if same_sizes(image, mask):\n",
    "        extractor = featureextractor.RadiomicsFeatureExtractor()\n",
    "    else:\n",
    "        print(f'[ERROR] Sizes are not the same for {image_file_path} and {mask_file_path}')\n",
    "    \n",
    "    # Extraer las características radiómicas\n",
    "    features = extractor.execute(image, mask)\n",
    "    \n",
    "    return features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "115644fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "def convert_columns_to_numeric(df):\n",
    "    \"\"\"\n",
    "    Intenta convertir todas las columnas de un DataFrame a valores numéricos, \n",
    "    incluyendo la transformación de booleanos a numéricos.\n",
    "    \n",
    "    Args:\n",
    "        df (pd.DataFrame): DataFrame original.\n",
    "    Returns:\n",
    "        df (pd.DataFrame): DataFrame transformado.\n",
    "    \"\"\"\n",
    "    # Crear una copia del DataFrame para no modificar el original\n",
    "    numeric_df = df.copy()\n",
    "    \n",
    "    for col in numeric_df.columns:\n",
    "        try:\n",
    "            # Convertir booleanos a numéricos explícitamente\n",
    "            if numeric_df[col].dtype == 'bool':\n",
    "                numeric_df[col] = numeric_df[col].astype(int)\n",
    "            \n",
    "            # Intentar convertir la columna a valores numéricos\n",
    "            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='raise')\n",
    "        except Exception as e:\n",
    "            # Imprimir un mensaje de error y eliminar la columna si falla\n",
    "            print(f'[ERROR] al convertir la columna {col} a número. {e}')\n",
    "            numeric_df.drop(columns=[col], inplace=True)\n",
    "    \n",
    "    return numeric_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6d91a629",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_correlation_matrix(df):\n",
    "    \"\"\"\n",
    "    Genera una matriz de correlación con un mapa de calor.\n",
    "    \n",
    "    Args:\n",
    "        df (DataFrame): DataFrame con los datos.\n",
    "    \"\"\"\n",
    "    df = df.loc[:, (df != df.iloc[0]).any()]\n",
    "    correlation_matrix = df.corr()\n",
    "    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Parte superior (cambiar a np.tril para inferior)\n",
    "\n",
    "    print('Be aware that columns with constant values will not be plot.')\n",
    "    plt.figure(figsize=(10, 8))\n",
    "    correlation_matrix = df.corr()\n",
    "    sns.heatmap(correlation_matrix, cmap='coolwarm', mask=mask)\n",
    "    plt.title(\"Matriz de Correlación\")\n",
    "    plt.savefig(f'{IMAGES_PATH}/correlation_matrix.png')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "85595fb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_lasso_path(X, y):\n",
    "    \"\"\"\n",
    "    Genera un gráfico del camino de Lasso para analizar la importancia de las variables.\n",
    "    \n",
    "    Args:\n",
    "        X (array-like): Variables independientes.\n",
    "        y (array-like): Variable dependiente.\n",
    "    \"\"\"\n",
    "    alphas, coefs, _ = lasso_path(X, y)\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    for coef in coefs:\n",
    "        plt.plot(-np.log10(alphas), coef)\n",
    "    plt.xlabel(\"-Log10(Alpha)\")\n",
    "    plt.ylabel(\"Coeficientes\")\n",
    "    plt.title(\"Camino de Lasso (Lasso Path)\")\n",
    "    plt.savefig(f'{IMAGES_PATH}/lasso_path.png')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95abf6bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_variable_distribution(df):\n",
    "    \"\"\"\n",
    "    Genera gráficos de distribución para todas las columnas numéricas.\n",
    "    \n",
    "    Args:\n",
    "        df (DataFrame): DataFrame con los datos.\n",
    "    \"\"\"\n",
    "    numeric_columns = df.select_dtypes(include=['number']).columns\n",
    "    for column in numeric_columns:\n",
    "        plt.figure(figsize=(8, 4))\n",
    "        sns.histplot(df[column], kde=True, bins=30)\n",
    "        plt.title(f\"Distribución de {column}\")\n",
    "        plt.xlabel(column)\n",
    "        plt.ylabel(\"Frecuencia\")\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d163a02d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def detect_missing_values(df):\n",
    "    \"\"\"\n",
    "    Genera un reporte visual y tabular de valores faltantes.\n",
    "    \n",
    "    Args:\n",
    "        df (DataFrame): DataFrame con los datos.\n",
    "    \"\"\"\n",
    "    missing_values = df.isnull().sum()\n",
    "    missing_percentage = (missing_values / len(df)) * 100\n",
    "    missing_report = pd.DataFrame({\n",
    "        'Valores Faltantes': missing_values,\n",
    "        'Porcentaje (%)': missing_percentage\n",
    "    }).sort_values(by='Valores Faltantes', ascending=False)\n",
    "    print(missing_report)\n",
    "\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    sns.heatmap(df.isnull(), cbar=False, cmap=\"viridis\")\n",
    "    plt.title(\"Mapa de Valores Faltantes\")\n",
    "    plt.savefig(f'{IMAGES_PATH}/missing_values.png')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1bfa4fb7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_feature_importance(X, y, feature_names):\n",
    "    \"\"\"\n",
    "    Genera un gráfico de importancia de características usando un modelo de Random Forest.\n",
    "    \n",
    "    Args:\n",
    "        X (array-like): Variables independientes.\n",
    "        y (array-like): Variable dependiente.\n",
    "        feature_names (list): Nombres de las características.\n",
    "    \"\"\"\n",
    "    model = RandomForestRegressor(random_state=0)\n",
    "    model.fit(X, y)\n",
    "    importance = model.feature_importances_\n",
    "\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    plt.barh(feature_names, importance)\n",
    "    plt.xlabel(\"Importancia\")\n",
    "    plt.ylabel(\"Características\")\n",
    "    plt.title(\"Importancia de las Características\")\n",
    "    plt.savefig(f'{IMAGES_PATH}/feature_importance_RF.png')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cfb88b52",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_outliers(df):\n",
    "    \"\"\"\n",
    "    Genera boxplots para identificar outliers en las columnas numéricas.\n",
    "    \n",
    "    Args:\n",
    "        df (DataFrame): DataFrame con los datos.\n",
    "    \"\"\"\n",
    "    numeric_columns = df.select_dtypes(include=['number']).columns\n",
    "    for column in numeric_columns:\n",
    "        plt.figure(figsize=(8, 4))\n",
    "        sns.boxplot(x=df[column])\n",
    "        plt.title(f\"Outliers en {column}\")\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "93eacb53",
   "metadata": {},
   "outputs": [],
   "source": [
    "def target_correlation_analysis(df, target):\n",
    "    \"\"\"\n",
    "    Analiza la correlación entre las variables independientes y la variable objetivo.\n",
    "    \n",
    "    Args:\n",
    "        df (DataFrame): DataFrame con los datos.\n",
    "        target (str): Nombre de la columna objetivo.\n",
    "    \"\"\"\n",
    "    correlation = df.corr()[target].sort_values(ascending=False)\n",
    "    print(f'Correlación con la variable objetivo:{correlation}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7762b13e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def calculate_vif(X):\n",
    "    \"\"\"\n",
    "    Calcula el Factor de Inflación de la Varianza (VIF) para detectar multicolinealidad.\n",
    "    \n",
    "    Args:\n",
    "        X (DataFrame): Variables independientes.\n",
    "    \"\"\"\n",
    "    vif_data = pd.DataFrame()\n",
    "    vif_data[\"Variable\"] = X.columns\n",
    "    vif_data[\"VIF\"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]\n",
    "    print(vif_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a16a7bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "def filter_columns_by_file(df, file_path):\n",
    "    \"\"\"\n",
    "    Filtra las columnas del dataframe según un archivo con nombres de columnas\n",
    "    seguidos de \"OK\" o \"NO\". Las columnas con \"NO\" se eliminan.\n",
    "    \"\"\"\n",
    "    columns_to_keep = []\n",
    "\n",
    "    with open(file_path, \"r\") as f:\n",
    "        for line in f:\n",
    "            column_info = line.strip().split()\n",
    "            if len(column_info) == 2:  # Asegurarse de que hay un nombre y un estado\n",
    "                column_name, status = column_info\n",
    "                if status == \"OK\":\n",
    "                    columns_to_keep.append(column_name)\n",
    "\n",
    "    df_clean = df.copy()\n",
    "    return df_clean[columns_to_keep]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "75c089e8",
   "metadata": {},
   "source": [
    "# Load data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "31021179",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Para cambiar los nombres de los ficheros series, ejecutar en terminal en la carpeta de inputs:\n",
    "# for file in series*.nrrd; do mv \"$file\" \"${file/series/serie}\"; done"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "499c9572",
   "metadata": {
    "notebookRunGroups": {
     "groupValue": "1"
    }
   },
   "outputs": [],
   "source": [
    "df_features = pd.DataFrame()\n",
    "\n",
    "for num in range(3,42):\n",
    "    for extension in ['oc', 'ccr']:\n",
    "        print(f'Executing {num} for {extension}')\n",
    "        nrrd_path = os.path.join(INPUT_PATH, extension, str(num)+extension, 'Seg'+str(num)+extension, 'serie'+str(num)+extension+'.nrrd')\n",
    "        mask_path = os.path.join(INPUT_PATH, extension, str(num)+extension, 'Seg'+str(num)+extension, extension+str(num)+'.nrrd')        \n",
    "        if not os.path.exists(nrrd_path):\n",
    "            print(f'ERROR: No such file for {nrrd_path}')\n",
    "        if not os.path.exists(mask_path):\n",
    "            print(f'ERROR: No such file for {mask_path}')\n",
    "        features_image = extract_features(nrrd_path, mask_path)\n",
    "        features_image['Cancer'] = mask_path.startswith('ccr')\n",
    "        df_features = pd.concat([df_features, pd.DataFrame([features_image])], ignore_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b3dc3a13",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_features.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "61e924ae",
   "metadata": {
    "notebookRunGroups": {
     "groupValue": "1"
    }
   },
   "outputs": [],
   "source": [
    "df_features.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c4ce20b0",
   "metadata": {},
   "source": [
    "# Clean dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec746f3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(f'{columns_file_text}', \"w\") as f:\n",
    "    for column in df_features.columns:\n",
    "        f.write(column + \" OK \\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c090adb1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_features.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fb72ce4b",
   "metadata": {},
   "source": [
    "### Edita el documento de columnas para eliminarlas. Pon \"NO\" en las que quieras eliminar."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8c234cff",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_clean = filter_columns_by_file(df_features, columns_file_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e10f9eea",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_clean.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "456b0706",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_clean = transform_dataframe(df_features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "80f658c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_clean.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5d26bb0d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_clean.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b41da261",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_numeric = convert_columns_to_numeric(df_clean)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "854f722f",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_numeric.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "accae703",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_numeric.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5e753a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f'Antes teniamos {df_numeric.shape[0]} columnas')\n",
    "df_numeric = df_numeric.dropna(axis=1, how='all')\n",
    "print(f'Ahora tenemos {df_numeric.shape[0]} columnas')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "364b00a3",
   "metadata": {},
   "source": [
    "# Look dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dea050b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split labels and features\n",
    "X = df_numeric.drop(columns=['Cancer'])\n",
    "y = df_numeric['Cancer']\n",
    "features_names = X.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c84da2d",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_correlation_matrix(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bfd9111b",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_lasso_path(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "75e52220",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_variable_distribution(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5241dc27",
   "metadata": {},
   "outputs": [],
   "source": [
    "detect_missing_values(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5d735627",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_feature_importance(X, y, features_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "30a42d59",
   "metadata": {},
   "outputs": [],
   "source": [
    "target_correlation_analysis(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6f069fd1",
   "metadata": {},
   "outputs": [],
   "source": [
    "calculate_vif(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0d52e697",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aac12daa",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

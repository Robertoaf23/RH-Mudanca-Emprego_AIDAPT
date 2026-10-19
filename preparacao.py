"""Preparação dos dados partilhada pelas Fases 3, 4 e 6.

Uma só cópia das regras: o que a Fase 3 aplica no treino é exatamente o que a
Fase 6 aplica aos candidatos novos. Depois de alterar este ficheiro, reiniciar o
kernel dos notebooks abertos.
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline


def limpar_texto(serie):
    """Tira espaços no início/fim e espaços repetidos. Só mexe em texto: um número
    numa coluna de texto (ex.: 7 ao lado de "5 anos") fica como está."""
    return serie.map(lambda v: " ".join(v.split()) if isinstance(v, str) else v)


def converter_anos(serie, especiais):
    """Converte texto de anos para número.
    1) aplica o dicionário de casos especiais;
    2) nos restantes, extrai o primeiro número do texto."""
    serie = serie.astype(str)  # deixa passar números além de texto
    numero = pd.to_numeric(serie.str.extract(r"(\d+)")[0], errors="coerce")
    return serie.map(especiais).fillna(numero).astype(float)  # float porque há NaN


def aplicar_regras(data, regras, codigos_missing):
    """Códigos de missing disfarçados e valores fora do intervalo válido -> NaN."""
    data = data.copy()
    for col, (minimo, maximo) in regras.items():
        data.loc[data[col].isin(codigos_missing), col] = np.nan
        data.loc[(data[col] < minimo) | (data[col] > maximo), col] = np.nan
    return data


def criar_features(data):
    """Cria variáveis derivadas com regras fixas (determinísticas).
    Não aprende nada com os dados: pode ser aplicada da mesma forma a treino e teste."""
    data = data.copy()  # não alterar o DataFrame original

    # 1. Indicador: faltam dados sobre a empresa atual (H6)
    data["empresa_em_falta"] = (
        data["tipo_empresa"].isna() | data["dimensao_empresa"].isna()
    ).astype(int)

    # 2. Número de valores em falta na linha (antes da imputação!)
    data["n_nulos"] = data.isna().sum(axis=1)

    # 3. Transformação logarítmica para reduzir a assimetria das horas
    data["horas_formacao_log"] = np.log1p(data["horas_formacao"])

    # 4. Indicador: nunca mudou de emprego ("Nunca" foi convertido em 0)
    data["nunca_mudou_emprego"] = (data["anos_desde_ultimo_emprego"] == 0).astype(int)

    # 5. Escalões de experiência com limites fixos escolhidos por nós
    data["escalao_experiencia"] = pd.cut(
        data["anos_experiencia"],
        bins=[-1, 2, 5, 10, 20, 21],
        labels=["0-2", "3-5", "6-10", "11-20", ">20"]
    ).astype("object")  # texto normal (mais fácil para imputação/encoding)

    return data


def preparar(df_novos, prep_params, target):
    """Passos da Fase 3 que ficam fora do pipeline, pela mesma ordem, para dados
    no formato do ficheiro original (com cidade e anos em texto)."""
    data = df_novos.copy()
    if "id_candidato" in data.columns:
        data = data.set_index("id_candidato")
    data = data.drop(columns=[target] + prep_params["colunas_removidas_selecao"], errors="ignore")

    for col in data.select_dtypes(exclude="number").columns:
        data[col] = limpar_texto(data[col])

    data["anos_experiencia"] = converter_anos(data["anos_experiencia"], prep_params["conversao_anos_experiencia"])
    data["anos_desde_ultimo_emprego"] = converter_anos(data["anos_desde_ultimo_emprego"], prep_params["conversao_anos_desde_ultimo_emprego"])

    data = aplicar_regras(data, prep_params["regras_validade"], prep_params["codigos_missing"])
    data = criar_features(data)
    return data.drop(columns=prep_params["colunas_removidas_redundancia"], errors="ignore")


def criar_prep(config, scaling=True):
    """ColumnTransformer da Fase 3 (imputação + scaling + encoding), a partir das
    decisões guardadas em config_projeto.json. As árvores não precisam de scaling."""
    if config["ESTRATEGIA_NUM"] == "knn":
        imputer_num = KNNImputer(n_neighbors=5, add_indicator=config["ADD_INDICATOR"])
    else:
        imputer_num = SimpleImputer(strategy=config["ESTRATEGIA_NUM"], add_indicator=config["ADD_INDICATOR"])

    passos_num = [("imputer", imputer_num)]
    if scaling:
        passos_num.append(("scaler", getattr(preprocessing, config["SCALER"])()))
    if config["USAR_PCA"]:
        passos_num.append(("pca", PCA(n_components=0.95, random_state=config["RANDOM_STATE"])))

    # ordinais: "Em falta" + OrdinalEncoder com ordem explícita
    pipe_ord = Pipeline([
        ("imputer", SimpleImputer(strategy=config["ESTRATEGIA_CAT"], fill_value=config["FILL_VALUE_CAT"])),
        ("encoder", preprocessing.OrdinalEncoder(categories=[config["ORDENS"][c] for c in config["ORD_COLS"]],
                                                 handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    # nominais: "Em falta" + OneHotEncoder
    pipe_nom = Pipeline([
        ("imputer", SimpleImputer(strategy=config["ESTRATEGIA_CAT"], fill_value=config["FILL_VALUE_CAT"])),
        ("encoder", preprocessing.OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", Pipeline(passos_num), config["NUM_COLS"]),
        ("bin", "passthrough", config["BIN_COLS"]),  # 0/1 sem nulos: passam como estão
        ("ord", pipe_ord, config["ORD_COLS"]),
        ("nom", pipe_nom, config["NOM_COLS"]),
    ], remainder="drop", verbose_feature_names_out=False)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Este código realiza a importação de informações sobre exoplanetas utilizando a 
API da base de dados do MAST. Inicialmente, é feita uma pesquisa em uma base de 
dados pública para relacionar o ID do exoplaneta com a magnitude de sua estrela 
associada. Com essa informação em mãos, uma segunda pesquisa é realizada em uma 
base de dados privada, onde se obtêm dados adicionais que não estão disponíveis 
na primeira consulta. A partir da união dessas informações, é construída uma 
tabela contendo todos os dados necessários para a análise das curvas de luz.
"""
#%%
"""
Variáveis de controle.
"""
SEE_DF_COMPLETE = True
DOWNLOAD_DF = False
#%%
"""
Bibliotecas 

"""
import requests # versão 2.32.2
import pandas as pd # versão 2.2.1
import io # versão 3.12.4
import warnings # versão 3.12.4

#%%
"""
Constrói a URL de consulta com base nos filtros fornecidos.
"""
def build_query_url(base_url, filters):
    # Cria a string de parâmetros de consulta
    filter_params = "&".join([f"{key}={value}" for key, value in filters.items()])
    # Concatena a URL base com os parâmetros
    query_url = f"{base_url}search.csv?{filter_params}"
    return query_url

#%%
"""
Utilizamos duas url's pois não conseguimos ter acesso a todas as informações importante 
do exoplaneta pela tabela EAOT, mas dela conseguimos linkar a magnitede da estrela (K_mag)
com o nome do exoplaneta. No segundo link, fazemos a pesquisa sobre as propriedade do 
exoplaneta baseada no seu nome. Assim conseguimos selecionar os alvos de interesse baseados
na magnitude da estrela.
"""
# URLs base
URL_NAME_MAGNITUDE = "https://catalogs.mast.stsci.edu/api/v0.1/eaot/"
URL_ALL_PROPERTIES = "https://exo.mast.stsci.edu/api/v0.1/exoplanets/"

"""
Definindo filtros que serão utilizados na pesquisa, ao total são 17 filtros possíveis,
para ver mais : 
    * exoplanets table : https://catalogs.mast.stsci.edu/eaot
"""
filters = {
    "K_mag.min": "4",
    "K_mag.max": "13",
    "Rp.min": "10",
    "catalog_name": "toi"}

# Montando a URL com os filtros
URL_nm = build_query_url(URL_NAME_MAGNITUDE, filters)

#%%
"""
Realiza a requisição para a URL fornecida e retorna um DataFrame com os dados.Se a 
resposta da requisição não for bem-sucedida (status code diferente de 200), ou caso 
não seja encontrado nenhum objeto mas a requisição seja bem sucedida, o  'Exception' 
fará com que seja retornado o erro 'e' e o código não seja interrompido de maneira 
abrupta.

"""
# Função para realizar a requisição e retornar o DataFrame
def fetch_dataframe_from_url(url):
    response = requests.get(url)
    # A requisição da certo se o status for 200
    if response.status_code == 200:
        print('Requisição bem-sucedida:', response.status_code)
        # Transforma o arquivo da requisição .csv() em df
        return pd.read_csv(io.StringIO(response.text))
    else:
        raise Exception(f"Erro na requisição: {response.status_code}")
        
#%%
# Obtendo os dados da tabela EAOT
try:
    df_name_mag = fetch_dataframe_from_url(URL_nm)
    print(f'Foram encontrados {len(df_name_mag)} objetos nessa pesquisa.')
except Exception as e:
    print(e)

#%%
"""
Obtém as propriedades dos exoplanetas com base em seus nomes. Essa função será 
responsável por construir um dataframe com todas as informações úteis.

"""
def fetch_exoplanet_properties(base_url, planet_names, columns):
    
    # df vazio com as colunas específicadas 
    df_prop = pd.DataFrame(columns=columns)
    for i, planet_name in enumerate(planet_names):
        try:
            url = f"{base_url}{planet_name}/properties/"
            # Requisição .get() pela url
            response = requests.get(url)
            # Levanta uma exceção para status de erro que será tratada pelo 'except'
            response.raise_for_status()  
            # Filtro para o df linha ter somente as informações úteis
            df_line_prop = pd.DataFrame(response.json())[columns]
            # Suprimir aviso por conter elementos 'NaNs'
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                # Empilhar df(linhas) e organizar os indíces de [0,n]
                df_prop = pd.concat([df_prop, df_line_prop], ignore_index=True)
        
        except requests.RequestException as e:
            # Imprime uma mensagem de erro se a requisição falhar
            print(f"Erro ao obter dados para {planet_name}: {e}")
        except KeyError as e:
            # Imprime uma mensagem de erro se houver problema ao processar os dados
            print(f"Erro ao processar dados para {planet_name}: {e}")
            
    return df_prop

#%%
"""
Motagem do datafram com as informações mais úteis sobre o exoplaneta e a estrela :
    * Nome do planeta ;
    * Raio do planeta ;
    * Massa do planeta ;
    * Duração do trânsito planetário ;
    * Período orbital ;
    * Nome da estrela
    * Raio da estrela ;
    * Massa da estrela ;
    * Magnitude da estrela ;
    * Temperatura da estrela. 

"""
# Garantia de que haverá um dataframe 'df_name_mag'
if 'df_name_mag' in locals():
    
    # Seleciona o nome dos planetas, removendo possiveis 'NaNs' e transforma em lista
    planet_names = df_name_mag['Planet_Name'].dropna().unique().tolist()
    
    columns = ['planet_name',
               'Rp', 'Mp', 
               'transit_duration', 'orbital_period', 
               'star_name',
               'Rs', 'Ms', 
               'Kmag', 'Teff']
    
    df_prop = fetch_exoplanet_properties(URL_ALL_PROPERTIES, planet_names, columns)
    print(f"Propriedades obtidas para {len(df_prop)} exoplanetas.")

#%%

if SEE_DF_COMPLETE:
    pd.set_option('display.max_rows', None)  # Exibir todas as linhas
    pd.set_option('display.max_columns', None)  # Exibir todas as colunas
    pd.set_option('display.max_colwidth', None)  # Exibir todo o conteúdo das células
    pd.set_option('display.expand_frame_repr', False)  # Não expandir o df em múltiplas linhas
    print(df_prop)
#%%

if DOWNLOAD_DF:
    diretorio = '/graduacao/joshuakipper/Documentos/ic/exoplanetas/light-curves/data_exoplanets/'
    nome_arquivo = 'data_MAST.csv'
    output_file = diretorio + nome_arquivo
    df_prop.to_csv(output_file, index=False)
    print("DataFrame salvo!")

#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Este código importa informações de exoplanetas e suas estrelas associadas a partir 
da base de dados do ExoFOP. Ele realiza uma consulta utilizando a API do ExoFOP 
para obter uma tabela completa contendo todas as informações disponíveis sobre o 
exoplaneta específico. Em seguida, a tabela é filtrada, resultando em uma versão 
reduzida que contém apenas os dados essenciais necessários para o funcionamento do 
código principal.
"""
#%%
import pandas as pd # versão 2.2.1

#%%
# URL do banco de dados ExoFOP
url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=pipe"

#%%
"""
Para saber qual é a sintaxe correta da coluna, devemos olhar no próprio site do ExoFOP.
"""
# Colunas de interesse
colunas_interesse = [
    'TOI', 'Planet Radius (R_Earth)', 'Predicted Mass (M_Earth)', 
    'Period (days)', 'Duration (hours)', 'TIC ID', 
    'Stellar Radius (R_Sun)', 'Stellar Mass (M_Sun)', 
    'Stellar Eff Temp (K)', 'TESS Mag', 'TESS Disposition', 
    'Source', 'Detection', 'Sectors'
    ]

#%%
# Carregar dados do ExoFOP
dados = pd.read_csv(url, delimiter='|')

#%%
# Filtrar o DataFrame para manter apenas as colunas de interesse
dados_filtrados = dados[colunas_interesse].reset_index(drop=True)

#%%
# Aplicar filtros específicos para exoplanetas de interesse
exoplanetas_filtrados = dados_filtrados[
    (dados_filtrados['Planet Radius (R_Earth)'] > 13.0) &
    (dados_filtrados['Planet Radius (R_Earth)'] < 15.0) &
    (dados_filtrados['TESS Mag'] > 13.0) &
    (dados_filtrados['TESS Mag'] < 15.0) &
    (dados_filtrados['TESS Disposition'] == 'KP') &
    (dados_filtrados['Detection'] == 'SPOC')
    ].reset_index(drop=True)

#%%
# Diretório e nome do arquivo de saída
diretorio = '/graduacao/joshuakipper/Documentos/ic/exoplanetas/light-curves/data_exoplanets/'
nome_arquivo = 'data_ExoFOP.csv'
arquivo_saida = diretorio + nome_arquivo

#%%
# Salvar o DataFrame filtrado em um arquivo CSV
exoplanetas_filtrados.to_csv(arquivo_saida, index=False)
print("DataFrame salvo!")

#%%

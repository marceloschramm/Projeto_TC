import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from Acao import Acao #importar minha classe personalizada de obtenção dos dados financeiros
from logger_config import logger # importa o sistema de log, agora basta escrever as mensagens com logger.info()
import plotly.graph_objects as go
from datetime import datetime
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score





#//////////////////////////////////////////////////////////////////////
# FUNÇÕES:
#----------------------------------------------------------------------
# Função utilizada para medir a qualidade do modelo
def wmape(y_true, y_pred):
  resultado = (np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())
  return resultado
   
#----------------------------------------------------------------------
# VERIFICAÇÃO DA ACURÁCIA DOS MODELOS ATRAVÉS DA CRIAÇÃO DE VARIÁVEL TARGET
def resultado_target(abertura, fechamento):
    val_ab = abertura.values
    val_fc = fechamento.values
    delta = val_ab - val_fc
    resultado = delta
    for i in range(len(delta)):
        resultado[i] = (2 if delta[i] < 0 else (1 if delta[i] == 0 else 0))
    return resultado
#///////////////////////////////////////////////////////////////////////




#inicia configuração do streamlit
st.set_page_config(layout="wide") #seta streamlit para usar a largura inteira da página

#---------------------------------
# TÍTULO DO APP
#---------------------------------
st.title('Prevendo tendências de papéis da B3')
logger.info('Aplicação iniciada')

#-----------------------------------
# CONTEÚDO DO CORPO CENTRAL DA PÁGINA
#-----------------------------------
# 
st.write('Navegue entre as abas abaixo. Em caso de dúvida acesse a aba Info')

#-----------------------------------
# DEFININDO AS OPÇÕES PARA VISUALIZAÇÃO
#-----------------------------------
dict_acoes = {
    'IBOVESPA': '^BVSP',
    'Banco do Brasil': 'BBAS3.SA'
}

dict_periodos = {
    '1 mês': "1mo",
    '3 meses': "3mo",
    '6 meses': "6mo",
    '1 ano': "1y",
    '3 anos': "3y",
    '5 anos': "5y",
    '10 anos': "10y",
    '5 dias': "5d"
}
#-----------------------------------
# CONFIGURANDO A COLUNA LATERAL
#-----------------------------------
with st.sidebar:
    
    st.image('bola_cristal.jpg')
    
    st.write("Selecione os períodos e uma ação ou índice.")
    
    add_selectbox_acao = st.selectbox(
    "Escolha o papel",
    dict_acoes
    )
    
    add_selectbox_periodo = st.selectbox(
    "Escolha o período de visualização",
    dict_periodos
    )
    
    add_previsao = st.number_input(
    "Selecione o período de previsão", 
    value=5, 
    placeholder="Digite um número de 1 a 15",
    step=1,
    min_value=1,
    max_value=15    
    )
    st.write("O período de previsão é ", add_previsao)
    
    logger.info(
    f"Usuário selecionou papel={add_selectbox_acao}, "
    f"período={add_selectbox_periodo}, "
    f"previsão={add_previsao} dias"
    )

    #--------------------------------------------------
    # BLOCO QUE FAZ A SELEÇÃO DO MODELO A UTILIZAR,
    # QUE FOI PREVIAMENTE TREINADO COM O NOTEBOOK
    # PRESENTE NO DIRETÓRIO DO APP.
    # ESTE TRECHO PODE SER ALTERADO PARA IMPLEMENTAR
    # MAIS PAPÉIS DA B3.
    # ESTE MESMO PARÂMETRO COMANDA A LEITURA DOS METADADOS
    # CORRETOS.
    #----------------------------------------------------

    PAPEL = ''

    match add_selectbox_acao:
        case 'Banco do Brasil':
            PAPEL = 'BANCO_DO_BRASIL'
        case 'IBOVESPA':
            PAPEL = 'IBOVESPA' 
    
    
#-------------------------------------------------------   
# BASEADO NAS OPÇÕES ESCOLHIDAS PELO USUÁRIO
# BAIXA OS VALORES DO PAPEL SELECIONADO
#------------------------------------------------------
#--------------------------------------------------------------------------------
# LÊ METADADOS E DEFINE A DATA LIMITE PARA BAIXAR OS DADOS DO PAPEL ESCOHIDO
# A DATA LIMITE É O ÚLTIMO DIA UTILIZADO PARA TREINO
#--------------------------------------------------------------------------------
metadados={} #vai guardar os metadados

# FUNÇÃO QUE LÊ O TXT E GUARDA NO DICIONÁRIO AS INFORMAÇÕES
with open(f'app/metadata/meta_{PAPEL}.txt', "r", encoding="utf-8") as arquivo:
    for linha in arquivo:
        linha = linha.strip()  # remove espaços e quebras de linha
        if linha:  # ignora linhas vazias
            chave, valor = linha.split("=", 1)
            metadados[chave] = valor

logger.info(f'Metadados (meta_{PAPEL}.txt) lidos com sucesso')

data_treino_str = metadados['data_treino'] #data final do DF utilizado no treino e teste do modelo
data_treino = datetime.strptime(data_treino_str, '%Y-%m-%d %H:%M:%S')
modelo = metadados['model_name']
data_final_str = metadados['data_teste']
data_final = pd.to_datetime(data_final_str) #transforma a string da data lida no txt no formato datetime
papel_treino = metadados['papel']

ticker = dict_acoes[add_selectbox_acao]
periodo = dict_periodos[add_selectbox_periodo] #periodo apenas para a aba de visualização
papel = Acao(ticker)
df_vis = papel.retorna_df(periodo, data_treino) #monta o df para visualização
df = papel.retorna_df('10y', data_final) # df para tabelas dos modelos de predição

logger.info('DataFrames carregados com sucesso')

#---------------------------------------------------
# BAIXANDO OS MODELOS E REALIZANDO PREVISÕES
#---------------------------------------------------
#--------------------------------------------
# ajuste do DF para uso dos modelos
#--------------------------------------------

h=add_previsao #horizonte de previsão
limite = df.index[-h] #timestamp que determina treino e teste
#especificações do statsforecast
df_stats = df[['minima', 'maxima', 'fechamento']].copy()
df_ar = df_stats.reset_index()
df_ar = df_ar.rename(columns={'Date': 'ds', 'fechamento': 'y'})
df_ar['unique_id'] = 1
#definição de treino e teste
treino = df_ar.loc[df_ar['ds'] < limite]
teste = df_ar.loc[df_ar['ds'] >= limite]
logger.info('Períodos de treino e teste definidos')
 
#------------------------------------------
# PREDIÇÃO DAS EXÓGENAS: (RETIRADO NA VERSÃO 1.1)
# Carrega os modelos já treinados, e depois junta as previsões
# em um único DF que irá servir de dados externos para o modelo
# principal.
#------------------------------------------

#model_min = joblib.load(f'app/models/model_prev_min_{PAPEL}.pkl')
#logger.info('Modelo de previsão da mínima carregado')
#exog_min = model_min.predict(h=h)
#exog_min.rename(columns={'AutoARIMA': 'minima'}, inplace=True)
#model_max = joblib.load(f'app/models/model_prev_max_{PAPEL}.pkl')
#logger.info('Modelo de previsão da máxima carregado')
#exog_max = model_max.predict(h=h)
#exog_max.rename(columns={'AutoARIMA': 'maxima'}, inplace=True)
#exog = exog_min.merge(exog_max[['ds', 'maxima']], on='ds', how='left')
#logger.info('DF de máxima e mínima criado')

#-------------------------------------------------------------
# LOAD DO MODELO TREINADO E PREDIÇÃO DO PERÍODO SELECIONADO
#-------------------------------------------------------------
#@st.cache_resource
def load_modelos():
    scaler = joblib.load(f'app/models/scaler_{PAPEL}.pkl')
    modelo = joblib.load(f'app/models/model_{PAPEL}.pkl')
    return scaler, modelo

scaler, modelo_ar = load_modelos()
#modelo_ar = joblib.load(f'app/models/model_{PAPEL}.pkl')
logger.info('Modelo de previsão principal carregado')
df_pred = modelo_ar.predict(h=h)#, X_df=exog)
df_pred['ARIMA'] = scaler.inverse_transform(df_pred['ARIMA'].values.reshape(-1,1)).flatten()

#------------------------------------------------------------
# MONTAGEM DA TABELA FINAL PARA CÁLCULOS DE PERFORMANCE
# Adiciono ao DF de previsão a coluna de abertura e fechamento real, a fim
# de verificar se a tendência de subida ou queda do índice
# ou ação está sendo prevista corretamente, já que prevemos
# o fechamento.
#------------------------------------------------------------
ab_fch = df.loc[df.index >= limite][['abertura', 'fechamento']].values
df_pred[['abertura', 'fechamento']] = ab_fch
df_pred = df_pred.rename(columns={'ARIMA': 'fechamento_predito'})
df_pred['tendencia_real'] = resultado_target(df_pred['abertura'], df_pred['fechamento']) #diferença entre abertura e fechamento real
df_pred['tendencia_predita'] = resultado_target(df_pred['abertura'], df_pred['fechamento_predito']) #diferença entre abertura e fechamento predito
# cálculo dos índices de performance
wmape_= wmape(df_pred['fechamento'].values, df_pred['fechamento_predito'].values)
acuracia = accuracy_score(df_pred['tendencia_real'].values, df_pred['tendencia_predita'].values)
f1 = f1_score(df_pred['tendencia_real'].values, df_pred['tendencia_predita'].values, average='macro')
recall = recall_score(df_pred['tendencia_real'].values, df_pred['tendencia_predita'].values, average='macro')
logger.info('Previsão e cálculos de performance realizados')


#-----------------------------------
# CONFIGURANDO AS ABAS
#-----------------------------------
tab1, tab2, tab3 = st.tabs(['Visualização', 'Detalhamento Previsão', 'Info'])
#-----------------------------------
#ABA DE VISUALIZAÇÃO DOS DADOS
#-----------------------------------
df_vis_pred = df_pred[['ds', 'fechamento_predito']].copy()
df_vis_pred = df_vis_pred.set_index('ds')
df_vis_pred = df_vis_pred.rename(columns={'fechamento_predito': 'fechamento'})

primeiro_dia = df_vis_pred.loc[df_vis_pred.index == df_vis_pred.index[0]]
df_vis_graf = df_vis[['fechamento']].copy()
df_vis_graf.loc[primeiro_dia.index[0]] = primeiro_dia['fechamento'].values

with tab1:
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=df_vis_graf.index,  # x-axis
        y=df_vis_graf['fechamento'],  # y-axis
        mode='lines',  # Connect data points with lines
        name='Fechamento do dia'  # Name in the legend
    ))

    fig1.add_trace(go.Scatter(
        x=df_vis_pred.index,  # x-axis
        y=df_vis_pred['fechamento'],  # y-axis
        mode='lines',  # Connect data points with lines
        name='Previsão',  # Name in the legend
        line_color='rgba(255, 182, 193, .9)'
    ))

    # Layout parameters
    fig1.update_layout(
        title=f'Fechamento do papel {add_selectbox_acao}',  # Title
        xaxis_title='Data',  # y-axis name
        yaxis_title='Fechamento',  # x-axis name
        xaxis_tickangle=45,  # Set the x-axis label angle
        showlegend=True,     # Display the legend
    )
    # IMPRIMIR AS INFORMAÇÕES NO CENTRO
    st.plotly_chart(fig1)
    st.dataframe(df_vis)
    
    
#--------------------------------
# ABA DE PREVISÃO
#--------------------------------
with tab2:
    st.write('Previsão para o período selecionado')
    
    #--------------------------------------------------------
    # VISUALIZAÇÃO DOS DADOS PREVISTOS
    #--------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="WMAPE", value=f'{wmape_:.2f}')
    col2.metric(label="Precisão", value=f'{acuracia:.2%}')
    col3.metric(label="F1-Score", value=f'{f1:.2%}') 
    col4.metric(label="Recall", value=f'{recall:.2%}')   
    #-----------------------------------------------------
    # MONTAGEM DO GRÁFICO DA PREVISÃO
    #-----------------------------------------------------
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df_pred.index,  # x-axis
        y=df_pred['fechamento'],  # y-axis
        mode='lines+markers',  # Connect data points with lines
        name='Fechamento do dia'  # Name in the legend
    ))
    
    fig2.add_trace(go.Scatter(
        x=df_pred.index,  # x-axis
        y=df_pred['fechamento_predito'],  # y-axis
        mode='lines+markers',  # Connect data points with lines
        name='Fechamento Predito',  # Name in the legend
        marker_color='rgba(255, 182, 193, .9)'
    ))
    
    fig2.add_trace(go.Scatter(
        x=df_pred.index,  # x-axis
        y=df_pred['abertura'],  # y-axis
        mode='lines+markers',  # Connect data points with lines
        name='Abertura', # Name in the legend  
        marker_color='rgba(0, 182, 193, .9)'
    ))

    # Layout parameters
    fig2.update_layout(
        title=f'Previsão do papel {add_selectbox_acao} para {add_previsao} dias.',  # Title
        xaxis_title='Data',  # y-axis name
        yaxis_title='Valores negociados',  # x-axis name
        xaxis_tickangle=45,  # Set the x-axis label angle
        showlegend=True,     # Display the legend
    )
    #-----------------------------------------------------------------
    # CONFIGURAR UMA TABELA PARA VISUALIZAÇÃO DAS PREVISÕES
    #-----------------------------------------------------------------
    df_vis_tendencia = df_pred[['ds', 'tendencia_real', 'tendencia_predita']].copy()
    df_vis_tendencia = df_vis_tendencia.rename(columns={'ds': 'Data',   'tendencia_real': 'Tendência Real', 'tendencia_predita': 'Tendência Predita'})
    df_vis_tendencia['Tendência Real'] = df_vis_tendencia['Tendência Real'].apply(lambda x: 'Subiu' if x == 2 else('Igual' if x == 1 else 'Desceu'))
    df_vis_tendencia['Tendência Predita'] = df_vis_tendencia['Tendência Predita'].apply(lambda x: 'Subiu' if x == 2 else('Igual' if x == 1 else 'Desceu'))
    # FUNÇÃO QUE APLICA O ESTILO NAS CÉLULAS
    def estilo_previsao(row):
        if row["Tendência Predita"] == row["Tendência Real"]:
            cor = "background-color: #d4edda; color: #155724"
        else:
            cor = "background-color: #f8d7da; color: #721c24"

        return [
            cor if col == "Tendência Predita" else ""
            for col in row.index
        ]
    
    # IMPRIMIR AS INFORMAÇÕES
    st.plotly_chart(fig2)
    
    col1_1, col1_2 = st.columns([1, 1])
    
    with col1_1:
        if df_vis_tendencia['Data'].count() > 6:
            df_estilizada = df_vis_tendencia.loc[0:6].style.apply(estilo_previsao, axis=1)
            st.dataframe(df_estilizada)
        else:
            df_estilizada = df_vis_tendencia.style.apply(estilo_previsao, axis=1)
            st.dataframe(df_estilizada)
            
    with col1_2:
        if df_vis_tendencia['Data'].count() > 7:
            df_estilizada = df_vis_tendencia.loc[7:].style.apply(estilo_previsao, axis=1)
            st.dataframe(df_estilizada)
        else:
            pass
        
    logger.info('Gráficos e visualizações apresentados')
    
    
    
#--------------------------------
# ABA DE INFORMAÇÕES
#--------------------------------
with tab3:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write('''
                 Esse app foi desenvolvido para prever se o papel selecionado irá fechar
                 em alta ou baixa em relação à abertura do pregão.
                 
                 Para isso, selecione o papel e o período de previsão, de até 15 dias adiante.
                 
                 O período de visualização serve apenas para visualizar o comportamento do papel
                 no gráfico presente na aba de visualização.
                 
                 Na aba de Previsão estarão o gráfico com a previsão do modelo e os índices de performance.
                 Além disso, abaixo do gráfico está posicionada uma tabela com o resultado da previsão
                para cada um dos dias, em comparação com o real.
                
                Qualquer problema com o app, favor enviar email para: celoschramm@gmail.com
                 ''')
    with col2:
        st.write('Informações do aplicativo:')
        st.markdown('''
        :gray[Versão: ] 1.0
            
        :gray[Desenvolvedor: ] Marcelo Eduardo Schramm Junior
        ''')
        st.markdown(f':gray[Data final de treino: ] {data_treino}')
        
        st.markdown(f':gray[Modelo utilizado: ] {modelo}')

        st.markdown(f':gray[Ação ou Índice utilizado no treino: ] {papel_treino}')

        
        
    logger.info('Aba de informações carregada')
    
    


#importa as bibliotecas necessárias para o funcionamento da classe.

import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from logger_config import logger

class Acao:
  # períodos válidos segundo yfinance
  VALID_PERIODS = {
      "1d", "5d",
      "1mo", "3mo", "6mo",
      "1y", "2y", "5y", "10y",
      "ytd", "max"
  }
    #------------------------------
    # A CLASSE RECEBE INICIALMENTE O PAPEL ESCOLHIDO
    #-------------------------
  def __init__(self, ticker='^BVSP'):
    self.ticker = ticker
    
  def calcular_data_inicial(self, data_final, periodo):
    if periodo == "5d":
        return data_final - relativedelta(days=5)

    elif periodo == "1mo":
        return data_final - relativedelta(months=1)

    elif periodo == "3mo":
        return data_final - relativedelta(months=3)

    elif periodo == "6mo":
        return data_final - relativedelta(months=6)

    elif periodo == "1y":
        return data_final - relativedelta(years=1)

    elif periodo == "3y":
        return data_final - relativedelta(years=3)

    elif periodo == "5y":
        return data_final - relativedelta(years=5)

    elif periodo == "10y":
        return data_final - relativedelta(years=10)

  def print_acao(self):
    print(f'O DataFrame representa a ação {self.ticker}')

    #------------------------------
    # FUNÇÃO QUE RETORNA O DATAFRAME PARA O USUÁRIO
    #-------------------------
  def retorna_df(self, periodo='1y', data_final='2025-12-05 00:00:00'):
    #------------------------------
    # VERIFICA SE OPÇÕES DE PERÍODO ESCOLHIDAS SÃO VÁLIDAS
    #-------------------------
    if periodo not in self.VALID_PERIODS:
      raise ValueError(
            f"Período inválido: '{periodo}'."
            f"Use um dos seguintes: {sorted(self.VALID_PERIODS)}"
                      )
    #---------------------------------
    # OBTÉM O DATAFRAME E FAZ OS PROCESSO BÁSICOS DE TRATAMENTO
    #---------------------------------
    else:
      print(f'Retornando os dados do papel {self.ticker}, no período de {periodo}')
      logger.info(f'Retornando os dados do papel {self.ticker}, no período de {periodo}')
      # download dos dados
      data_inicial = self.calcular_data_inicial(data_final, periodo)
      df = yf.download(self.ticker, start=data_inicial, end=data_final)
      # padroniza os nomes das colunas
      df.rename(columns={'Close': 'fechamento', 'High': 'maxima', 'Low': 'minima', 'Open': 'abertura'}, inplace=True)
      logger.info('Colunas renomeadas')
      # excluir nível de colunas, não há necessidade de manter
      df.columns = df.columns.droplevel('Ticker')
      # exclui coluna não relevante para a análise
      df.drop('Volume', axis=1, inplace=True)

      # VERIFICAÇÃO DE NULOS E DUPLICADOS
      if df.isnull().sum().sum() != 0:
        logger.warning('Dados nulos no DF do yfinance')
        df.dropna(inplace=True)
      else:
        logger.info('Sem dados nulos')
        pass
      
      if df.loc[df.duplicated()].count().sum() != 0:
        logger.warning(f'Dados duplicados em {df.loc[df.duplicated()].index.unique()}')
        df.drop_duplicates(inplace=True)
      else:
        logger.info('Sem dados duplicados')
        pass
      
      return df
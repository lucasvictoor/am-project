"""
Pipeline de Treinamento e Validação.
Executa o treinamento em paralelo, avalia as métricas de justiça (SPD e DI) e performance,
e utiliza Regressão Logística padrão como fallback para bases com viés extremo.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import resample

# Configuração do caminho para o módulo Kamishima (AIF360)
import aif360
kam_path = os.path.join(os.path.dirname(aif360.__file__), 'algorithms', 'inprocessing', 'kamfadm-2012ecmlpkdd')
if kam_path not in sys.path:
    sys.path.insert(0, kam_path)
from fadm.lr.pr import LRwPRType4

# Configurações globais
ETAS = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0] 
N_TREINO = 10000
MAX_TESTE = 20000 
FEATURES = ['VALOR_TRANSACAO', 'VALOR_SALDO', 'valor_total_dia', 'n_transacoes_dia', 
            'valor_medio_dia', 'dia_da_semana', 'final_de_semana', 'pct_saldo_gasto']
SENSITIVE = 'RAMO_ATIVIDADE_1'
TARGET = 'I-d'
SEED = 42

OUT_DIR = './resultados'
DATA_DIR = './Equidade - bases de dados públicas'

BASES_MAP = {
    'pd_v0': ('pd_v0.csv', 'PD', 0.0), 'pd_v0.25': ('pd_v0_25.csv', 'PD', 0.25),
    'pd_v0.5': ('pd_v0_5.csv', 'PD', 0.5), 'pd_v0.75': ('pd_v0_75.csv', 'PD', 0.75),
    'pd_v1': ('pd_v1.csv', 'PD', 1.0), 'gs_v0': ('gs_v0.csv', 'GS', 0.0),
    'gs_v0.25': ('gs_v0_25.csv', 'GS', 0.25), 'gs_v0.5': ('gs_v0_5.csv', 'GS', 0.5),
    'gs_v0.75': ('gs_v0_75.csv', 'GS', 0.75), 'gs_v1': ('gs_v1.csv', 'GS', 1.0)
}

class PrejudiceRemoverModel(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=1.0):
        self.eta = eta
        self.model = None
        self.fallback = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        
        s_data = X[:, -1]
        X_data = X[:, :-1]
        
        is_valida = pd.crosstab(s_data, y).shape == (2, 2)
        
        if is_valida:
            self.model = LRwPRType4(eta=self.eta, C=1.0)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model.fit(X.astype(np.float32), y, ns=1, itype=3)
            except Exception:
                self.fallback = LogisticRegression(max_iter=1000).fit(X_data, y)
        else:
            self.fallback = LogisticRegression(max_iter=1000).fit(X_data, y)
            
        return self
        
    def predict(self, X):
        if self.fallback is not None:
            return self.fallback.predict(X[:, :-1])
        return np.argmax(self.model.predict_proba(X.astype(np.float32)), axis=1)

def carregar_e_preparar_dados(nome_base):
    path = os.path.join(DATA_DIR, BASES_MAP[nome_base][0])
    df = pd.read_csv(path, low_memory=False)
    
    df = df[df[SENSITIVE].isin([1, 4])].copy()
    df[SENSITIVE] = df[SENSITIVE].replace({4: 0}).astype(np.int8)
    
    df['DATA_LANCAMENTO'] = pd.to_datetime(df['DATA_LANCAMENTO'], errors='coerce')
    df = df.dropna(subset=['DATA_LANCAMENTO']).sort_values(['CPF_CNPJ_TITULAR', 'DATA_LANCAMENTO'])
    df['dia_da_semana'] = df['DATA_LANCAMENTO'].dt.dayofweek.astype(np.int8)
    df['final_de_semana'] = df['dia_da_semana'].isin([5, 6]).astype(np.int8)
    df['pct_saldo_gasto'] = (df['VALOR_TRANSACAO'] / (df['VALOR_SALDO'].abs() + 1)).astype(np.float32)
    
    df['data_dia'] = df['DATA_LANCAMENTO'].dt.date
    grp = df.groupby(['CPF_CNPJ_TITULAR', 'data_dia'])['VALOR_TRANSACAO']
    df['valor_total_dia'] = grp.transform('sum').astype(np.float32)
    df['n_transacoes_dia'] = grp.transform('count').astype(np.int32)
    df['valor_medio_dia'] = grp.transform('mean').astype(np.float32)
    
    return df[FEATURES + [SENSITIVE, TARGET]].fillna(0)

def calcular_vies_dados_iniciais():
    registros_vies = []
    for nome, (arq, serie, vies_v) in BASES_MAP.items():
        df = carregar_e_preparar_dados(nome)
        
        # A classe favorável é 0 (Transação Aprovada/Normal)
        taxa_u = (df[df[SENSITIVE] == 0][TARGET] == 0).mean()
        taxa_p = (df[df[SENSITIVE] == 1][TARGET] == 0).mean()
        
        # Diferença (SPD) e Razão (DI)
        spd = taxa_u - taxa_p
        di = (taxa_u / taxa_p) if taxa_p > 0 else 0.0
        
        registros_vies.append({
            'base': nome, 'serie': serie, 'vies': vies_v,
            'abs_spd_dados': abs(spd),
            'di_dados': di
        })
    return pd.DataFrame(registros_vies)

def treinar_e_avaliar_base(nome_treino):
    df_tr = carregar_e_preparar_dados(nome_treino)
    
    n_amostras = min(len(df_tr[df_tr[TARGET]==0]), len(df_tr[df_tr[TARGET]==1]), N_TREINO // 2)
    df_bal = pd.concat([
        resample(df_tr[df_tr[TARGET]==0], n_samples=n_amostras, replace=False, random_state=SEED),
        resample(df_tr[df_tr[TARGET]==1], n_samples=n_amostras, replace=False, random_state=SEED)
    ]).sample(frac=1, random_state=SEED)
    
    X_train = df_bal[FEATURES + [SENSITIVE]].values
    y_train = df_bal[TARGET].values
    
    modelos_treinados = {}
    for eta in ETAS:
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('pr_model', PrejudiceRemoverModel(eta=eta))
        ])
        pipeline.fit(X_train, y_train)
        modelos_treinados[eta] = pipeline

    resultados_tarefa = []
    
    for test_k in BASES_MAP:
        if test_k == nome_treino: continue
        
        df_te = carregar_e_preparar_dados(test_k)
        
        if len(df_te) > MAX_TESTE: 
            df_te = df_te.sample(n=MAX_TESTE, random_state=SEED)
        
        X_test = df_te[FEATURES + [SENSITIVE]].values
        y_test = df_te[TARGET].values
        s_test = df_te[SENSITIVE].values 
        
        for eta in ETAS:
            y_pred = modelos_treinados[eta].predict(X_test)
            
            mask_u, mask_p = (s_test == 0), (s_test == 1)
            
            # Cálculo de justiça sobre a classe favorável (0 = Transação Aprovada)
            taxa_u = (y_pred[mask_u] == 0).mean() if mask_u.any() else 0.0
            taxa_p = (y_pred[mask_p] == 0).mean() if mask_p.any() else 0.0
            
            spd = taxa_u - taxa_p
            di = (taxa_u / taxa_p) if taxa_p > 0 else 0.0
            
            resultados_tarefa.append({
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'abs_spd': abs(spd), 
                'di': di, 
                'eta': eta, 
                'base_treino': nome_treino, 
                'serie_treino': BASES_MAP[nome_treino][1], 
                'vies_treino': BASES_MAP[nome_treino][2]
            })
            
    return resultados_tarefa

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()
    
    print("Processando viés original das bases de dados (SPD e DI)...")
    df_vies_inicial = calcular_vies_dados_iniciais()
    df_vies_inicial.to_csv(os.path.join(OUT_DIR, 'vies_dados.csv'), index=False)
    
    print("Iniciando validação...")
    jobs = Parallel(n_jobs=-1)(delayed(treinar_e_avaliar_base)(k) for k in BASES_MAP)
    
    df_res = pd.DataFrame([item for sub in jobs for item in sub])
    df_res.to_csv(os.path.join(OUT_DIR, 'resultados.csv'), index=False)

    print("\nMédias consolidadas por ETA:")
    resumo = df_res.groupby('eta')[['f1', 'recall', 'abs_spd', 'di']].mean()
    print(resumo.to_string())
    
    print(f"\nTreinamento concluído em {time.time() - t_start:.1f} segundos.")

if __name__ == '__main__':
    main()

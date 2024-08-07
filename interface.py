import pandas as pd
import numpy as np
from cpf_generator import CPF
from datetime import date, timedelta
import cplex
import numpy as np
from pyomo.environ import *
import streamlit as st
st.set_page_config(layout='wide')

def random_dates(start, end, K):
    start_u = start.value//10**9
    end_u = end.value//10**9
    return pd.to_datetime(np.random.randint(start_u, end_u, K), unit='s')


st.number_input("Número de Registros", key="n_rows", placeholder=0, min_value=1)
st.date_input("Data Inicial", key="date_init")
st.date_input("Data Final", key="date_end")
st.number_input('Número de Máquinas', key="n_machines", min_value=1)
st.number_input('Tempo Máximo na fila (minutos)', key="max_time", min_value=1)
st.number_input('Tempo de Procedimento no RaioX (segundos)', key="processing_time", min_value=1)
N_ROWS = st.session_state.n_rows
DATE_INIT = st.session_state.date_init
DATE_END = st.session_state.date_end
N_MACHINES = st.session_state.n_machines
MAX_TIME = st.session_state.max_time / 60
PROCESSING_TIME = st.session_state.processing_time / 3600


if st.button('Calcular'):
    # getting random dates 
    dates = random_dates(pd.to_datetime(DATE_INIT), pd.to_datetime(DATE_END), N_ROWS)

    df = pd.DataFrame({'id': range(0, N_ROWS),
                    'cpf': [CPF.generate() for _ in range(0, N_ROWS)],
                    'data': dates,})

    #truncate the timestamp to hour
    df['data'] = df['data'].dt.floor('h')
    data_set = set(df['data'])
    data_dict = {k: v for v, k in enumerate(data_set)}
    df['data'] = df['data'].map(data_dict)
    df_gp = df.groupby(['data']).size().reset_index(name='counts')

    model = ConcreteModel("Multiobjective Optimization")

    #Implementing model above with Pyomo

    model = ConcreteModel()

    model.datas = Set(initialize=df_gp.data, ordered=True)

    model.maquinas = Set(initialize=list(range(0, N_MACHINES)), ordered=True)

    model.pessoas = Param(model.datas, initialize=df_gp['counts'].to_dict())

    model.m = Var(model.maquinas, model.datas, domain=Binary)

    #Constraints

    def constraint1(model, d):
        return model.pessoas[d] * PROCESSING_TIME <= sum(model.m[m, d] for m in model.maquinas) * MAX_TIME

    model.constraint1 = Constraint(model.datas, rule=constraint1)


    def obj1(model):
        return sum(model.m[m, d] for m in model.maquinas for d in model.datas)

    model.obj1 = Objective(rule= obj1, sense=minimize)

    solver = SolverFactory('cplex')

    results = solver.solve(model)
    st.write(results["Solver"])
    st.write(results["Solution"])

    #geting m values
    m_values = []

    for d in model.datas: 
        m_data = []
        for m in model.maquinas:
            m_data.append(model.m[m, d].value)
        m_values.append(m_data)

    df_final = pd.DataFrame({'Horário': list(data_dict.keys()), 'Quantidade de Máquinas': [sum(m) for m in m_values]})
    st.dataframe(df_final.sort_values('Horário'), use_container_width=True)


import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle 
import lime
import lime.lime_tabular
from pickle import load
from PIL import Image
from matplotlib import pyplot as plt
import re
import mysql.connector

app = Flask(__name__)
#route -> endereço exemplo upecaruaru.com.br/deeptub
#função -> o que vai exibir naquela página
#template 
# Configuração da conexão com o banco de dados MySQL
db_config = {
    'host': '130.211.212.31',
    'user': 'maicon',
    'password': 'Hacker23Anos!',
    'database': 'tito'
}

# Estabelecer conexão com o banco de dados
conn = mysql.connector.connect(**db_config)


loaded_model = pickle.load(open('SVM03-11-2022_02-50-37.sav','rb'))
scalerfile = 'scaler2_03-19-2022_02-19-47.pkl'
scaler = load(open(scalerfile, 'rb'))
X_train = pd.read_csv('X_trainLIME_03-19-2022_03-59-34.csv', sep=';', index_col=False)
Y_train = pd.read_csv('Y_trainLIME_03-19-2022_03-59-34.csv', sep=';', index_col=False)

colunas = ('NU_IDADE_N','TRATAMENTO','RAIOX_TORA','TESTE_TUBE','FORMA','AGRAVDOENC','BACILOSC_E','BACILOS_E2','HIV','BACILOSC_6','DIAS')
prognosis = ''
def prognosis_tuberculosis(input_data):
    input_data_numpy = np.asarray(input_data)
    input_reshape = input_data_numpy.reshape(1,-1)
    #print(input_reshape)
    base_x = pd.DataFrame(input_reshape, columns=colunas)
    #print(base_x.head())
        
    test_scaled_set = scaler.transform(base_x)
    test_scaled_set = pd.DataFrame(test_scaled_set, columns=colunas)

    #print(test_scaled_set.head())

    class_names = ["Cura","Óbito"]
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, training_labels=Y_train,class_names=class_names, feature_names=X_train.columns, kernel_width=3, discretize_continuous=True, verbose=False)
    exp = explainer.explain_instance(test_scaled_set.values[0], loaded_model.predict_proba, num_features=11)
    #exp.show_in_notebook()
    lista = exp.as_list()
    lista2 = []
    predictions = loaded_model.predict(test_scaled_set)


    #predictions
    if(predictions[0]==3):
        #retorno = "A probabilidade de **óbito** no prognósitco da Tuberculose é de: {}%"
        #retorno = retorno.format(round(exp.predict_proba[1]*100,2))
        return (predictions[0],round(exp.predict_proba[1]*100,2),lista2,lista)
    else:
        #retorno = "A probabilidade de **cura** no prognósitco da Tuberculose é de: {}%"
        #retorno = retorno.format(round(exp.predict_proba[0]*100,2))
        return (predictions[0],round(exp.predict_proba[0]*100,2),lista2,lista)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prognostico")
def prognostico():
    return render_template("prognostico.html")

@app.route('/prognostico_form', methods=['POST'])
def processar_formulario():
    # Acessar os dados do formulário
    dados =  request.form
    form_tipo_de_tratamento = dados['form_tipo_de_tratamento']
    form_idade_do_paciente = dados['form_idade_do_paciente']
    form_radiografia_torax = dados['form_radiografia_torax']
    form_teste_tuberculinio = dados['form_teste_tuberculinio']
    form_forma_da_tuberculose = dados['form_forma_da_tuberculose']
    form_agravos_doenca_mental = dados['form_agravos_doenca_mental']
    form_hiv = dados['form_hiv']
    form_bacilosc_e = dados['form_bacilosc_e']
    form_bacilosc_e2 = dados['form_bacilosc_e2']
    form_bacilosc_6 = dados['form_bacilosc_6']
    form_dias_em_tratamento = dados['form_dias_em_tratamento']

    
    # Faça o processamento necessário com os dados
    prognosis = prognosis_tuberculosis([form_idade_do_paciente, form_tipo_de_tratamento, form_radiografia_torax, form_teste_tuberculinio, form_forma_da_tuberculose, form_agravos_doenca_mental, form_bacilosc_e, form_bacilosc_e2, form_hiv, form_bacilosc_6, form_dias_em_tratamento])
    
    value=str(prognosis[1])

    cursor = conn.cursor()
    consulta = """INSERT INTO tito_classificacoes 
              (idade, tipo_de_tratamento, radiografia_do_torax, teste_tuberculineo, forma_tuberculose, agravos_doenca_mental, hiv, baciloscopia_1_amostra, baciloscopia_2_amostra, baciloscopia_6_mes, dias_em_tratamento, classificacao_predita, probabilidade_predita) 
              VALUES 
              (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    valores = (
        int(form_idade_do_paciente),
        int(form_tipo_de_tratamento),
        int(form_radiografia_torax),
        int(form_teste_tuberculinio),
        int(form_forma_da_tuberculose),
        int(form_agravos_doenca_mental),
        int(form_hiv),
        int(form_bacilosc_e),
        int(form_bacilosc_e2),
        int(form_bacilosc_6),
        int(form_dias_em_tratamento),
        float(prognosis[0]),  # Converter para tipo float se necessário
        float(value)  # Converter para tipo float se necessário
    )
    print(consulta)
    print(valores)
    cursor.execute(consulta, valores)
    conn.commit()
    cursor.close()

    ListaResultado = []
    for X in prognosis[2]:
        ListaResultado.append(X)
    
    df = pd.DataFrame({"Attributes" : ListaResultado})
    if prognosis[0]==1:
        print('Classificado como: Cura 🔵')
        print('<h2>Probabilidade de: ', value,'% </h2> ')
        #st.metric(label=' ',value=str(prognosis[1])+'%')
        print("Atributos que influenciaram para este resultado por ordem de importância")
        #st.dataframe(prognosis[2])
        print(df)
        return render_template("resultado_prognostico.html",percentual=value, tipopredito='cura')
    else:
        print('Classificado como: Óbito 🔴')
        print('<h2>Probabilidade de: ', value,'% </h2> ')
        #st.metric(label=' ',value=str(prognosis[1])+'%')
        print("Atributos que influenciaram para este resultado por ordem de importância")
        #st.dataframe(prognosis[2])
        print(df)
        return render_template("resultado_prognostico.html",percentual=value,  tipopredito='obito')

    # Retorne a resposta ao cliente
    #return f'form_tipo_de_tratamento: {form_tipo_de_tratamento}, form_idade_do_paciente: {form_idade_do_paciente}, form_radiografia_torax: {form_radiografia_torax}, form_teste_tuberculinio: {form_teste_tuberculinio}, form_forma_da_tuberculose: {form_forma_da_tuberculose}, form_agravos_doenca_mental: {form_agravos_doenca_mental}, form_hiv: {form_hiv}, form_bacilosc_e: {form_bacilosc_e}, form_bacilosc_e2: {form_bacilosc_e2}, form_bacilosc_6: {form_bacilosc_6}, form_dias_em_tratamento: {form_dias_em_tratamento}'

@app.route("/artigospublicados")
def artigospublicados():
    return render_template("artigospublicados.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))



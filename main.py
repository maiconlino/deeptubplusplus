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
import os
from google.cloud.sql.connector import Connector
import sqlalchemy
import pymysql
import bcrypt

    #import mysql.connector

app = Flask(__name__)
#route -> endereço exemplo upecaruaru.com.br/deeptub
#função -> o que vai exibir naquela página
#template 
#Configuração para testar no MySQL local
# Configuração da conexão com o banco de dados MySQL
    # db_config = {
    #     'host': '130.211.212.31',
    #     'user': 'maicon',
    #     'password': 'Hacker23Anos!',
    #     'database': 'tito'
    # }

# Estabelecer conexão com o banco de dados
    #conn = mysql.connector.connect(**db_config)


loaded_model = pickle.load(open('SVM03-11-2022_02-50-37.sav','rb'))
scalerfile = 'scaler2_03-19-2022_02-19-47.pkl'
scaler = load(open(scalerfile, 'rb'))
X_train = pd.read_csv('X_trainLIME_03-19-2022_03-59-34.csv', sep=';', index_col=False)
Y_train = pd.read_csv('Y_trainLIME_03-19-2022_03-59-34.csv', sep=';', index_col=False)

colunas = ('NU_IDADE_N','TRATAMENTO','RAIOX_TORA','TESTE_TUBE','FORMA','AGRAVDOENC','BACILOSC_E','BACILOS_E2','HIV','BACILOSC_6','DIAS')
prognosis = ''





# initialize Connector object
connector = Connector()

# function to return the database connection
def getconn() -> pymysql.connections.Connection:
    conn: pymysql.connections.Connection = connector.connect(
        "deeptub:us-central1:tito",
        "pymysql",
        user="maicon",
        password="Hacker23Anos!",
        db="tito"
    )
    return conn

# create connection pool
pool = sqlalchemy.create_engine(
    "mysql+pymysql://",
    creator=getconn,
)

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

    # insert statement
    insert_stmt = sqlalchemy.text(
        """INSERT INTO tito_classificacoes 
                   (idade, tipo_de_tratamento, radiografia_do_torax, teste_tuberculineo, forma_tuberculose, agravos_doenca_mental, hiv, baciloscopia_1_amostra, baciloscopia_2_amostra, baciloscopia_6_mes, dias_em_tratamento, classificacao_predita, probabilidade_predita) 
                   VALUES 
                   (:idade, :tipo_de_tratamento, :radiografia_do_torax, :teste_tuberculineo, :forma_tuberculose, :agravos_doenca_mental, :hiv, :baciloscopia_1_amostra, :baciloscopia_2_amostra, :baciloscopia_6_mes, :dias_em_tratamento, :classificacao_predita, :probabilidade_predita)""",
    )

    with pool.connect() as db_conn:
        # insert into database
        db_conn.execute(insert_stmt, parameters={"idade": form_idade_do_paciente, "tipo_de_tratamento": form_tipo_de_tratamento, "radiografia_do_torax": form_radiografia_torax, "teste_tuberculineo":form_teste_tuberculinio, "forma_tuberculose":form_forma_da_tuberculose, "agravos_doenca_mental":form_agravos_doenca_mental, "hiv":form_hiv, "baciloscopia_1_amostra":form_bacilosc_e, "baciloscopia_2_amostra":form_bacilosc_e2, "baciloscopia_6_mes":form_bacilosc_6, "dias_em_tratamento":form_dias_em_tratamento, "classificacao_predita":prognosis[0], "probabilidade_predita":value})
        db_conn.commit()
    connector.close()
   
    

    ListaResultado = []
    for X in prognosis[2]:
        ListaResultado.append(X)
    
    df = pd.DataFrame({"Attributes" : ListaResultado})
    if prognosis[0]==1:
        #print('Classificado como: Cura 🔵')
        #print('<h2>Probabilidade de: ', value,'% </h2> ')
        #st.metric(label=' ',value=str(prognosis[1])+'%')
        #print("Atributos que influenciaram para este resultado por ordem de importância")
        #st.dataframe(prognosis[2])
        #print(df)
        return render_template("resultado_prognostico.html",percentual=value, tipopredito='cura')
    else:
        #print('Classificado como: Óbito 🔴')
        #print('<h2>Probabilidade de: ', value,'% </h2> ')
        #st.metric(label=' ',value=str(prognosis[1])+'%')
        #print("Atributos que influenciaram para este resultado por ordem de importância")
        #st.dataframe(prognosis[2])
        #print(df)
        return render_template("resultado_prognostico.html",percentual=value,  tipopredito='obito')

    # Retorne a resposta ao cliente
    #return f'form_tipo_de_tratamento: {form_tipo_de_tratamento}, form_idade_do_paciente: {form_idade_do_paciente}, form_radiografia_torax: {form_radiografia_torax}, form_teste_tuberculinio: {form_teste_tuberculinio}, form_forma_da_tuberculose: {form_forma_da_tuberculose}, form_agravos_doenca_mental: {form_agravos_doenca_mental}, form_hiv: {form_hiv}, form_bacilosc_e: {form_bacilosc_e}, form_bacilosc_e2: {form_bacilosc_e2}, form_bacilosc_6: {form_bacilosc_6}, form_dias_em_tratamento: {form_dias_em_tratamento}'

@app.route("/artigospublicados")
def artigospublicados():
    return render_template("artigospublicados.html")

@app.route('/cadastro', methods=['GET', 'POST'])
def cadastro():
    senha_criptografada = ""
    if request.method == 'POST':
        # Obter os dados do formulário
        form_nome_completo = request.form['form_nome_completo']
        form_cpf = request.form['form_cpf']
        form_email = request.form['form_email']
        form_sen = request.form['form_sen']
        form_senc = request.form['form_senc']
       
        #validacao back-end seguranca se o usuario desativar javascript ou manipula-lo no front
        vazio = False
        if form_nome_completo=="" or form_cpf=="" or form_email=="" or form_sen=="" or form_senc=="":
            vazio = True
        
        lenSenhaMenorQue6 = False
        if len(form_sen)<6:
            lenSenhaMenorQue6 = True
        
        lenSenhaMenorQue6Confirmacao = False
        if len(form_senc)<6:
            lenSenhaMenorQue6Confirmacao = True
 
        senhasIguais = False
        if form_sen==form_senc:
            senhasIguais = True
        # Aqui você pode realizar o processamento necessário, como salvar os dados em um banco de dados
        # ou qualquer outra lógica desejada.

        submissao = True
        if vazio or lenSenhaMenorQue6 or lenSenhaMenorQue6Confirmacao or not(senhasIguais):
            submissao = False
        else:
            #tudo certo pode cadastrar no banco
            # Gerar um salt (valor aleatório utilizado na criptografia)
            salt = bcrypt.gensalt()
            
            # Criptografar a senha com o salt
            senha_criptografada = bcrypt.hashpw(form_sen.encode('utf-8'), salt)

            # Exibir a senha criptografada
            #print(senha_criptografada.decode('utf-8'))

             # insert statement
            insert_stmt = sqlalchemy.text("""INSERT INTO tito_usuarios 
                        (nomeCompleto, cpf, senhaCriptografada, email) 
                        VALUES 
                        (:nomeCompleto, :cpf, :senhaCriptografada, :email)""",
            )

            with pool.connect() as db_conn:
                # insert into database
                db_conn.execute(insert_stmt, parameters={"nomeCompleto": form_nome_completo,"cpf": form_cpf,"senhaCriptografada": senha_criptografada,"email": form_email})
                db_conn.commit()
            connector.close()   
        return render_template('cadastro.html', submissao=submissao, vazio=vazio, lenSenhaMenorQue6=lenSenhaMenorQue6, lenSenhaMenorQue6Confirmacao=lenSenhaMenorQue6Confirmacao, senhasIguais=senhasIguais, teste=senha_criptografada)

    return render_template('cadastro.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))



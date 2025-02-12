import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model, setup, create_model, assign_model, plot_model, save_model  # type: ignore
import plotly.express as px  # type: ignore
from pathlib import Path
from openai import OpenAI

#init
for conf_var in ['df', 'trained', 'setup', 'model', 'clustered_df', 'loaded', 'history']:
    if conf_var not in st.session_state:
        st.session_state[conf_var] = None
st.set_page_config(layout='wide')

def load_data(file_obj, data_type='.csv'):
    """Załadowanie danch z CSV ;lub XLSX"""
    if data_type == '.csv':
        st.session_state['df'] = pd.read_csv(file_obj, sep=';')
    elif data_type =='.xlsx':
        st.session_state['df'] = pd.read_excel(file_obj)
    st.session_state['trained'] = False
    # st.info('Nowe dane, przetrenuj model...')
    st.session_state['loaded'] = True

def can_train():
    """Sprawdzenie czy model powinien być trenowany patrzac na zmienne sesji"""
    if not st.session_state['trained']: #Nie wytrenowano
        # st.warning('Nie wytrenowano')
        return True
    if st.session_state['trained'] and st.session_state['clustered_df'] is not None: #Trening już zakończony
        # st.warning('Trening już zakończony')
        return False
    if st.session_state['trained'] and st.session_state['clustered_df'] is None: #Brak klastra mimo iż powinien być
        # st.warning('Brak klastra mimo iż powinien być')
        return True

def train_model(data, session_id=1, force_train=False):
    """Trening modelu za pomocą wczytanych danych"""


    if force_train or can_train():
        st.session_state['setup'] = setup(data, session_id=session_id)
        kmeans = create_model('kmeans', num_clusters=8)
        st.session_state['clustered_df'] = assign_model(kmeans)
        save_model(kmeans, 'welcome_survey_cluster_v2')
        st.session_state['model'] = load_model('welcome_survey_cluster_v2')
        st.session_state['trained'] = True
        st.info('Trening zakończony')
        st.write(st.session_state)
    else:
        st.warning('Model już wytrenowany')

def prepare_prompts(source_df_data:pd.DataFrame):
    """Dane już w postaci df['Cluster'] == 'Cluster value'"""
    df_json = source_df_data.to_json()

    system_heredoc = f"""
    Jesteś analitykiem danych z 20 letnim stażem. Poniżej otrzymujesz dane na temat grupy osób.
    Są one uszeregowane w kolumny jak wiek, ulubione zwierzęta i inne
    Poniżej otrzymujesz te dane w postaci Pandas DataFrame serializowane za pomocą wbudowanej metody .to_json()
    
    {df_json}
    
    Zostaniesz zapytane o ten zbiór. Odpowiedz najlepiej jak potrafisz. Odpowiedź sformatuj jako Markdown    
    """

    # user_heredoc = f"{user_prompt}"

    return system_heredoc

@st.cache_resource
def get_open_ai_client(open_ai_key:str) -> OpenAI:
    """Stworzenie klienta OpenAI za pomocą klucza"""
    openai_client = OpenAI(api_key=open_ai_key)
    return openai_client

def ask_open_ai(open_ai_client, system_prompt:str, user_prompt:str, open_ai_model='gpt-4o-mini') -> str:
    """Przesłanie zapytania, wymagany zarówno normalny jak i systemowy prompt"""
    response = open_ai_client.chat.completions.create(
        model=open_ai_model, #model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content



with st.sidebar:


    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])



tab_train, tab_data, tab_predict, tab_history = st.tabs(['Trening', 'Dane', 'Predykcja', 'Historia zapytań'])
with tab_train:
    st.header('Trening Modelu')
    uploaded_file = st.file_uploader("Ładuj plik")
    if uploaded_file is not None:
        st.write(uploaded_file.name)
        data_type = Path(uploaded_file.name).suffix
        if data_type not in ['.csv', '.xlsx']:
            st.error('Zły format')
        else:
            st.info(f'OK, plik {data_type}')
            load_data(uploaded_file, data_type=data_type)
            st.write(st.session_state)

    if st.session_state['loaded']:

        train_button = st.button('Trenuj')

        if train_button:
            # st.write('TRENING')
            with st.spinner("Trenowanie..."):
                train_model(st.session_state['df'])
            # st.success('Zakończono')
with tab_data:
    base_c, cluster_c = st.columns(2)
    with base_c:
        st.header('Bazowy DF')
        st.dataframe(st.session_state['df'])
    with cluster_c:
        st.header('DF z klastrami')
        st.dataframe(st.session_state['clustered_df'])
with tab_predict:

    person_df = pd.DataFrame([
            {
                'age': age,
                'edu_level': edu_level,
                'fav_animals': fav_animals,
                'fav_place': fav_place,
                'gender': gender,
            }
        ])
    # st.dataframe(person_df)
    if st.session_state['model']:
        predict_with_clusters_df = predict_model(st.session_state['model'], data=person_df)
        st.dataframe(predict_with_clusters_df)
        your_cluster = predict_with_clusters_df['Cluster'].unique()[0]
        st.info(f"Twój klaster to {str(your_cluster)}")
        with st.expander('Dane Twojego klastra'):
            st.dataframe(st.session_state['clustered_df'].loc[st.session_state['clustered_df']['Cluster'] == your_cluster])
    else:
        st.write('Brak modelu')
    open_ai_key_input = st.text_input('Klucz OpenAI', type='password')

    if open_ai_key_input:
        open_ai_client = get_open_ai_client(open_ai_key_input)
        if open_ai_client:
            st.info('Klient OpenAI gotowy')
        prompt_df = st.session_state['clustered_df'].loc[st.session_state['clustered_df']['Cluster'] == your_cluster]
        system_prompt_text = prepare_prompts(prompt_df)
        select_open_ai_model = st.selectbox('Model OpenAI:', ('gpt-4o-mini', 'gpt-4o'))
        with st.expander('Domyślny prompt systemowy'):
            system_prompt_area = st.text_area('Prompt systemowy:', system_prompt_text)

        # st.write(open_ai_key_input)
        # st.write(str(open_ai_client.api_key))
        user_prompt_area = st.text_area('Zapytanie dodatkowe użytkownika')

        ask_button = st.button('Zapytaj')
        if ask_button:

            with st.spinner('Przetwarzam zapytanie...'):
                ai_response = ask_open_ai(open_ai_client=open_ai_client, system_prompt=system_prompt_area,
                                          user_prompt=user_prompt_area, open_ai_model=select_open_ai_model)
            #     ai_response = ask_open_ai(open_ai_client=open_ai_client, system_prompt=return_text, user_prompt=user_prompt_area)

            st.markdown(ai_response)
            if st.session_state['history'] is None:
                st.session_state['history'] = list()

            st.session_state['history'].append(ai_response)

with tab_history:
    st.title('Historia zapytań')
    # st.write(st.session_state['history'])
    for hist_q in st.session_state['history']:
        st.markdown(hist_q)
        st.write("---")

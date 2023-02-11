import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(layout="wide")




### Titre Sidebar
st.sidebar.title("HapPyWorld Project")
st.sidebar.caption("by Pierre, Loris et Eleonora")

### Image
st.sidebar.image('world-population.png')

### Import et preprocessing 
df = pd.read_csv('world-happiness-report-2021.csv')
df2 = pd.read_csv('world-happiness-report.csv')

### création df_full
#Création liste des régions du monde par pays
country_region = df.iloc[:,0:2]

#Mise en forme des 2 datasets en prévision du concat
#dataset de 2021 :
df_1 = df.iloc[:,:12]
df_1 = df_1.drop(df.iloc[:,3:6], axis=1)
df_1 = df_1.drop('Regional indicator', axis=1)
df_1['year']='2021'

#dataset historique depuis 2005 :
df_2 = df2.iloc[:, :-2]
dico = {'Life Ladder' : 'Ladder score', 'Log GDP per capita' : 'Logged GDP per capita', 'Healthy life expectancy at birth' : 'Healthy life expectancy'}
df_2.rename(dico, axis=1, inplace=True)

#création du fichier contenants les données depuis 2005 :
df_full = pd.concat([df_1, df_2], axis=0)
df_full['year'] = df_full['year'].astype('int')
df_full = df_full.sort_values(by='Country name', ascending=True)

#ajout colonne Region d'après le df country_region créé précédemment :
df_full = df_full.merge(country_region, on = 'Country name', how ='inner')

#ajout du rank par année de chaque pays
df_full["rank"] = df_full.groupby("year")["Ladder score"].rank("dense", ascending=False).astype('int')


#Remplacement des NaN par les moyennes des variables en fonction de chaque pays
for i in df_full.iloc[:,2:8].columns : 
    df_full[i] = df_full[i].fillna(df_full.groupby('Country name')[i].transform("mean"))

#ajout colonne à country-region avec les noms de pays => Espace internes remplacés par '_'
country_region['Country name_2'] = country_region['Country name'].str.replace(' ', '_')

#Liste des pays et regions
#countries = country_region['Country name'].unique()
countries = country_region['Country name'].unique()
countries.sort()
regions = country_region['Regional indicator'].unique()
regions.sort()

## Préparatoin données au SVM
df_full = df_full.sort_values(['year', 'Country name'])
df_full.head()

## variable cile
target_full = df_full['Ladder score']

## df pour le model
col_feat = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']

X_full = df_full[col_feat]
X_train, X_test, y_train, y_test = train_test_split(X_full, target_full, test_size=0.2, random_state=150)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



menu = st.sidebar.radio('HapPyWorld', ('Intro', 'PreProcessing', 'Data Visualisation', 'Focus', 'Modélisation' ))

if menu == 'Intro' :
### sidebar
    st.sidebar.markdown('### Intro')
### Titre
    st.header("HapPyWorld Project")

### Carte du monde
    st.markdown("## Evolution du Happiness Score au fil des ans")
    df_full_by_year = df_full.sort_values('year')

    fig = px.choropleth(df_full_by_year, locations = 'Country name', locationmode = 'country names' ,color = 'Ladder score', projection = 'equirectangular', scope = "world",
                    animation_frame = "year", hover_data=['rank'], hover_name='Country name',range_color=(0, 8))
    fig.update_layout(title = 'Score "bonheur" dans le monde', margin=dict(l=10, r=10, t=30, b=10))

    st.plotly_chart(fig, use_container_width=True)
    with st.container() : 
        st.write("Le world happiness report est une étude annuelle ayant pour objectif d’estimer le bonheur des pays du monde à l’aide de mesures socio-économiques telles que le PIB/habitant, l’espérance de vie, la liberté de choix, la générosité, l’aide sociale ou encore la perception de la corruption.")
        st.write("Le score global de bonheur, ici appelé “Ladder score”, s’obtient grâce à plusieurs variables du dataset, le but étant de déterminer les combinaisons de facteurs permettant d’expliquer pourquoi certains pays voire certaines régions du monde sont mieux classé(e)s que d’autres.")
        st.write("Cette analyse a pour but de comprendre quels sont les facteurs impactant sur le score de bonheur mais aussi de pouvoir prévoir celui-ci à partir des données brutes de chaque pays. La mise en application de différents modèles prédictifs nous permet quant à elle de vérifier si ces modèles peuvent être affectés à l’ensemble des pays ou s’ils ne sont fonctionnels que pour un certain nombre.")

if menu == 'PreProcessing' :
    st.sidebar.markdown("### Preprocessing")

    st.markdown("### Afficher les datasets")
    
    if st.checkbox("Données de 2021") :
        st.dataframe(df)
    
    if st.checkbox("Données antérieures") :
        st.dataframe(df2)
    
    if st.checkbox("Datasets fusionnées") :
        st.dataframe(df_full)
        
if menu == 'Data Visualisation': 
    st.sidebar.markdown("### Data Visualisation")
    
    if st.sidebar.checkbox("2021",value=True) :
    
        #remplacer les colonnes non Explained by
        data_explained = df.drop(df.iloc[:,3:13], axis=1)
        
        
        st.text('Top 10 & last 10 de 2021')
        #Ajout rank dataframe de 2021
        df['rank'] = df['Ladder score'].rank(ascending=False).astype('int')
        
        #retire les colonnes non pertinentes
        data_explained = df.drop(df.iloc[:,3:13], axis=1)
        #Histogrammes horizontaux par pays (Note et répartition des variables) du Top 10 et Last 10
    
        col_var = ['Explained by: Log GDP per capita',
                                          'Explained by: Social support', 
                                          'Explained by: Healthy life expectancy', 
                                          'Explained by: Freedom to make life choices', 
                                          'Explained by: Generosity', 
                                          'Explained by: Perceptions of corruption',
                                          'Dystopia + residual']
        
        data_top10 = data_explained.sort_values(by='rank', ascending=False).tail(10)
        data_last10 = data_explained.sort_values(by='rank', ascending=False).head(10)
        
        ax1 = data_top10.plot.barh(x = 'Country name', 
                                      y = col_var,
                                       stacked = True,
                                       figsize = (10,6),
                                       grid = True,
                                       xticks = range(0,10),
                                       title = 'TOP 10 de 2021',
                                       legend = False
                                  )
        st.pyplot()
        
        ax2 = data_last10.plot.barh(x = 'Country name', 
                                      y = col_var,
                                       stacked = True,
                                       figsize = (10,6),
                                       grid = True,
                                       xticks = range(0,10),
                                       title = "LAST 10 de 2021")
        ax2.legend(bbox_to_anchor=(1.0, 1.0));
        st.pyplot()
        
        data_1 = data_explained.groupby('Regional indicator').mean()
        data_1['Region'] = data_1.index
        
        data_1 = data_1.sort_values(by='rank', ascending=False).tail(15)
        
        ax = data_1.plot.barh(x = 'Region', 
                                  y = ['Explained by: Log GDP per capita',
                                      'Explained by: Social support', 
                                      'Explained by: Healthy life expectancy', 
                                      'Explained by: Freedom to make life choices', 
                                      'Explained by: Generosity', 
                                      'Explained by: Perceptions of corruption',
                                      'Dystopia + residual'],
                                   stacked = True,
                                   figsize = (10,6),
                                   xticks = range(0,10),
                                   grid = True
                                )
    
        #ax.legend(bbox_to_anchor=(1.0, 0.0));
        st.pyplot()
        
        st.text('Classement 2021 par région')
        
        
        #Boxplots variables
        
       
        st.markdown('## Dispersion des variables explicatives pour 2021')
            
        plt.figure(figsize=(15, 3))
    
        plt.boxplot([df['Explained by: Log GDP per capita'], df['Explained by: Social support'], df['Explained by: Healthy life expectancy'], df['Explained by: Freedom to make life choices'], df['Explained by: Generosity'], df['Explained by: Perceptions of corruption']], labels=['Log GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption' ])
        plt.grid(alpha=0.5)
        plt.title("Boxplots variables des 149 pays de l'étude pour 2021");
        st.pyplot()
            
        expander_1 = st.expander(label= 'Affcicher la dispersion par Région du Monde')    
       
        with expander_1 :
            plt.figure(figsize=(20,50))
            
            plt.subplot(8,2,1)
            ax = sns.boxplot(x='Ladder score', y='Regional indicator', orient='h', data=data_explained, palette='Set2')
            ax.set_yticklabels(ax.get_yticklabels(),fontsize = 15)
            ax.set_title('Score', fontsize = 18)
            ax.set(xlabel = None, ylabel = None)
            
            plt.subplot(8,2,2)
            ax = sns.boxplot(x='Explained by: Log GDP per capita', y='Regional indicator', data=data_explained, orient='h', palette='Set2')
            ax.set(yticklabels = [])
            ax.set(xlabel = None, ylabel = None)
            ax.set_title('Log GDP per capita', fontsize = 18)
            
            plt.subplot(8,2,3)
            ax = sns.boxplot(x='Explained by: Social support', y='Regional indicator', data=data_explained, orient='h', palette='Set2')
            ax.set_yticklabels(ax.get_yticklabels(),fontsize = 15)
            ax.set(xlabel = None, ylabel = None)
            ax.set_title('Social Support', fontsize = 18)
            
            plt.subplot(8,2,4)
            ax = sns.boxplot(x='Explained by: Healthy life expectancy', y='Regional indicator', data=data_explained, orient='h', palette='Set2')
            ax.set(yticklabels = [])
            ax.set(xlabel = None, ylabel = None)
            ax.set_title('Healthy life expectancy', fontsize = 18)
            
            plt.subplot(8,2,5)
            ax = sns.boxplot(x='Explained by: Freedom to make life choices', y='Regional indicator', data=data_explained, orient='h', palette='Set2')
            ax.set_yticklabels(ax.get_yticklabels(),fontsize = 15)
            ax.set(xlabel = None, ylabel = None)
            ax.set_title('Freedom to make life choices', fontsize = 18)
            
            plt.subplot(8,2,6)
            ax = sns.boxplot(x='Explained by: Generosity', y='Regional indicator', data=data_explained, orient='h', palette='Set2')
            ax.set(yticklabels = [])
            ax.set(xlabel = None, ylabel = None)
            ax.set_title('Generosity', fontsize = 18)
            
            plt.subplot(8,2,7)
            ax = sns.boxplot(x='Explained by: Perceptions of corruption', y='Regional indicator', data=data_explained, orient='h', palette='Set2')
            ax.set_yticklabels(ax.get_yticklabels(),fontsize = 15)
            ax.set(xlabel = None, ylabel = None)
            ax.set_title('Perceptions of corruption', fontsize = 18)
            
            plt.subplot(8,2,8)
            ax = sns.boxplot(x='Dystopia + residual', y='Regional indicator', data=data_explained, orient='h', palette='Set2')
            ax.set(yticklabels = [])
            ax.set(xlabel = None, ylabel = None)
            ax.set_title('Dystopia + residual', fontsize = 18);
            
            st.pyplot()
    
                
        ### Pie plot pays par région
        st.text('Proportion / poids des régions')
        
        fig = px.pie(values = df.groupby('Regional indicator').count()['Country name'].sort_values().values, names = df.groupby('Regional indicator').count()
        ['Country name'].sort_values().index, title = 'Répartition des pays par "Regional indicator"', 
        color_discrete_map = {'Sub-Saharan Africa':'#636EFA',
                        'Western Europe':'EF553B',
                        'Middle East and North Africa':'19D3F3',
                        'Latin America and Caribbean':'FECB52',
                        'Commonwealth of Independent States':'FB0D0D',
                        'Southeast Asie':'1CFFCE',
                        'South Asia':'0D2A63',
                        'East Asia':'7F7F7F',
                        'North America and ANZ':'AF0038'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap corrélation variables
        st.markdown('### Heatmap - Corrélation entre les variables')
        plt.figure(figsize=(15,15))
        num_var = df.drop(['Country name',"Regional indicator"], axis= 1)
        sns.heatmap(data = num_var.corr(), annot=True, cmap ="winter");
        st.pyplot()
        
        expander_2 = st.expander(label= 'Afficher la corrélation par Région du Monde')    
       
        with expander_2 :
            
            data_corr =pd.DataFrame(df[['Regional indicator','Logged GDP per capita', 'Social support', 'Healthy life expectancy',
               'Freedom to make life choices', 'Generosity',
               'Perceptions of corruption','Dystopia + residual']])
        
        
            corr_data_exp= data_corr.corr()
        
            plt.subplots(figsize=(20, 50))
        
            j=1
        
            for i in range(len(data_corr["Regional indicator"].unique())):
          
                region = data_corr["Regional indicator"].unique()[i]
          
                data_map = data_corr[data_corr["Regional indicator"] == region]
            
                matrix = np.triu(corr_data_exp)
                corr_data_exp= data_map.corr()
                plt.subplot(5,2,j)
                sns.heatmap(corr_data_exp, annot=True, cmap='winter', mask=matrix)
                plt.subplots_adjust(hspace=0.7)
                plt.title(region, position=(0.5, -10),fontdict={'family': 'serif', 'color' : 'darkred','weight': 'bold','size': 16});
                j += 1
     
              
            st.pyplot()
     
    if st.sidebar.checkbox('Données cumulées de 2005 à 2021') :          
        
        st.text('Top 3 de 2005 à 2021')
        #Pays ayant été dans le top 3 depuis 2005
        df_full_top3 = df_full[df_full['rank'] <= 3].sort_values(by='year')
        
        #Pays ayant été 1ers depuis 2005
        df_full_top = df_full[df_full['rank'] == 1].sort_values(by='year')
        
        ax = sns.relplot(x='year', y='Country name', hue='rank', size= 'Ladder score', height=5, aspect = 2, palette = 'tab10', data=df_full_top3)
        st.pyplot()
        
        st.text('Evolution du ladder score et des variables explicatives des 5 pays ayant été Top1')
        # les 5 pays qui se disputent la 1ère place depuis 2005
    
        first_countries = ['Denmark', 'Finland', 'Switzerland', 'Canada', 'Norway']
        
        df_full_first_countries = df_full[df_full['Country name'].isin(first_countries)]
        liste_var = ('Ladder score','Logged GDP per capita', 'Social support','Healthy life expectancy','Freedom to make life choices', 'Generosity', 'Perceptions of corruption')
    
        for i in liste_var :
            sns.relplot(x='year', y=i, kind='line',hue='Country name', style='Country name', height=4, aspect = 2, data = df_full_first_countries).set(title=(i + ' des 5 pays Top1'));
            st.pyplot()

if menu =='Focus' :
    
    st.sidebar.markdown("### Focus")
       
        
    st.sidebar.markdown("Régions du monde") 
        
    col_region, col_annee = st.columns(2)
    
    #choix pays
    with col_region :
        
        Region = st.selectbox('Région', regions)
        
    
    # choix année
    with col_annee :
        min_ts = min(df_full['year'])
        max_ts = max(df_full['year'])
        year_sel = st.slider("Année", min_value=min_ts, max_value=max_ts, value=max_ts)
    
    #création DataSet avec données du pays et année sélectionnés
    df_full_filter = df_full[(df_full['year'] == year_sel) & (df_full['Regional indicator'] == Region)]

        
    #set colonnes
    col1, col2 = st.columns(2)

    col1.header("Données")
    
    with col1: 
        df_full_filter.loc[:,['Country name', 'Ladder score', 'rank']]
        
        
    with col2 : 
        #carte monde

        fig = px.choropleth(df_full_filter, locations = 'Country name', locationmode = 'country names' ,color = 'Ladder score', projection = 'natural earth', scope = "world", hover_data=['rank', 'year'], hover_name='Country name',range_color=(0, 8))
    
        fig.update_geos(fitbounds="locations")
        fig.update_layout(title = Region, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig)
        scatter = fig.data[0]
        
                
    with col1:    
        Pays = st.selectbox('Pays', df_full_filter['Country name'])
        #df_full_filter_country = df_full_filter[(df_full_filter['year'] == year_sel) & (df_full_filter['Country name'] == Pays)]
        #df_full_filter_country = df_full_filter_country.drop(df_full_filter_country.columns[0], axis=1)
        
    df_full_country = df_full[df_full['Country name'] == Pays]
    df_full_country = df_full_country.sort_values(by='year')
    df_full_country = df_full_country.drop(df_full_country.loc[:,['Country name', 'Regional indicator']], axis=1)     
    df_full_country.set_index('year', inplace=True)
    
    st.write(df_full_country.transpose())
    

    

### MODELISATION -----------------------------------------------------------------------------------------------------
if menu == 'Modélisation' :
    
    st.sidebar.markdown("### Modélisation")
    st.sidebar.image('ML_ampoule.png')
    
    st.write("## Régression linéaire")
    
    if st.button("RL") :
        col_rl1, col_rl2 = st.columns(2)
        
        
        model = LinearRegression()
    
        model.fit(X_train_scaled, y_train)
        
        with col_rl1 :
            st.write('Coef de détermination du modèle :', round(model.score(X_train_scaled, y_train),2))
            st.write('Coef de détermination obtenu par cv :', round(cross_val_score(model, X_train_scaled, y_train).mean(),2))
            st.write('Score test :', round(model.score(X_test_scaled, y_test),2))
        
        with col_rl2 :
            pred_test = model.predict(X_test_scaled)
            plt.scatter(pred_test, y_test)
            plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()),c='r');
            
            st.pyplot()
        
        
        sk = SelectKBest(f_regression, k=3)
    
        sk.fit(X_train, y_train)
        
        with col_rl1:
            st.write("### Recherche des variables significatives avec SelectKBest")
            significant_feat = X_full.columns[sk.get_support()]
            
            st.write("Significant features :", significant_feat)
            
            model.fit(X_train[significant_feat], y_train)
            
            st.write("#### RL entraînée avec les variables significatives :")
            st.write('score train :', round(model.score(X_train[significant_feat], y_train),2))
            st.write('score test :', round(model.score(X_test[significant_feat], y_test),2))
    
    
## SVM  
    st.write("## Support Vector Machine")
    
    ## SVM
    if st.button("SVM") :
    
        #recherche best params
                
        BestParams = {'C':5, 'gamma': 0.5, 'kernel':'rbf'}
        
        
        col3, col4 = st.columns(2)
        
        with col3 :
            if st.button("Recherche des best_params_ (attention long process)") :
                model = SVR()
                parametres = {'C':[1,5,10,25,50,100], 'kernel':['rbf', 'linear'], 'gamma' : [0.001,0.1,0.5,1,10,100]}
                grid_clf = model_selection.GridSearchCV(estimator=model, param_grid=parametres)
                grille = grid_clf.fit(X_train_scaled, y_train)
                BestParams = grid_clf.best_params_
                
        
            st.write("Paramètres renvoyés par best\_params\_ :", BestParams)
                                  
            model = SVR(**BestParams)
            
            model.fit(X_train_scaled, y_train)
        
            st.write('Coef de détermination du modèle :', round(model.score(X_train_scaled, y_train),2))
            st.write('Coef de détermination obtenu par cv :', round(cross_val_score(model, X_train_scaled, y_train).mean(),2))
            st.write('Score test :', round(model.score(X_test_scaled, y_test),2))
            
            pred_test = model.predict(X_test_scaled)
            pred_train = model.predict(X_train_scaled)
                
        
        with col4 :
            plt.scatter(pred_test, y_test)
            plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c='r');
            st.pyplot()
        
        
        exp_param = st.expander(label='Modification des Hyperparamètres')
        with exp_param :
           
            col_1_param, col_2_param = st.columns(2)
            
            with col_1_param :
                C_slider = st.slider('Paramètre C', 1,100,5)
                Kernel_slider = st.select_slider('Kernel', options=['rbf', 'linear', 'poly'])
                Gamma_slider = st.slider('Gamma', 0.001, 100.000, 0.500)
            
            with col_2_param :
                
                model = SVR(C=C_slider, kernel = Kernel_slider, gamma = Gamma_slider)
                model.fit(X_train_scaled, y_train)
                st.write('Coef de détermination du modèle :', round(model.score(X_train_scaled, y_train),2))
                st.write('Coef de détermination obtenu par cv :', round(cross_val_score(model, X_train_scaled, y_train).mean(),2))
                st.write('Score test :', round(model.score(X_test_scaled, y_test),2))
            
                pred_test = model.predict(X_test_scaled)
                pred_train = model.predict(X_train_scaled)
                plt.scatter(pred_test, y_test)
                plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c='r');
                st.pyplot()
            
        col5, col6 = st.columns((1,2))        
        with col5 :
           
            # Permutation importance
            st.write("### Permutation Importance")
            Y = y_train
            X = X_train_scaled
            
            perm_importance = permutation_importance(model, X, Y, n_repeats=20, random_state=0)
            
            features = np.array(col_feat)
            
            sorted_idx = perm_importance.importances_mean.argsort()
            #plt.figure(figsize=(10,5))
            plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
            plt.xlabel("Permutation Importance");
            st.pyplot()
                
        with col6 :
               
            Regions = country_region['Regional indicator'].unique()
           
            expander_permut_region = st.expander(label='Permutation Importance par Région du Monde')
            with expander_permut_region :    
                        
                i=1
                for reg in Regions :
                            
                    Region = df_full[df_full['Regional indicator'] == reg]
            
                    name_region = Region['Regional indicator'][:1].values
                    X_region = Region[col_feat]
                    y_region = Region['Ladder score']
            
                    X_region_scaled = scaler.transform(X_region)
            
                    model.fit(X_region_scaled, y_region)
            
                    perm_importance = permutation_importance(model, X_region_scaled, y_region, n_repeats=10, random_state=0)
            
                    features = np.array(col_feat)
            
                    sorted_idx = perm_importance.importances_mean.argsort()
                    plt.figure(figsize=(20,7))
                    plt.subplot(5,2,i)
                    ax=sns.barplot(x = perm_importance.importances_mean[sorted_idx],y = features[sorted_idx] , palette = "Blues", orient='h')
                    ax.set_xlim(left=0, right=1)
                    #plt.xlabel("Permutation Importance")
                    plt.title(name_region)
                    i += 1
                    st.pyplot()
            
    
        st.write("## Fonctionnement du modèle par Région du Monde")    
        st.write("### Moyennes par région du monde / différence entre la moyenne du Ladder score et du score prédit")
    
           
        df_scores_test = pd.DataFrame({'Ladder score' : y_test, 'Score prédit' : pred_test}, index = y_test.index)
        df_scores_train = pd.DataFrame({'Ladder score' : y_train, 'Score prédit' : pred_train}, index = y_train.index)
        df_scores = pd.concat([df_scores_test, df_scores_train], axis=0)
        df_scores.drop('Ladder score', axis=1, inplace=True)
        df_full_score = df_full.merge(df_scores, how='left', left_index=True, right_index=True)
        df_full_score['diff_scores'] = abs(df_full_score['Score prédit'] - df_full_score['Ladder score'])
        df_full_score.sort_values(by='diff_scores')
        regions_group = df_full_score.groupby(by='Regional indicator').agg(
            Ladder_score_mean = ('Ladder score', 'mean'), 
            GDP_mean = ('Logged GDP per capita', 'mean'),
            Social_support_mean = ('Social support', 'mean'),
            Healthy_life_mean = ('Healthy life expectancy', 'mean'),
            Life_choices_mean = ('Freedom to make life choices', 'mean'),
            Generosity_mean = ('Generosity', 'mean'),
            Perc_corruption_mean = ('Perceptions of corruption', 'mean'),
            rank_min = ('rank', 'min'),
            rank_max = ('rank', 'max'),
            Score_predit_mean = ('Score prédit', 'mean'),
            nb_pays = ('Country name', 'nunique')
            ) 
    
        regions_group['diff_moy_reg'] = abs(regions_group['Score_predit_mean'] - regions_group['Ladder_score_mean'])
        
        regions_group.sort_values(by='diff_moy_reg', inplace=True)
        
        st.dataframe(regions_group)

        regions_group = regions_group.sort_values(by='diff_moy_reg')
        bar_width = 0.5
        fig = plt.figure(figsize=(15,8))
        
        plt.bar(regions_group.index, regions_group['Score_predit_mean'], label='Moyenne Ladder score prédit', alpha=0.5)
        plt.bar(regions_group.index, regions_group['Ladder_score_mean'], label='Moyenne Ladder score', width = bar_width)
        plt.xticks(rotation=30, ha='right')
        plt.grid(alpha=0.5)
        plt.legend();
        st.pyplot()            
        
            
            
            

st.sidebar.markdown('## Participants :')
st.sidebar.write('[Pierre Thomas](https://www.linkedin.com/in/pierre12-thomas)')
st.sidebar.write('[Loris Sedeaud](https://www.linkedin.com/in/loris-sedeaud/)')
st.sidebar.write("[Eleonora Fabris](https://www.linkedin.com/in/eleonora-fabris-4147366a/)")

### Image
st.sidebar.image('DS_logo.png')
st.sidebar.write('Projet réalisé dans le cadre de la formation Data Analyst de ', "[DataScientest](https://datascientest.com/)")
st.sidebar.markdown('Session bootcamp Août 2021')

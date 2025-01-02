import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(page_title="Prédiction de la Texture du Sol", layout="wide")

# Styles CSS
st.markdown("""
<style>
    .main { padding: 2rem; }
    .title-container {
        background: linear-gradient(90deg, #2E7D32 0%, #388E3C 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .result-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2E7D32 0%, #388E3C 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Dictionnaire des textures
texture_translations = {
    'Clay loam': 'Limon argileux - Clay loam',
    'Loam': 'Limon - Loam',
    'Sandy loam': 'Limon sableux - Sandy loam',
    'Loamy sand': 'Sable limoneux - Loamy sand',
    'Clay': 'Argile - Clay',
    'Sandy clay loam': 'Limon argilo-sableux - Sandy clay loam'
}

# Recommandations culturales
cultures_recommendations = {
    'Limon argileux - Clay loam': {
        'principales_cultures': 'Cacao, Palmier à huile, Hévéa',
        'cultures_vivrières': 'Banane plantain, Igname, Taro',
        'cultures_maraicheres': 'Tomate, Piment, Gombo, Aubergine, Poivron',
        'zones': 'Sud, Est, Centre-Ouest',
        'irrigation': 'Modérée',
        'fertilisation': 'Modérée à forte',
        'precautions': 'Drainage en saison pluvieuse, surveillance compaction'
    },
    'Limon - Loam': {
        'principales_cultures': 'Anacarde, Coton, Maïs',
        'cultures_vivrières': 'Manioc, Igname, Arachide',
        'cultures_maraicheres': 'Tomate, Piment, Choux, Concombre, Laitue, Carotte',
        'zones': 'Centre, Nord',
        'irrigation': 'Moyenne',
        'fertilisation': 'Modérée',
        'precautions': 'Maintien matière organique, rotation cultures'
    },
    'Limon sableux - Sandy loam': {
        'principales_cultures': 'Ananas, Manioc, Arachide',
        'cultures_vivrières': 'Patate douce, Maraîchage',
        'cultures_maraicheres': 'Oignon, Poireau, Aubergine, Piment, Tomate cerise',
        'zones': 'Centre, Sud-Est',
        'irrigation': 'Fréquente',
        'fertilisation': 'Régulière',
        'precautions': 'Irrigation en saison sèche, enrichissement organique'
    },
    'Sable limoneux - Loamy sand': {
        'principales_cultures': 'Manioc, Ananas',
        'cultures_vivrières': 'Arachide, Pastèque',
        'cultures_maraicheres': 'Melon, Pastèque, Concombre, Courgette',
        'zones': 'Littoral, Sud',
        'irrigation': 'Très fréquente',
        'fertilisation': 'Forte',
        'precautions': 'Irrigation obligatoire, enrichissement organique constant'
    },
    'Argile - Clay': {
        'principales_cultures': 'Riz irrigué, Palmier à huile',
        'cultures_vivrières': 'Taro, Banane plantain',
        'cultures_maraicheres': 'Choux, Epinard, Basilic, Persil, Céleri',
        'zones': 'Bas-fonds, Zones humides',
        'irrigation': 'Faible',
        'fertilisation': 'Modérée',
        'precautions': 'Drainage crucial, travail sol adapté'
    },
    'Limon argilo-sableux - Sandy clay loam': {
        'principales_cultures': 'Cacao, Hévéa, Anacarde',
        'cultures_vivrières': 'Maïs, Igname, Manioc',
        'cultures_maraicheres': 'Tomate, Piment, Aubergine, Gombo, Poivron, Concombre',
        'zones': 'Centre-Ouest, Sud-Ouest',
        'irrigation': 'Modérée',
        'fertilisation': 'Modérée',
        'precautions': 'Équilibre drainage-rétention'
    }
}

@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv("texture.csv", sep=';')
        df['Soil Texture'] = df['Soil Texture'].map(texture_translations)
        
        X = df[['Argile(%)', 'Limon(%)', 'Sable(%)']]
        y = df['Soil Texture']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler
    except Exception as e:
        st.error(f"Erreur : {str(e)}")
        return None, None

# Interface principale
st.markdown("""
    <div class='title-container'>
        <h1>🌱 Prédiction de la Texture du Sol</h1>
        <p style='font-size: 1.2em;'>Analyse et Recommandations Culturales pour la Côte d'Ivoire</p>
    </div>
""", unsafe_allow_html=True)

model, scaler = load_and_prepare_data()

if model and scaler:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Entrée des données")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        argile = st.number_input("🧱 Argile (%)", 0, 100, 25)
    with col2:
        limon = st.number_input("🏺 Limon (%)", 0, 100, 30)
    with col3:
        sable = st.number_input("🏖️ Sable (%)", 0, 100, 45)

    total = argile + limon + sable
    
    # Barre de progression pour le total
    progress_color = 'normal' if total != 100 else 'success'
    st.progress(total/100, text=f"Total : {total}%")
    
    if total != 100:
        st.warning(f"⚠️ Total : {total}% (doit être égal à 100%)")
    
    if st.button("🔍 Prédire la texture"):
        if total == 100:
            input_data = np.array([[argile, limon, sable]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            proba = max(model.predict_proba(input_scaled)[0]) * 100
            
            # Affichage des résultats
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("### 🎯 Résultats de l'analyse")
                st.metric("Texture prédite", prediction)
                st.metric("Niveau de confiance", f"{proba:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Recommandations
            if prediction in cultures_recommendations:
                rec = cultures_recommendations[prediction]
                
                with col2:
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown("### 🌱 Recommandations Culturales")
                    
                    tab1, tab2 = st.tabs(["Cultures", "Gestion"])
                    
                    with tab1:
                        st.markdown("#### 🏆 Cultures Principales")
                        st.success(rec['principales_cultures'])
                        
                        st.markdown("#### 🍎 Cultures Vivrières")
                        st.success(rec['cultures_vivrières'])
                        
                        st.markdown("#### 🥬 Cultures Maraîchères")
                        st.success(rec['cultures_maraicheres'])
                        
                        st.markdown("#### 📍 Zones Favorables")
                        st.info(rec['zones'])
                    
                    with tab2:
                        st.markdown("#### ⚠️ Points d'Attention")
                        st.write(f"- Irrigation : {rec['irrigation']}")
                        st.write(f"- Fertilisation : {rec['fertilisation']}")
                        st.write(f"- Précautions : {rec['precautions']}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("⚠️ Ajustez les pourcentages pour obtenir un total de 100%")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Guide d'utilisation
    with st.expander("📖 Guide d'utilisation"):
        st.markdown("""
        1. Entrez les pourcentages d'argile, limon et sable de votre sol
        2. Vérifiez que le total fait 100%
        3. Cliquez sur 'Prédire la texture'
        4. Consultez les recommandations adaptées à votre sol
        """)

# Pied de page
st.markdown("""
    <div style='text-align: center; padding: 1rem; color: #666;'>
        <p>Application développée pour l'analyse des sols en Côte d'Ivoire</p>
    </div>
""", unsafe_allow_html=True)
"""
🎯 PROJET JOB INTELLIGENT - Système de Recommandation d'Emplois
Interface Streamlit pour la recommandation d'emplois basée sur TF-IDF et similarité cosinus

👥 Auteurs:
   - Mohamed Sabbar 
   - Lamadi Youssef 
   - Mohammed Rida Boukich 
   - Abdelhafid Kbiri Alaoui 

📝 Description:
   Application interactive pour trouver les emplois les plus pertinents
   selon votre profil et vos préférences. Utilise un modèle TF-IDF
   pré-calculé pour des recommandations ultra-rapides (< 2 secondes).

🔧 Technologies:
   - Streamlit: Interface web interactive
   - Scikit-learn: Machine Learning & TF-IDF
   - Pandas/NumPy: Manipulation de données
   - LinkedIn Data: 553K+ offres d'emploi

⚡ Performance: Temps de chargement < 2 secondes
💾 Mémoire: ~500MB (modèle optimisé)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# ============================================================================
# CONFIGURATION STREAMLIT
# ============================================================================
# Configuration de la page - paramètres visuels et comportementaux
st.set_page_config(
    page_title="🎯 Job Intelligent - Recommandation d'Emplois",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .job-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border-left: 6px solid #ffd700;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white !important;
    }
    .job-card h3 {
        color: #ffd700 !important;
        margin-bottom: 15px;
        font-weight: bold;
    }
    .job-card p {
        color: white !important;
        font-size: 1.05rem;
        margin: 8px 0;
        line-height: 1.6;
    }
    .job-card strong {
        color: #ffd700 !important;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    /* Améliorer la lisibilité des expanders */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_model_and_data():
    """
    Charge le modèle TF-IDF pré-calculé et les données (ULTRA RAPIDE ⚡)
    
    Cette fonction utilise le cache Streamlit (@st.cache_data) pour charger
    le modèle une seule fois en mémoire, garantissant des performances
    maximales lors des appels ultérieurs.
    
    Returns:
        tuple: (vectorizer, tfidf_matrix, jobs_df, metadata)
            - vectorizer: Modèle TF-IDF entraîné
            - tfidf_matrix: Matrice TF-IDF pré-calculée
            - jobs_df: DataFrame avec les données d'emploi
            - metadata: Métadonnées du modèle (date création, stats, etc.)
    
    Raises:
        FileNotFoundError: Si le dossier 'model' n'existe pas
        pickle.UnpicklingError: Si un fichier .pkl est corrompu
    """
    
    MODEL_DIR = "model"
    
    # Vérifier si le modèle existe
    if not os.path.exists(MODEL_DIR):
        st.error("""
        ❌ **Modèle non trouvé !**
        
        Vous devez d'abord préparer le modèle en exécutant :
        ```
        python prepare_model.py
        ```
        
        Cela créera un modèle pré-calculé pour des performances optimales.
        """)
        st.stop()
    
    try:
        # Charger les métadonnées
        with open(os.path.join(MODEL_DIR, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
        
        # Charger le vectorizer
        with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Charger la matrice TF-IDF
        with open(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"), 'rb') as f:
            tfidf_matrix = pickle.load(f)
        
        # Charger les données des emplois
        with open(os.path.join(MODEL_DIR, "jobs_data.pkl"), 'rb') as f:
            jobs_df = pickle.load(f)
        
        return vectorizer, tfidf_matrix, jobs_df, metadata
    
    except Exception as e:
        st.error(f"""
        ❌ **Erreur lors du chargement du modèle :** {str(e)}
        
        Essayez de régénérer le modèle :
        ```
        python prepare_model.py
        ```
        """)
        st.stop()


def recommend_jobs(profile_text, jobs_df, vectorizer, tfidf_matrix, 
                   n_recommendations=10, location_filter=None, 
                   min_salary=0, experience_filter=None, work_type_filter=None,
                   remote_only=False):
    """
    Recommande les emplois les plus pertinents pour un profil donné
    
    Utilise le modèle TF-IDF et la similarité cosinus pour scorer
    chaque emploi par rapport au profil de l'utilisateur.
    
    Args:
        profile_text (str): Description du profil candidat
        jobs_df (pd.DataFrame): DataFrame des emplois
        vectorizer: Modèle TF-IDF pré-entraîné
        tfidf_matrix: Matrice TF-IDF pré-calculée
        n_recommendations (int): Nombre d'emplois à retourner
        location_filter (str): Filtrer par localisation
        min_salary (float): Salaire minimum
        experience_filter (str): Niveau d'expérience
        work_type_filter (str): Type de contrat
        remote_only (bool): Retourner seulement les postes en remote
    
    Returns:
        pd.DataFrame: Emplois recommandés triés par score de similarité
    """
    
    # Transformer le profil en vecteur TF-IDF
    profile_vector = vectorizer.transform([profile_text])
    
    # Calculer la similarité cosinus
    similarities = cosine_similarity(profile_vector, tfidf_matrix).flatten()
    
    # Créer un dataframe avec les scores
    jobs_df = jobs_df.copy()
    jobs_df['similarity_score'] = similarities
    
    # Appliquer les filtres
    filtered_jobs = jobs_df.copy()
    
    # Filtre remote
    if remote_only and 'remote_allowed' in filtered_jobs.columns:
        filtered_jobs = filtered_jobs[filtered_jobs['remote_allowed'] == 1.0]
    
    if location_filter and location_filter != "Tous":
        # Filtre localisation avec vérification des colonnes
        location_mask = pd.Series([False] * len(filtered_jobs), index=filtered_jobs.index)
        if 'location' in filtered_jobs.columns:
            location_mask |= filtered_jobs['location'].str.contains(location_filter, case=False, na=False)
        if 'state' in filtered_jobs.columns:
            location_mask |= filtered_jobs['state'].str.contains(location_filter, case=False, na=False)
        if 'city' in filtered_jobs.columns:
            location_mask |= filtered_jobs['city'].str.contains(location_filter, case=False, na=False)
        filtered_jobs = filtered_jobs[location_mask]
    
    if min_salary > 0 and 'med_salary' in filtered_jobs.columns:
        filtered_jobs = filtered_jobs[filtered_jobs['med_salary'] >= min_salary]
    
    if experience_filter and experience_filter != "Tous" and 'formatted_experience_level' in filtered_jobs.columns:
        filtered_jobs = filtered_jobs[
            filtered_jobs['formatted_experience_level'].str.contains(experience_filter, case=False, na=False)
        ]
    
    if work_type_filter and work_type_filter != "Tous" and 'formatted_work_type' in filtered_jobs.columns:
        filtered_jobs = filtered_jobs[
            filtered_jobs['formatted_work_type'].str.contains(work_type_filter, case=False, na=False)
        ]
    
    # Trier par score de similarité
    recommendations = filtered_jobs.nlargest(n_recommendations, 'similarity_score')
    
    return recommendations


def display_job_card(job, rank):
    """
    Affiche une carte d'emploi stylisée avec tous les détails
    
    Crée une représentation visuelle attrayante d'une offre d'emploi
    avec tous les détails pertinents et expanders pour les infos complémentaires.
    
    Args:
        job (pd.Series): Ligne du DataFrame contenant les données d'emploi
        rank (int): Numéro du classement (pour l'affichage)
    
    Visual Elements:
        - Titre avec numéro et badge remote si applicable
        - Entreprise, localisation, salaire
        - Niveau d'expérience et type de contrat
        - Score de correspondance en pourcentage
        - Expanders pour description et compétences
    """
    
    similarity_pct = job.get('similarity_score', 0) * 100
    salary = job.get('med_salary', 0)
    salary_display = f"${salary:,.0f}" if salary > 0 else "Non spécifié"
    
    # Badge Remote
    remote_badge = ""
    if job.get('remote_allowed') == 1.0:
        remote_badge = " 🌍 <span style='background-color: #10b981; color: white; padding: 2px 8px; border-radius: 5px; font-size: 0.85rem;'>REMOTE</span>"
    
    with st.container():
        st.markdown(f"""
        <div class="job-card">
            <h3>#{rank} - {job.get('title', 'Titre non disponible')}{remote_badge}</h3>
            <p><strong>🏢 Entreprise:</strong> {job.get('company_name', 'Non spécifié')}</p>
            <p><strong>📍 Localisation:</strong> {job.get('location', 'Non spécifié')}</p>
            <p><strong>💰 Salaire médian:</strong> {salary_display}</p>
            <p><strong>📊 Expérience:</strong> {job.get('formatted_experience_level', 'Non spécifié')}</p>
            <p><strong>💼 Type:</strong> {job.get('formatted_work_type', 'Non spécifié')}</p>
            <p><strong>🎯 Score de correspondance:</strong> {similarity_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Afficher la description si disponible
        description = job.get('description', '')
        if description and str(description).strip():
            with st.expander("📝 Voir la description"):
                desc_text = str(description)
                st.write(desc_text[:1000] + "..." if len(desc_text) > 1000 else desc_text)
        
        # Afficher les compétences si disponibles
        skills_text = job.get('skills_text', '')
        if skills_text and str(skills_text).strip():
            with st.expander("🔧 Compétences requises"):
                skills = str(skills_text).split()[:15]
                st.write(" • ".join(skills))


def main():
    """
    Fonction principale de l'application Streamlit
    
    Orchestrate l'ensemble de l'application :
    1. Charge le modèle pré-calculé
    2. Affiche le header et les infos du modèle
    3. Gère les filtres et la recherche
    4. Affiche les résultats et les cartes d'emploi
    5. Permet l'export des résultats
    
    Flow:
        Load Data → Display UI → User Input → Filter & Recommend → Display Results
    """
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Job Intelligent - Système de Recommandation</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Charger les données
    try:
        vectorizer, tfidf_matrix, jobs_df, metadata = load_model_and_data()
        
        # Afficher les infos du modèle
        st.success(f"✅ Modèle chargé instantanément ! {len(jobs_df):,} offres disponibles")
        
        with st.expander("ℹ️ Informations sur le modèle"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Emplois indexés", f"{metadata['n_jobs']:,}")
            with col_b:
                st.metric("Vocabulaire", f"{metadata['vocabulary_size']:,}")
            with col_c:
                created = metadata['created_at'].split('T')[0]
                st.metric("Créé le", created)
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données: {str(e)}")
        st.info("Assurez-vous que les fichiers CSV sont dans le dossier 'dataset/'")
        return
    
    # Sidebar - Filtres
    st.sidebar.header("🔍 Filtres de Recherche")
    
    # Profils prédéfinis
    st.sidebar.subheader("📋 Profils Prédéfinis")
    profile_presets = {
        "Personnalisé": "",
        "Data Scientist": "data scientist machine learning python deep learning tensorflow pytorch statistics modeling predictive analytics neural networks",
        "Data Analyst": "data analyst SQL Excel visualization Tableau Power BI reporting analytics business intelligence dashboard KPI metrics",
        "Data Engineer": "data engineer ETL pipeline Spark Hadoop SQL Python cloud AWS Azure data warehouse big data processing streaming",
        "ML Engineer": "machine learning engineer MLOps deployment model training optimization Python scikit-learn production infrastructure",
        "Business Analyst": "business analyst requirements analysis stakeholder management documentation process improvement project management"
    }
    
    selected_preset = st.sidebar.selectbox("Choisir un profil prédéfini", list(profile_presets.keys()))
    
    # Filtres avancés
    st.sidebar.subheader("⚙️ Filtres Avancés")
    
    # Filtre Remote
    remote_only = st.sidebar.checkbox("🌍 Postes en remote uniquement", value=False)
    
    # Filtre localisation - vérifier si la colonne existe
    locations = ["Tous"]
    if 'state' in jobs_df.columns:
        state_list = list(jobs_df['state'].dropna().unique())
        locations.extend(state_list[:20])
    location_filter = st.sidebar.selectbox("📍 État/Région", locations)
    
    # Filtre salaire minimum - vérifier si la colonne existe
    min_salary = 0
    if 'med_salary' in jobs_df.columns:
        min_salary = st.sidebar.slider("💰 Salaire minimum ($)", 0, 200000, 0, 10000)
    
    # Filtre niveau d'expérience - vérifier si la colonne existe
    experience_levels = ["Tous"]
    if 'formatted_experience_level' in jobs_df.columns:
        exp_list = list(jobs_df['formatted_experience_level'].dropna().unique())
        experience_levels.extend(exp_list)
    experience_filter = st.sidebar.selectbox("📊 Niveau d'expérience", experience_levels)
    
    # Filtre type de travail - vérifier si la colonne existe
    work_types = ["Tous"]
    if 'formatted_work_type' in jobs_df.columns:
        work_list = list(jobs_df['formatted_work_type'].dropna().unique())
        work_types.extend(work_list)
    work_type_filter = st.sidebar.selectbox("💼 Type de contrat", work_types)
    
    # Nombre de recommandations
    n_recommendations = st.sidebar.slider("📋 Nombre de recommandations", 5, 30, 10)
    
    # Zone principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Décrivez votre profil")
        
        # Zone de texte pour le profil
        default_text = profile_presets[selected_preset]
        profile_text = st.text_area(
            "Entrez vos compétences, expériences et domaines d'intérêt:",
            value=default_text,
            height=150,
            placeholder="Ex: Python, Machine Learning, Data Analysis, SQL, 5 ans d'expérience en finance..."
        )
        
        # Bouton de recherche
        search_clicked = st.button("🔍 Trouver les emplois correspondants", use_container_width=True)
    
    with col2:
        st.subheader("📊 Statistiques")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Offres totales", f"{len(jobs_df):,}")
        with col_b:
            if 'med_salary' in jobs_df.columns:
                avg_salary = jobs_df[jobs_df['med_salary'] > 0]['med_salary'].mean()
                st.metric("Salaire moyen", f"${avg_salary:,.0f}" if not pd.isna(avg_salary) else "N/A")
            else:
                st.metric("Salaire moyen", "N/A")
    
    st.markdown("---")
    
    # Résultats des recommandations
    if search_clicked and profile_text.strip():
        with st.spinner("🔄 Analyse de votre profil en cours..."):
            recommendations = recommend_jobs(
                profile_text=profile_text,
                jobs_df=jobs_df,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                n_recommendations=n_recommendations,
                location_filter=location_filter if location_filter != "Tous" else None,
                min_salary=min_salary,
                experience_filter=experience_filter if experience_filter != "Tous" else None,
                work_type_filter=work_type_filter if work_type_filter != "Tous" else None,
                remote_only=remote_only
            )
        
        if len(recommendations) > 0:
            st.success(f"🎯 {len(recommendations)} emplois trouvés correspondant à votre profil!")
            
            # Afficher les métriques des résultats
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                avg_score = recommendations['similarity_score'].mean() * 100
                st.metric("Score moyen", f"{avg_score:.1f}%")
            with col2:
                max_score = recommendations['similarity_score'].max() * 100
                st.metric("Meilleur match", f"{max_score:.1f}%")
            with col3:
                if 'med_salary' in recommendations.columns:
                    avg_sal = recommendations[recommendations['med_salary'] > 0]['med_salary'].mean()
                    st.metric("Salaire moyen", f"${avg_sal:,.0f}" if avg_sal > 0 else "N/A")
                else:
                    st.metric("Salaire moyen", "N/A")
            with col4:
                if 'remote_allowed' in recommendations.columns:
                    remote_count = (recommendations['remote_allowed'] == 1.0).sum()
                    st.metric("🌍 Remote", f"{remote_count}/{len(recommendations)}")
                else:
                    st.metric("🌍 Remote", "N/A")
            with col5:
                st.metric("Résultats", len(recommendations))
            
            st.markdown("---")
            st.subheader("🏆 Emplois Recommandés")
            
            # Afficher les cartes d'emploi
            for rank, (idx, job) in enumerate(recommendations.iterrows(), 1):
                display_job_card(job, rank)
                st.markdown("")
            
            # Option pour télécharger les résultats
            st.markdown("---")
            csv = recommendations.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv,
                file_name="recommandations_emplois.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Aucun emploi trouvé avec ces critères. Essayez d'ajuster vos filtres.")
    
    elif search_clicked:
        st.warning("⚠️ Veuillez entrer une description de votre profil.")
    
    # Footer
    st.markdown("---")
    
    # Section À propos
    with st.expander("👥 À propos des auteurs"):
        st.markdown("""
        ### 🎯 Équipe JOB INTELLIGENT
        
        Ce projet a été développé par une équipe de passionnés en science des données et IA :
        
        - **👨‍💼 Mohamed Sabbar** - Lead Data Scientist & Architecture ML
        - **👨‍💼 Lamadi Youssef** - Data Engineer & Backend Development
        - **👨‍💼 Mohammed Rida Boukich** - Full Stack Developer & UI/UX
        - **👨‍💼 Abdelhafid Kbiri Alaoui** - Data Analysis & Business Intelligence
        
        Ensemble, nous avons créé une solution complète de recommandation d'emplois basée sur l'IA.
        """)
    
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎯 <strong>Job Intelligent</strong> - Système de recommandation basé sur TF-IDF et similarité cosinus</p>
        <p>📊 Données LinkedIn Job Postings | 🔧 Propulsé par Streamlit & Scikit-learn</p>
        <p><em>© 2026 - Mohamed Sabbar, Lamadi Youssef, Mohammed Rida Boukich, Abdelhafid Kbiri Alaoui</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

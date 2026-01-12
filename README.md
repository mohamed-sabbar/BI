# 🎯 PROJET JOB INTELLIGENT

> **Système intelligent de recommandation d'emplois basé sur l'IA**  
> Analyse de 553,206+ offres LinkedIn avec interface Streamlit avancée et dashboard PowerBI

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)]()
[![Version](https://img.shields.io/badge/Version-2.0-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Ready-brightgreen.svg)]()

---

## 🎯 Vue d'ensemble du Projet

Ce projet combine les meilleures technologies en **Data Science**, **Machine Learning** et **Web Development** :

- **📊 Analyse exploratoire avancée** des données du marché de l'emploi (553K+ offres)
- **🤖 Moteur de recommandation intelligent** basé sur TF-IDF et similarité cosinus
- **🚀 Interface web interactive** avec Streamlit (démarrage ultra-rapide ⚡ < 2 secondes)
- **📈 Dashboard PowerBI complet** pour la visualisation business et insights
- **🔍 Moteur de recherche avancé** avec filtres intelligents
- **💾 Architecture optimisée** avec modèle TF-IDF pré-calculé
- **📱 Interface responsive** et user-friendly avec CSS personnalisé

---

## 🏗️ Architecture RÉELLE du Projet v2.0

### 📂 Structure des Fichiers

```
PROJET-JOB-INTELLIGENT/
│
├── 📌 FICHIERS PRINCIPAUX (Critiques)
│   ├── 🚀 app.py                                              # Application Streamlit
│   ├── 📓 job-market-analysisi.pynb  # Notebook EDA + ML
│   ├── 📦 requirements.txt                                    # Dépendances Python
│   └── 📘 README.md                                           # Documentation complète
│
├── 🎛️ FICHIERS DE CONFIGURATION
│   └── .gitignore                                             # Fichiers ignorés par Git
```

---

## 🎯 Rôle de Chaque Fichier Principal

### 1️⃣ Notebook (`decoding-the-job-market-an-in-depth-exploration.ipynb`)

**Objectif :** Analyse EDA complète + Entraînement ML + Export des données

✅ **Ce qu'il fait :**

- ✨ Charge et nettoie 553,206 offres d'emploi LinkedIn
- 🧹 Pré-traitement des textes
- 📊 Crée 50+ visualisations professionnelles
- 🤖 Entraîne le modèle TF-IDF pour recommandations
- 💾 Exporte le modèle dans `model/`
- 💾 Exporte les données dans `powerbi_data/`

⏱️ **Temps :** ~5-10 minutes (une seule fois)

---

### 2️⃣ Application Streamlit (`app.py`)

**Objectif :** Interface web interactive pour recommandations

✅ **Ce qu'il fait :**

- ⚡ Charge le modèle TF-IDF pré-calculé (< 2 sec)
- 🎨 Fournit interface intuitive et moderne
- 🔍 Recommande emplois pertinents selon profil
- 🎛️ Filtres avancés (localisation, salaire, expérience, etc.)
- 📊 Affichage scores de correspondance (0-100%)
- 📥 Export résultats en CSV
- 👥 5 profils pré-définis

🚀 **Lancement :**

```bash
streamlit run app.py
```

---

### 3️⃣ Dashboard PowerBI (`dashboard.pbix`)

**Objectif :** Visualisations business interactives

✅ **Contient :**

- 📊 50+ visualisations pré-configurées
- 📈 Analyses salariales
- 📍 Distribution géographique
- 🎯 Compétences demandées
- 🏢 Analyse par secteur
- 💼 Types de contrats et télétravail

**Utilisation :**

1. Ouvrir avec PowerBI Desktop
2. Importer données de `powerbi_data/` si besoin
3. Analyser les insights

---

### 4️⃣ Rapport du Projet (`rapport_projet.pdf`)

**Objectif :** Document complet avec méthodologie et résultats

✅ **Contient :**

- 📋 Vue d'ensemble
- 🎯 Objectifs et méthodologie
- 📊 Analyse des données
- 🤖 Détails du modèle ML
- 📈 Résultats et métriques
- 🎨 Architecture application
- 🚀 Guide d'utilisation
- 📝 Conclusions

---

### 5️⃣ Dépendances (`requirements.txt`)

**Objectif :** Gestion des dépendances Python

**Contient :**

- `pandas>=2.0.0` - Manipulation données
- `numpy>=1.24.0` - Calculs numériques
- `scikit-learn>=1.3.0` - TF-IDF, ML
- `matplotlib>=3.7.0` - Graphiques
- `seaborn>=0.12.0` - Visualisations stats
- `plotly>=5.15.0` - Graphiques interactifs
- `streamlit>=1.28.0` - Framework web
- `jupyter>=1.0.0` - Support notebooks

✅ **Installation :**

```bash
pip install -r requirements.txt
```

---

## 🚀 Démarrage Rapide (3 étapes)

### ✅ Étape 1 : Installation

```bash
# Cloner le projet
git clone https://github.com/votre-user/JOB-INTELLIGENT.git
cd JOB-INTELLIGENT

# Créer environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt
```

⏱️ **Temps :** ~2-3 minutes

---

### ✅ Étape 2 : Exécuter le Notebook (Une fois)

Ouvrez `job-market-analysis.ipynb` et exécutez **Cell → Run All**.

**Options :**

**Via Jupyter Lab :**

```bash
jupyter notebook
```

**Ou via nbconvert :**

```bash
jupyter nbconvert --to notebook --execute job-market-analysis.ipynb
```

⏱️ **Temps :** ~5-10 minutes

⚠️ **Important :** À la fin du notebook, vous verrez :

```
✅ Model files saved to: ./model/
✅ PowerBI exports saved to: ./powerbi_data/
```

---

### ✅ Étape 3 : Lancer l'Application Streamlit

```bash
streamlit run app.py
```

🚀 **Résultat :**

- Application se lance sur **http://localhost:8501**
- Démarrage ultra-rapide (< 2 secondes)
- Interface prête à utiliser

---

## 📊 Utilisation de l'Application Streamlit

### 🎯 Page principale

1. **Entrez votre profil** (description libre ou choix profil pré-défini)
2. **Ajustez les filtres** :
   - 📍 Localisation
   - 💰 Salaire minimum
   - 📊 Expérience requise
   - 💼 Type contrat
   - 🌍 Remote only
3. **Cliquez "Chercher emplois"**
4. **Consultez résultats** avec scores matching

### 📋 Résultats

- **Tableau récapitulatif** avec tous les emplois
- **Cartes détaillées** (cliquez pour voir complet)
- **📥 Export CSV** des résultats

### 👥 Profils pré-définis

- **Data Scientist** - Python, ML, Big Data
- **Data Analyst** - SQL, Tableau, Excel
- **Data Engineer** - Spark, Hadoop, ETL
- **ML Engineer** - Deep Learning, TensorFlow
- **Business Analyst** - Excel, SQL, BI

---

## 📊 Utilisation du Dashboard PowerBI

1. **Ouvrir PowerBI Desktop**
2. **Importer données** :
   - File → Import → Folder
   - Sélectionner `./powerbi_data/`
   - Appuyer "Load"
3. **Créer relations** entre tables
4. **Créer visualisations** personnalisées
5. **Analyser insights**

---

## ✨ Fonctionnalités Principales

### 🔍 Recommandations Intelligentes

- **TF-IDF Vectorization** : 3000+ features
- **Cosine Similarity** : Matching précis
- **Scoring 0-100%** : Temps réel
- **Pré-calcul** : Rapidité maximale

### 🎛️ Filtres Avancés

- 📍 Localisation
- 💰 Salaire
- 📊 Expérience
- 💼 Type contrat
- 🌍 Remote
- 🏢 Secteur

### 📊 Visualisations

- Tableaux stylisés
- Cartes détaillées
- Scores matching
- Compétences requises
- Infos entreprise
- Export CSV

---

## 🔄 Workflow Complet

```
DONNÉES (postings.csv)
   ↓
NOTEBOOK (Jupyter)
   ├→ EDA + Visualisations
   ├→ TF-IDF Training
   └→ Exports
   ↓
MODEL/ + POWERBI_DATA/
   ↓
STREAMLIT APP → Dashboard PowerBI
   ↓
UTILISATEURS FINAUX
```

---

## 🛠️ Technologies Utilisées

| Catégorie         | Technologies                            |
| ----------------- | --------------------------------------- |
| **Langage**       | Python 3.12+                            |
| **Data Science**  | Pandas, NumPy                           |
| **ML & NLP**      | Scikit-learn, TF-IDF, Cosine Similarity |
| **Visualisation** | Matplotlib, Seaborn, Plotly             |
| **Web App**       | Streamlit                               |
| **BI**            | PowerBI                                 |
| **Notebook**      | Jupyter                                 |

---

## 📈 Métriques du Projet

| Métrique               | Valeur  |
| ---------------------- | ------- |
| 📊 Offres analysées    | 553,206 |
| 🏢 Entreprises         | 24,473  |
| 💼 Emplois indexés     | 50,000  |
| 🔤 Features TF-IDF     | 3,000+  |
| 📂 Fichiers CSV        | 10+     |
| ⏱️ Démarrage Streamlit | < 2 sec |
| 💾 Taille modèle       | ~500 MB |

---

## ❓ FAQ Complète

### ❓ Quel est l'ordre correct d'exécution ?

**Ordre CRITIQUE :**

1. **EN PREMIER :** Exécuter le notebook
2. **ENSUITE :** Lancer l'app Streamlit

❌ Ne pas faire l'inverse !

---

### ❓ L'app Streamlit est lente au démarrage ?

**Solutions :**

- Vérifier dossier `model/` existe
- Vérifier 4 fichiers `.pkl` présents
- Fermer autres applications
- Disque SSD recommandé

---

### ❓ Comment ajouter mes propres données ?

1. Préparer fichier CSV avec colonnes : `job_id`, `job_title`, `job_description`, `salary`, `location`, etc.
2. Remplacer `dataset/postings.csv`
3. Exécuter notebook complet
4. L'app utilise automatiquement les nouvelles données

---

### ❓ Puis-je déployer en production ?

**Options :**

- **Streamlit Cloud** (gratuit)
- **Heroku** ($7+/mois)
- **AWS/GCP/Azure** ($20-50+/mois)
- **VPS Local** ($5-10/mois)

---

### ❓ Comment améliorer la précision ?

1. Augmenter `max_features` TF-IDF
2. Ajouter plus de données d'entraînement
3. Utiliser Word2Vec ou Transformers
4. Enrichir features (compétences, certifications)

---

## 📝 Notes Importantes

### ⚠️ Fichiers ESSENTIELS

| Fichier            | Pourquoi      | Action                                |
| ------------------ | ------------- | ------------------------------------- |
| `app.py`           | Interface web | NE PAS modifier si ça marche          |
| `requirements.txt` | Dépendances   | INSTALLER avec pip                    |
| `model/`           | Modèle ML     | GÉNÉRÉ par notebook (ne pas modifier) |

### 🔐 Fichiers Générés (ne pas modifier)

```
Ne PAS éditer manuellement :
├── model/*.pkl (tous les fichiers)
└── powerbi_data/*.csv (tous les fichiers)
```

### 💾 Recommandation Sauvegarde

```
Sauvegarder régulièrement :
├── app.py
├── requirements.txt
├── model/
└── README.md

Ne pas sauvegarder :
├── __pycache__/
├── .streamlit/cache/
└── powerbi_data/ (peut être régénéré)
```

---

## 🎓 Auteurs et Rôles

| 👤 Nom                      | 🎓 Rôle               | 📊 Responsabilités           |
| --------------------------- | --------------------- | ---------------------------- |
| **Mohamed Sabbar**          | Lead Data Scientist   | ML, TF-IDF, Architecture     |
| **Lamadi Youssef**          | Data Engineer         | ETL, Backend, Infrastructure |
| **Mohammed Rida Boukich**   | Full Stack Developer  | Streamlit, Frontend, UX/UI   |
| **Abdelhafid Kbiri Alaoui** | Business Intelligence | PowerBI, Analytics, Insights |

---

## 🤝 Guide Contribution

### 📋 Avant de Commencer

1. **Fork** le repository
2. **Clone** en local
3. **Créer branche** : `git checkout -b feature/MaFonctionnalite`

### 🛠️ Pendant le Développement

1. Installer dépendances
2. Faire modifications
3. Respecter PEP8
4. Tester code

### ✅ Soumettre Pull Request

1. **Push** vers votre fork
2. **Ouvrir PR** sur repo principal
3. **Décrire** changements clairement
4. **Attendre review** et merger

---

## 📄 Licence

**MIT License** - Libre d'utilisation personnelle et professionnelle.

### Copyright

```
Copyright (c) 2026 Équipe JOB INTELLIGENT
License: MIT
Authors: Mohamed Sabbar, Lamadi Youssef,
         Mohammed Rida Boukich, Abdelhafid Kbiri Alaoui
```

---

## 📞 Support

- 📝 Issues GitHub
- 💬 Discussions
- 📧 Email (pour enterprise)

---

**Dernière mise à jour :** Janvier 2026  
**Version :** 2.0 (Production Ready)  
**Statut :** ✅ Actif et maintenu

**⭐ Star si vous trouvez ce projet utile !**

© 2026 - JOB INTELLIGENT

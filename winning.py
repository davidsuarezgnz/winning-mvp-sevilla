import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

############################################################################################################################################################
###                                                                  PAGE CONFIG                                                                         ###
############################################################################################################################################################
st.set_page_config(page_title="Winning", layout="wide")

############################################################################################################################################################
###                                                                  DATA CONFIG                                                                         ###
############################################################################################################################################################
# Cargamos el dataset y creamos la lista de despliegue
data = pd.read_csv("players.csv", encoding="utf-8")
players = list(data["player_name"].unique())
options = [None] + players

# Definimos las características por posición
position_features = {
    "Goalkeeper": ["total_gkSaves", "total_gkConcededGoals", "total_gkExits", "total_gkShotsAgainst", "total_gkSuccessfulExits"],
    "Defender": ["total_interceptions", "total_clearances", "total_successfulDefensiveAction", "total_duelsWon", "total_successfulPasses"],
    "Midfielder": ["total_passesToFinalThird", "total_successfulSmartPasses", "total_recoveries", "total_successfulDribbles", "total_keyPasses"],
    "Forward": ["total_goals", "total_shotsOnTarget", "total_xgShot", "total_successfulAttackingActions", "total_successfulDribbles"]
}

# Definimos el conjunto de características por defecto, por si la posición no está en el diccionario
default_features = ["goals", "successfulPasses", "duelsWon", "xgShot", "successfulDribbles"]

############################################################################################################################################################
###                                                                     HEADER                                                                           ###
############################################################################################################################################################
col_top_left, col_top_spacer, col_top_right = st.columns([1, 6, 1])

with col_top_left:
    st.image("w-logo.png", width=160)
with col_top_spacer:
    pass
with col_top_right:
    st.image("w-icon.png", width=40)

st.markdown("---")

############################################################################################################################################################
###                                                                     TITLE                                                                            ###
############################################################################################################################################################
st.markdown("<h2 style='text-align: center;'>\n\nBienvenido a Winning, encuentra el reemplazo ideal para cualquier jugador</h2>", unsafe_allow_html=True)

############################################################################################################################################################
###                                                                 SELECT PLAYER                                                                        ###
############################################################################################################################################################
st.markdown("<h3 style='text-align: center;'>Selecciona a un jugador</h3>", unsafe_allow_html=True)

# Creamos columnas para centrar el selectbox
col_left, col_mid, col_right = st.columns([2, 3, 2])
with col_mid:
    selected_player = st.selectbox(
        "Buscar",
        options,
        index=0,
        format_func=lambda x: "" if x is None else x,
        key="player_select"
    )

    
    if selected_player is None:
        st.write("No has seleccionado ningún jugador aún.")
    else:
        player_info = data.loc[data["player_name"] == selected_player].iloc[0]
        st.write(f"Has seleccionado a: {selected_player}")
        
        # Centramos la cara
        col_left, col_mid, col_right = st.columns([2, 2, 1])
        with col_mid:
            st.image(player_info["imageDataURL"], width=100)


# Separación
st.markdown("<h3 style='text-align: center;'>\n\n</h3>", unsafe_allow_html=True)

############################################################################################################################################################
###                                                           FIND SIMILA PLAYERS FUNCTION                                                               ###
############################################################################################################################################################
def find_similar_players(player_name, n_neighbors=10):
    # Obtener los datos del jugador específico por nombre
    myplayer = data[data['player_name'] == player_name]
    if myplayer.empty:
        st.write(f"No se encontró al jugador: {player_name}")
        return pd.DataFrame()
    
    # Obtener la posición y seleccionar las características correspondientes
    role = myplayer.iloc[0]["role_name"]
    features = position_features.get(role, default_features)

    # Filtrar solo las filas con la misma posición del jugador
    same_role_players = data[data["role_name"] == role]

    # Si no hay suficientes jugadores en esa posición, devolver vacío
    if len(same_role_players) < n_neighbors + 1:
        st.write(f"No hay suficientes jugadores en la posición {role} para hacer la comparación.")
        return pd.DataFrame()
    
    # Seleccionamos las columnas de las características y escalamos los datos
    X = same_role_players[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Inicializar KNN y buscar jugadores más cercanos
    knn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(same_role_players)), algorithm="auto").fit(X_scaled)
    
    # Encontrar el índice relativo del jugador dentro del grupo
    player_index = myplayer.index[0]
    rel_index = same_role_players.index.get_loc(player_index)
    player_features = X_scaled[rel_index].reshape(1, -1)
    
    # Encontrar los vecinos más cercanos
    distances, indices = knn.kneighbors(player_features)

    # Omitir el jugador mismo y devolver los más cercanos
    similar_indices = indices[0][0:]
    return same_role_players.iloc[similar_indices]

############################################################################################################################################################
###                                                         PLOTEAR JUGADORES EN 2D                                                                      ###
############################################################################################################################################################
def plot_similar_players_2d(same_role_players, X_scaled, target_index, similar_indices, selected_player):
    plt.style.use('dark_background')
    
    # Reducimos la dimensionalidad a 2 componentes
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    # Jugadores de la misma posición
    all_indices = same_role_players.index

    # Separamos el índice del jugador y los vecinos
    target_idx = all_indices[target_index]
    similar_real_indices = all_indices[similar_indices]

    for i, idx in enumerate(all_indices):
        if idx == target_idx:
            plt.scatter(X_2d[i, 0], X_2d[i, 1], c='red', marker='o', s=150, label='Jugador Seleccionad')
        elif idx in similar_real_indices:
            plt.scatter(X_2d[i, 0], X_2d[i, 1], c='green', marker='*', s=150, label='Jugadorres Similares')
        else:
            plt.scatter(X_2d[i, 0], X_2d[i, 1], c='gray', marker='x', alpha=0.5)

    plt.title(f"Jugadores similares a '{selected_player}' (Misma Posición)")
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys())
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    st.pyplot(plt.gcf())
    plt.close()

############################################################################################################################################################
###                                                                 RUN ALGORITHM                                                                        ###
############################################################################################################################################################
st.markdown("<h3 style='text-align: center;'>Reemplazo ideal</h3>", unsafe_allow_html=True)

if selected_player is not None:
    if st.button("Calcular reemplazo ideal"):
        similar_players = find_similar_players(selected_player)
        
        if similar_players.empty:
            st.write("No se encontró reemplazo ideal.")
        else:
            # Mostramos tablas
            st.write(f"Estadísticas de {selected_player}:")
            st.table(similar_players.iloc[0:1][['player_name', 'role_name'] + position_features.get(player_info["role_name"], default_features)])
            st.write(f"Jugadores similares a {selected_player}:")
            st.table(similar_players.iloc[1:][['player_name', 'role_name'] + position_features.get(player_info["role_name"], default_features)])

            # Mostramos gráfica
            role = player_info["role_name"]
            features = position_features.get(role, default_features)
            same_role_players = data[data["role_name"] == role]

            X = same_role_players[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            myplayer_index = player_info.name  # índice real del DF
            target_index = same_role_players.index.get_loc(myplayer_index)

            similar_indices = [same_role_players.index.get_loc(idx) for idx in similar_players.index[1:]]

            plot_similar_players_2d(
                same_role_players, 
                X_scaled, 
                target_index=target_index, 
                similar_indices=similar_indices, 
                selected_player=selected_player
            )

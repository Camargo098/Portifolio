import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

st.set_page_config(layout='wide', page_title='Harry Potter Analysis', page_icon='⚡')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative&display=swap');
.title { font-family: 'Cinzel Decorative', cursive; color: #4B0082; }
</style>
<h2 class="title">Data Magic: A Harry Potter Analysis</h2>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base_url = 'https://raw.githubusercontent.com/Camargo098/Portifolio/main/Harry_Potter_Movies/data/'
    
    # Load all data with consistent encoding
    data_files = {
        'Chapters': pd.read_csv(base_url + 'Chapters.csv', encoding='latin1'),
        'Characters': pd.read_csv(base_url + 'Characters.csv', encoding='latin1'),
        'Dialogue': pd.read_csv(base_url + 'Dialogue.csv', encoding='latin1'),
        'Movies': pd.read_csv(base_url + 'Movies.csv', encoding='latin1'),
        'Places': pd.read_csv(base_url + 'Places.csv', encoding='latin1'),
        'Houses': pd.read_csv(base_url + 'Houses.csv', encoding='latin1'),
        'Spells': pd.read_csv(base_url + 'Spells.csv', encoding='latin1')
    }
    
    # Clean column names (remove BOM if present)
    for df in data_files.values():
        df.columns = df.columns.str.replace('ï»¿', '')
    
    # Create merged relationships
    dialogue_chapters = pd.merge(
        data_files['Dialogue'],
        data_files['Chapters'],
        on='Chapter ID',
        how='left'
    )
    
    full_data = pd.merge(
        dialogue_chapters,
        data_files['Movies'],
        on='Movie ID',
        how='left'
    )
    
    return {**data_files, 'Merged': full_data}

# Load all data
data = load_data()
Chapters = data['Chapters']
Characters = data['Characters']
Dialogue = data['Dialogue']
Movies = data['Movies']
Places = data['Places']
Houses = data['Houses']
Spells = data['Spells']
Merged = data['Merged']

# Sidebar filters
st.sidebar.header('Filters')
movie_options = ['All'] + Movies.sort_values('Movie ID')['Movie Title'].tolist()
selected_movie = st.sidebar.selectbox('Select Movie', options=movie_options)

min_appearances = st.sidebar.slider("Minimum Appearances", 0, 2000, 5)

# Character Analysis
st.header('🏰 Character Presence')

# Filter by movie if selected
if selected_movie != "All":
    filtered_data = Merged[Merged['Movie Title'] == selected_movie]
else:
    filtered_data = Merged

# Calculate appearances
character_stats = (
    filtered_data['Character ID']
    .value_counts()
    .reset_index()
    .rename(columns={'count': 'appearances'})
)

df_chars = pd.merge(
    character_stats,
    Characters,
    on='Character ID',
    how='left'
).query(f'appearances >= {min_appearances}')

# Fill missing houses
df_chars['House'] = df_chars['House'].fillna('Unknown')

# Visualization
fig = px.treemap(
    df_chars,
    path=['House', 'Character Name'],
    values='appearances',
    color='appearances',
    hover_data=['appearances'],
    color_continuous_scale='Purples',
    title=f'Character Appearances {"in " + selected_movie if selected_movie != "All" else "Across All Movies"}'
)

st.plotly_chart(fig, use_container_width=True)

# Sentiment Analysis
st.header('📜 Dialogue Sentiment Analysis')

def analyze_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

# Apply movie filter to sentiment analysis
if selected_movie != "All":
    movie_id = Movies[Movies['Movie Title'] == selected_movie]['Movie ID'].iloc[0]
    chapters_in_movie = Chapters[Chapters['Movie ID'] == movie_id]['Chapter ID']
    filtered_dialogues = Dialogue[Dialogue['Chapter ID'].isin(chapters_in_movie)]
else:
    filtered_dialogues = Dialogue

# Sample for performance if large dataset
dialogues_sample = filtered_dialogues.sample(min(10000, len(filtered_dialogues))) if len(filtered_dialogues) > 1000 else filtered_dialogues
dialogues_sample['sentiment'] = dialogues_sample['Dialogue'].apply(analyze_sentiment)

# Sentiment visualization
fig_sentiment = px.histogram(
    dialogues_sample,
    x='sentiment',
    nbins=30,
    title=f'Sentiment Distribution {"in " + selected_movie if selected_movie != "All" else "Across All Movies"}',
    color_discrete_sequence=['#8A2BE2'],
    labels={'sentiment': 'Sentiment Polarity'}
)

# Add mean line
mean_sentiment = dialogues_sample['sentiment'].mean()
fig_sentiment.add_vline(
    x=mean_sentiment, 
    line_dash="dash", 
    line_color="red",
    annotation_text=f"Mean: {mean_sentiment:.2f}"
)

st.plotly_chart(fig_sentiment, use_container_width=True)

# Additional metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Dialogues Analyzed", len(dialogues_sample))
col2.metric("Average Sentiment", f"{mean_sentiment:.2f}")
most_positive = dialogues_sample.loc[dialogues_sample['sentiment'].idxmax()]
most_positive_name = Characters.loc[
    Characters['Character ID'] == most_positive['Character ID'],
    'Character Name'
].values[0]

col3.metric(
    "Most Positive Dialogue",
    most_positive_name,
    help=f"Character with most positive dialogue: {most_positive['sentiment']:.2f} sentiment score"
)


# Análise de Sentimento por Filme - Gráfico de Linha Atualizado
st.header('🎬 Sentiment Evolution Across Movies')

# Preparar dados de sentimento por filme
sentiment_by_movie = (
    dialogues_sample
    .merge(Merged[['Dialogue ID', 'Movie ID']], on='Dialogue ID')
    .merge(Movies, on='Movie ID')
    .groupby(['Movie ID', 'Movie Title'], as_index=False)
    ['sentiment'].mean()
    .sort_values('Movie ID')
)

# Criar gráfico de linha aprimorado
fig_movie_sentiment = px.line(
    sentiment_by_movie,
    x='Movie Title',
    y='sentiment',
    markers=True,
    title="<b>Average Sentiment by Movie</b>",
    line_shape='spline',
    color_discrete_sequence=['#4B0082'],
    labels={
        'Movie Title': 'Movie',
        'sentiment': 'Average Sentiment',
    },
    template='plotly_white'
)

# Melhorias visuais
fig_movie_sentiment.update_layout(
    hovermode='x unified',
    title_font_size=20,
    title_x=0.5,
    xaxis=dict(
        tickmode='array',
        tickvals=sentiment_by_movie['Movie Title'],
        tickangle=45
    ),
    yaxis=dict(
        range=[sentiment_by_movie['sentiment'].min() - 0.05, 
               sentiment_by_movie['sentiment'].max() + 0.05]
    )
)

# Adicionar linha de média geral
fig_movie_sentiment.add_hline(
    y=mean_sentiment,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Overall Mean: {mean_sentiment:.2f}",
    annotation_position="bottom right"
)

# Adicionar pontos de destaque
max_movie = sentiment_by_movie.loc[sentiment_by_movie['sentiment'].idxmax()]
min_movie = sentiment_by_movie.loc[sentiment_by_movie['sentiment'].idxmin()]

fig_movie_sentiment.add_annotation(
    x=max_movie['Movie Title'],
    y=max_movie['sentiment'],
    text=f"Most Positive: {max_movie['sentiment']:.2f}",
    showarrow=True,
    arrowhead=1,
    yshift=10
)

fig_movie_sentiment.add_annotation(
    x=min_movie['Movie Title'],
    y=min_movie['sentiment'],
    text=f"Most Negative: {min_movie['sentiment']:.2f}",
    showarrow=True,
    arrowhead=1,
    yshift=-10
)

st.plotly_chart(fig_movie_sentiment, use_container_width=True)
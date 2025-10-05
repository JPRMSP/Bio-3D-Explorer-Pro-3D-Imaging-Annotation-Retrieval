import streamlit as st
import numpy as np
import plotly.graph_objects as go
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial.distance import cosine

st.set_page_config(page_title="Bio-3D Explorer", layout="wide")
st.title("🧠 Bio-3D Explorer: 3D Scan Synthesizer & Retrieval Tool")

# --- Generate synthetic 3D volume ---
def generate_volume(size, num_blobs):
    volume = np.zeros((size, size, size))
    for _ in range(num_blobs):
        x0, y0, z0 = np.random.randint(0, size, 3)
        sigma = np.random.randint(size//10, size//5)
        x, y, z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing='ij')
        blob = np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2*sigma**2))
        volume += blob
    volume = (volume / volume.max() * 255).astype(np.uint8)
    return volume

# --- Feature extraction ---
def shape_features(volume):
    coords = np.argwhere(volume > 50)
    centroid = coords.mean(axis=0)
    vol_size = np.sum(volume > 50)
    surface = np.count_nonzero(np.gradient(volume)[0] > 10)
    return np.array([centroid[0], centroid[1], centroid[2], vol_size, surface])

def texture_features(volume):
    mid_slice = volume[volume.shape[0] // 2]
    glcm = graycomatrix(mid_slice, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    return np.array([contrast, energy, homogeneity])

# --- 3D visualization ---
def plot_3d(volume):
    x, y, z = np.where(volume > 50)
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=1, color=volume[x, y, z], colorscale='Viridis', opacity=0.7)
    )])
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
    return fig

# --- UI ---
size = st.sidebar.slider("Volume Size", 32, 128, 64)
blobs = st.sidebar.slider("No. of Synthetic Structures", 1, 10, 3)

if st.button("🧬 Generate New 3D Volume"):
    vol = generate_volume(size, blobs)
    st.session_state['current_volume'] = vol

if 'current_volume' in st.session_state:
    st.subheader("📊 Generated 3D Volume")
    st.plotly_chart(plot_3d(st.session_state['current_volume']), use_container_width=True)

    sf = shape_features(st.session_state['current_volume'])
    tf = texture_features(st.session_state['current_volume'])

    st.write("**Shape Features:**", sf)
    st.write("**Texture Features:**", tf)

    # Save this volume to image set library
    if st.button("📁 Save to Image Set Library"):
        if 'library' not in st.session_state:
            st.session_state['library'] = []
        st.session_state['library'].append({'shape': sf, 'texture': tf})
        st.success("Volume added to library!")

# --- Retrieval Section ---
st.subheader("🔍 Retrieve Most Similar Image Set")
if 'library' in st.session_state and len(st.session_state['library']) > 1:
    current_sf = shape_features(st.session_state['current_volume'])
    current_tf = texture_features(st.session_state['current_volume'])
    best_match, best_score = None, -1

    for idx, entry in enumerate(st.session_state['library'][:-1]):
        shape_sim = 1 - cosine(entry['shape'], current_sf)
        texture_sim = 1 - cosine(entry['texture'], current_tf)
        score = (shape_sim + texture_sim) / 2
        if score > best_score:
            best_match, best_score = idx, score

    st.metric("📈 Best Match Similarity", f"{best_score*100:.2f}%")
    st.write(f"✅ Closest match found at index #{best_match} in your saved image sets.")
else:
    st.info("Generate and save at least 2 volumes to test retrieval.")

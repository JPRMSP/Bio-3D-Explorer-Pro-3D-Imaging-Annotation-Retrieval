import streamlit as st
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial.distance import cosine
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Bio-3D Explorer Pro", layout="wide")
st.title("🧠 Bio-3D Explorer Pro: 3D Imaging + Annotation + Retrieval")

# ----------------- Generate Synthetic Volume -----------------
def generate_volume(size, blobs):
    vol = np.zeros((size, size, size))
    for _ in range(blobs):
        x0, y0, z0 = np.random.randint(0, size, 3)
        sigma = np.random.randint(size // 12, size // 6)
        x, y, z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing='ij')
        blob = np.exp(-((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * sigma ** 2))
        vol += blob
    vol = (vol / vol.max() * 255).astype(np.uint8)
    return vol

# ----------------- Features Extraction -----------------
def shape_features(volume):
    coords = np.argwhere(volume > 50)
    centroid = coords.mean(axis=0)
    vol_size = np.sum(volume > 50)
    surface = np.count_nonzero(np.gradient(volume)[0] > 10)
    return np.array([centroid[0], centroid[1], centroid[2], vol_size, surface])

def texture_features(volume):
    mid_slice = volume[volume.shape[0] // 2]
    glcm = graycomatrix(mid_slice, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return np.array([contrast, energy, homogeneity])

# ----------------- Visualization -----------------
def plot_3d(volume):
    x, y, z = np.where(volume > 50)
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=1, color=volume[x, y, z], colorscale='Viridis', opacity=0.7)
    )])
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
    return fig

# ----------------- Sidebar Controls -----------------
st.sidebar.header("⚙️ Volume Generation Settings")
size = st.sidebar.slider("Volume Size", 32, 128, 128)
blobs = st.sidebar.slider("No. of Synthetic Structures", 1, 10, 4)

if st.sidebar.button("🧬 Generate New 3D Volume"):
    st.session_state['volume'] = generate_volume(size, blobs)
    st.session_state['annotations'] = {}

if 'volume' in st.session_state:
    vol = st.session_state['volume']
    st.subheader("📊 3D Volume Visualization")
    st.plotly_chart(plot_3d(vol), use_container_width=True)

    sf = shape_features(vol)
    tf = texture_features(vol)
    st.write("**Shape Features:**", sf)
    st.write("**Texture Features:**", tf)

    if st.button("📁 Save Volume to Library"):
        if 'library' not in st.session_state:
            st.session_state['library'] = []
        st.session_state['library'].append({'shape': sf, 'texture': tf})
        st.success("✅ Volume added to library!")

    # ----------------- Slice Viewer & Annotation -----------------
    st.subheader("🖼️ Slice-by-Slice Viewer & Annotation")
    slice_idx = st.slider("Select Slice", 0, vol.shape[0]-1, vol.shape[0]//2)
    slice_img = vol[slice_idx].copy()

    # Auto annotation
    auto_mask = np.zeros_like(slice_img)
    auto_mask[slice_img > 180] = 255

    # Combine auto and manual for hybrid
    hybrid_mask = auto_mask.copy()
    if slice_idx in st.session_state['annotations']:
        for (x, y) in st.session_state['annotations'][slice_idx]:
            cv2.circle(hybrid_mask, (y, x), 3, 255, -1)

    col1, col2, col3 = st.columns(3)
    col1.image(slice_img, caption=f"Original Slice {slice_idx}", use_column_width=True)
    col2.image(auto_mask, caption="Auto-Detected Abnormalities", use_column_width=True)
    col3.image(hybrid_mask, caption="Hybrid Annotation", use_column_width=True)

    # Manual annotation (pixel-click)
    st.markdown("### ✏️ Manual Pixel Annotation")
    st.info("Click on pixels in the slice (x, y) to mark regions. (Simulated with input below)")
    x = st.number_input("X coordinate", min_value=0, max_value=vol.shape[1]-1, step=1)
    y = st.number_input("Y coordinate", min_value=0, max_value=vol.shape[2]-1, step=1)
    if st.button("✅ Annotate Pixel"):
        if slice_idx not in st.session_state['annotations']:
            st.session_state['annotations'][slice_idx] = []
        st.session_state['annotations'][slice_idx].append((int(x), int(y)))
        st.success(f"Annotated pixel at ({int(x)}, {int(y)}) on slice {slice_idx}")

    # ----------------- Retrieval -----------------
    st.subheader("🔍 Image Set Retrieval")
    if 'library' in st.session_state and len(st.session_state['library']) > 1:
        current_sf = shape_features(vol)
        current_tf = texture_features(vol)
        best_match, best_score = None, -1
        for idx, entry in enumerate(st.session_state['library'][:-1]):
            shape_sim = 1 - cosine(entry['shape'], current_sf)
            texture_sim = 1 - cosine(entry['texture'], current_tf)
            score = (shape_sim + texture_sim) / 2
            if score > best_score:
                best_match, best_score = idx, score
        st.metric("📈 Best Match Similarity", f"{best_score * 100:.2f}%")
        st.success(f"Closest match: Volume #{best_match} in your library.")
    else:
        st.info("📁 Save at least 2 volumes to test retrieval.")
else:
    st.warning("Click '🧬 Generate New 3D Volume' to start.")

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.digit-button {
    font-size: 1.5rem;
    margin: 0.25rem;
    padding: 0.5rem 1rem;
    border-radius: 10px;
    border: 2px solid #1f77b4;
    background-color: white;
    color: #1f77b4;
}

.generated-image {
    border: 2px solid #ddd;
    border-radius: 10px;
    padding: 10px;
    margin: 10px;
}
</style>
""", unsafe_allow_html=True)

# Generator model definition (same as training script)
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_emb = self.label_emb(labels)
        gen_input = torch.cat([noise, label_emb], dim=1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

@st.cache_resource
def load_model():
    """Load the trained GAN model"""
    try:
        # Model parameters
        LATENT_DIM = 100
        NUM_CLASSES = 10
        IMAGE_SIZE = 28
        
        # Initialize model
        generator = Generator(LATENT_DIM, NUM_CLASSES, IMAGE_SIZE)
        
        # Load trained weights
        checkpoint = torch.load('mnist_gan_model.pth', map_location='cpu')
        generator.load_state_dict(checkpoint['generator'])
        generator.eval()
        
        return generator, LATENT_DIM
    except FileNotFoundError:
        st.error("Model file 'mnist_gan_model.pth' not found. Please upload the trained model.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_digit_images(generator, digit, latent_dim, num_samples=5):
    """Generate images for a specific digit"""
    with torch.no_grad():
        # Create labels and noise
        labels = torch.full((num_samples,), digit, dtype=torch.long)
        noise = torch.randn(num_samples, latent_dim)
        
        # Generate images
        fake_imgs = generator(noise, labels)
        
        # Denormalize from [-1, 1] to [0, 1]
        fake_imgs = (fake_imgs + 1) / 2
        fake_imgs = fake_imgs.clamp(0, 1)
        
        return fake_imgs.numpy()

def create_image_grid(images, digit):
    """Create a grid of generated images"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(f'Generated Digit: {digit}', fontsize=20, fontweight='bold')
    
    for i in range(5):
        axes[i].imshow(images[i, 0], cmap='gray', interpolation='nearest')
        axes[i].set_title(f'Sample {i+1}', fontsize=14)
        axes[i].axis('off')
        axes[i].set_aspect('equal')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Convert to bytes for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def main():
    # Title
    st.markdown('<h1 class="main-header">🔢 MNIST Handwritten Digit Generator</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">
    Generate realistic handwritten digits (0-9) using a trained Conditional GAN model.
    Select a digit below and click "Generate" to create 5 unique samples!
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model..."):
        generator, latent_dim = load_model()
    
    if generator is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    st.sidebar.markdown("Select which digit to generate:")
    
    # Digit selection
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Select Digit")
        
        # Create digit buttons in a grid
        button_cols = st.columns(5)
        selected_digit = None
        
        for i in range(10):
            col_idx = i % 5
            with button_cols[col_idx]:
                if st.button(str(i), key=f"digit_{i}", help=f"Generate digit {i}"):
                    selected_digit = i
        
        # Alternative: Use selectbox
        st.markdown("---")
        digit_select = st.selectbox("Or select from dropdown:", range(10), format_func=lambda x: f"Digit {x}")
        
        if selected_digit is None:
            selected_digit = digit_select
    
    with col2:
        st.subheader("Generation Settings")
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            seed = st.number_input("Random Seed (for reproducibility)", value=42, min_value=0, max_value=10000)
            temperature = st.slider("Generation Temperature", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            st.info("Higher temperature = more diverse but potentially less realistic images")
        
        # Generate button
        generate_btn = st.button("🎨 Generate Images", type="primary", help="Generate 5 samples of the selected digit")
    
    # Generation and display
    if generate_btn or st.session_state.get('auto_generate', False):
        if selected_digit is not None:
            with st.spinner(f"Generating digit {selected_digit}..."):
                # Set seed for reproducibility
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Generate images
                generated_images = generate_digit_images(generator, selected_digit, latent_dim, num_samples=5)
                
                # Apply temperature scaling if different from 1.0
                if temperature != 1.0:
                    generated_images = np.power(generated_images, 1.0/temperature)
                    generated_images = np.clip(generated_images, 0, 1)
                
                # Create and display image grid
                image_buf = create_image_grid(generated_images, selected_digit)
                st.image(image_buf, caption=f"5 Generated Samples of Digit {selected_digit}", use_column_width=True)
                
                # Individual images
                st.subheader("Individual Samples")
                img_cols = st.columns(5)
                
                for i in range(5):
                    with img_cols[i]:
                        # Convert numpy array to PIL Image
                        img_array = (generated_images[i, 0] * 255).astype(np.uint8)
                        pil_image = Image.fromarray(img_array, mode='L')
                        
                        # Resize for better display
                        pil_image = pil_image.resize((112, 112), Image.NEAREST)
                        
                        st.image(pil_image, caption=f"Sample {i+1}", use_column_width=True)
                
                # Download option
                st.markdown("---")
                if st.button("💾 Download All Images"):
                    # Create zip file with all images
                    import zipfile
                    zip_buf = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buf, 'w') as zip_file:
                        for i in range(5):
                            img_array = (generated_images[i, 0] * 255).astype(np.uint8)
                            pil_image = Image.fromarray(img_array, mode='L')
                            
                            img_bytes = io.BytesIO()
                            pil_image.save(img_bytes, format='PNG')
                            
                            zip_file.writestr(f'digit_{selected_digit}_sample_{i+1}.png', img_bytes.getvalue())
                    
                    zip_buf.seek(0)
                    
                    st.download_button(
                        label="📥 Download ZIP",
                        data=zip_buf.getvalue(),
                        file_name=f"digit_{selected_digit}_samples.zip",
                        mime="application/zip"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    🤖 Powered by Conditional GAN trained on MNIST dataset<br>
    Model trained using PyTorch on Google Colab T4 GPU
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions for first-time users
    with st.sidebar:
        st.markdown("---")
        st.header("📋 How to Use")
        st.markdown("""
        1. **Select a digit** (0-9) using buttons or dropdown
        2. **Click "Generate Images"** to create 5 samples
        3. **Adjust settings** in Advanced Options if needed
        4. **Download images** using the download button
        
        Each generation creates unique, AI-generated handwritten digits!
        """)
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This app uses a **Conditional Generative Adversarial Network (GAN)** 
        trained on the famous MNIST handwritten digit dataset.
        
        The model learned to generate realistic digit images by competing 
        against a discriminator network during training.
        """)

if __name__ == "__main__":
    main()


# install dependencies first
# !pip install transformer-lens torch matplotlib scikit-learn

import torch
import transformer_lens
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
import numpy as np

# Force matplotlib to use a compatible backend (for Mac issues)
matplotlib.use('TkAgg')  

# 1. LOAD GPT-2 SMALL
print("Loading GPT-2 small model...")
model = HookedTransformer.from_pretrained("gpt2-small", center_unembed=True, center_writing_weights=True)
print("Model loaded successfully.\n")

# 2. DEFINE AFFECTIVE PROMPTS
print("Defining affective prompts...")
prompts = [
    "You just won the lottery and feel ecstatic!",
    "Your best friend moved away, and you feel lonely.",
    "Someone just cut you off in traffic.",
    "You hear footsteps behind you in the dark.",
    "You received a heartfelt compliment from a stranger.",
    "Your beloved pet just passed away.",
    "Someone just insulted you publicly.",
    "You're standing on stage, forgetting your lines.",
    "You just got promoted at work after months of hard effort.",
    "Your favorite childhood toy was found in an old box.",
    "You failed an important exam despite studying hard.",
    "A stranger helped you when you dropped your groceries.",
    "You're about to give a speech in front of hundreds of people.",
    "Your friend forgot your birthday.",
    "You’re watching the sunset on your dream vacation.",
    "Your partner surprises you with a thoughtful gift.",
    "Your new recipe turned out horribly wrong.",
    "You just solved a difficult puzzle after hours of trying.",
    "You're waiting for an important medical test result.",
    "Your crush just smiled at you from across the room.",
    "You lost your wallet in a foreign country.",
    "You’re sitting alone in a café, watching the rain fall.",
    "You overheard someone spreading rumors about you.",
    "Your team just won the championship.",
    "You accidentally broke a family heirloom.",
    "You just adopted a puppy from the shelter.",
    "Your favorite TV show got canceled unexpectedly.",
    "A long-lost friend messaged you out of the blue.",
    "You missed your flight by just a few minutes.",
    "You’re in a haunted house and hear a door creak open.",
    "You completed a marathon after months of training.",
    "You found an old letter from a loved one.",
    "You forgot an important meeting with your boss.",
    "You just received a rejection letter from your dream school.",
    "You’re reunited with your family after years apart.",
    "You spilled coffee on your laptop during a presentation.",
    "You just got engaged to the love of your life.",
    "You’re walking through a quiet forest at sunrise.",
    "You just learned a close friend lied to you.",
    "Your hard work finally got recognized at an awards ceremony.",
    "You’re stuck in traffic on the way to an important interview.",
    "You’re about to meet your idol for the first time.",
    "Your sibling got into a serious accident.",
    "You successfully defended your thesis after months of work.",
    "Your partner forgot your anniversary.",
    "You just bought your first home.",
    "You're standing at the edge of a cliff, looking down.",
    "You found a surprise birthday party waiting for you.",
    "You spent the entire day helping at a local shelter.",
    "You found out a secret you weren't supposed to know."
]
print(f"Total prompts: {len(prompts)}\n")

# 3. TOKENIZE PROMPTS
print("Tokenizing prompts...")
tokens = model.to_tokens(prompts)
print("Tokenization complete.\n")

# 4. SET UP HOOKS TO GRAB ACTIVATIONS FROM LAYER 2
print("Setting up hooks to capture activations from Layer 2...")
activations = {}

# Hook function to capture activations
def hook_fn(value, hook):
    activations[hook.name] = value

# Run model with hook on layer 2's pre-residual stream
print("Running model and capturing activations...")
_ = model.run_with_hooks(
    tokens,
    return_type="logits",
    fwd_hooks=[("blocks.2.hook_resid_pre", hook_fn)]
)
print("Activations captured.\n")

# 5. ANALYZE ACTIVATIONS: NEURON VARIANCE ACROSS PROMPTS
print("Analyzing activations...")

# Average activations across tokens for each prompt
avg_activations = activations["blocks.2.hook_resid_pre"].mean(dim=1)  # shape: (num_prompts, hidden_dim)

# Check for NaNs or Infs
if np.isnan(avg_activations.detach().numpy()).any():
    print("Warning: NaN detected in activations.")
if np.isinf(avg_activations.detach().numpy()).any():
    print("Warning: Inf detected in activations.")

# Compute variance across prompts for each neuron
neuron_variance = avg_activations.var(dim=0)  # shape: (hidden_dim,)

# Plot top neurons by variance
#top_k = 10
#top_neurons = torch.topk(neuron_variance, top_k)

print("Plotting top neurons by variance...")
#plt.figure(figsize=(8, 5))
#plt.bar(range(top_k), top_neurons.values.detach().numpy())
#plt.xlabel("Neuron Index")
#plt.ylabel("Activation Variance Across Prompts")
#plt.title("Top Neurons Sensitive to Affect (Layer 2)")
#plt.show(block=True)  # Ensure plot displays properly

# 6. VISUALIZE PROMPT ACTIVATIONS VIA PCA CLUSTERING
print("Running PCA for prompt clustering...")

# Reduce to 2D using PCA for visualization
pca = PCA(n_components=2, svd_solver='full')
reduced = pca.fit_transform(avg_activations.detach().numpy())

# Map prompts to emotions (for simplicity, random mapping here)
emotions = ["happy", "sad", "angry", "fearful"] * (len(prompts) // 4 + 1)
colors = {
    "happy": "green",
    "sad": "blue",
    "angry": "red",
    "fearful": "purple"
}

# Plot PCA-reduced activations
print("Plotting PCA projection...")
plt.figure(figsize=(8, 6))
for i, emotion in enumerate(emotions[:len(prompts)]):
    plt.scatter(reduced[i, 0], reduced[i, 1], label=emotion if i < 4 else "", color=colors[emotion], s=100)

plt.legend()
plt.title("Prompt Activations in Layer 2 (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show(block=True)  # Ensure plot displays properly

print("\nScript complete. All plots displayed successfully.")

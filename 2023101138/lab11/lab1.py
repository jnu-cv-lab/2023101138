import torch
import math
import matplotlib.pyplot as plt

#mission1
def sinusoidal_pos_encoding(seq_len: int, d_model: int):
    assert d_model % 2 == 0, "Dimension must be even"
    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, 1::2] = torch.cos(pos * div_term)
    return pos_enc

#mission2
def rotate_2d(x: torch.Tensor, theta: float):
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x1, x2 = x[0], x[1]
    x_rot1 = x1 * cos_t - x2 * sin_t
    x_rot2 = x1 * sin_t + x2 * cos_t
    return torch.tensor([x_rot1, x_rot2])

#mission3
def rope(x: torch.Tensor, pos: torch.Tensor):
    seq_len, d_model = x.shape
    assert d_model % 2 == 0, "Dimension must be even"
    d = d_model // 2

    theta = torch.exp(torch.arange(0, d, 1) * (-math.log(10000.0) / d))
    cos = torch.cos(pos.unsqueeze(1) * theta)
    sin = torch.sin(pos.unsqueeze(1) * theta)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    x_rot1 = x1 * cos - x2 * sin
    x_rot2 = x1 * sin + x2 * cos
    return torch.cat([x_rot1, x_rot2], dim=-1)

#mission4
print("===== 4. Compare E+pos and RoPE Input Mode =====")
seq_len = 8
d_model = 64
E = torch.randn(seq_len, d_model)
pos = torch.arange(0, seq_len)

# E + Sinusoidal Position Encoding
pos_enc = sinusoidal_pos_encoding(seq_len, d_model)
E_plus_pos = E + pos_enc
print(f"E+pos mode: Add embedding and position encoding directly, output shape: {E_plus_pos.shape}")

# RoPE mode
E_rope = rope(E, pos)
print(f"RoPE mode: Embed fused with position via 2D rotation, output shape: {E_rope.shape}")

#mission5
print("\n===== 5. Verify Relative Position Property of RoPE =====")
d_test = 4
x_base = torch.tensor([1.0, 2.0, 3.0, 4.0])

# Same relative distance (k=3)
p1, p2 = 2, 5
p3, p4 = 4, 7

x_p1 = rope(x_base.unsqueeze(0), torch.tensor([p1]))[0]
x_p2 = rope(x_base.unsqueeze(0), torch.tensor([p2]))[0]
x_p3 = rope(x_base.unsqueeze(0), torch.tensor([p3]))[0]
x_p4 = rope(x_base.unsqueeze(0), torch.tensor([p4]))[0]

dot1 = torch.dot(x_p1, x_p2).item()
dot2 = torch.dot(x_p3, x_p4).item()
print(f"Relative distance = 3, Position (2,5) dot product: {dot1:.4f}")
print(f"Relative distance = 3, Position (4,7) dot product: {dot2:.4f}")
print("Conclusion: Dot products are equal with same relative distance, RoPE has relative position invariance.")

# Different relative distance
p_a, p_b = 2, 3
x_a = rope(x_base.unsqueeze(0), torch.tensor([p_a]))[0]
x_b = rope(x_base.unsqueeze(0), torch.tensor([p_b]))[0]
dot_diff = torch.dot(x_a, x_b).item()
print(f"Relative distance = 1, Position (2,3) dot product: {dot_diff:.4f}")
print("Dot product changes obviously with different relative distances, RoPE can distinguish relative positions.")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(pos_enc.numpy(), cmap="YlGnBu")
plt.title("Sinusoidal Position Encoding")
plt.xlabel("Dimension")
plt.ylabel("Position")

plt.subplot(1,2,2)
dot_list = [torch.dot(rope(E[i:i+1], torch.tensor([i]))[0], E[0]).item() for i in range(seq_len)]
plt.plot(range(seq_len), dot_list)
plt.title("RoPE Dot Product vs Relative Position")
plt.xlabel("Relative Position")
plt.ylabel("Dot Product")

plt.tight_layout()
#plt.savefig("rope_pos_compare.png", dpi=300, bbox_inches="tight")
plt.show()